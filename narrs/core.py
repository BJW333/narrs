from __future__ import annotations

import math
import os
import pickle
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .clustering import dbscan_style_cluster, find_nearby_points
from .grid import choose_points_inside_region
from .samplers import BaseSampler, HybridSampler
from .validation import validate_setup
from .types import (
    ConfidenceReport,
    ContextStats,
    MetricRule,
    NARRSConfig,
    ObjectiveResult,
    ParameterSpec,
    ParamDict,
    PointResult,
    Region,
    SearchContext,
)
from .utils import (
    average_metric_profile,
    confidence_interval,
    confidence_interval_width,
    deflated_holdout_threshold,
    estimate_weighted_grid_spacing,
    percentile,
    region_density,
    region_ranges,
    region_width,
    safe_mean,
    safe_std,
    size_bonus,
    unique_points,
    violates_hard_metric_rules,
    weighted_geometric_center,
    weighted_normalized_distance,
    weighted_metric_score,
    normalize_metrics,
)


ObjectiveFn = Callable[[ParamDict, SearchContext, int], ObjectiveResult]
ConstraintFn = Callable[[ParamDict], bool]


def _evaluate_one_point(
    point: ParamDict,
    round_index: int,
    search_contexts: Sequence[SearchContext],
    metric_rules: Dict[str, MetricRule],
    metric_weights: Dict[str, float],
    repetitions_per_point: int,
    objective_function: ObjectiveFn,
) -> Tuple[PointResult, int, bool]:
    """
    Full evaluation of one candidate across all contexts and repetitions.
    Module-level and side-effect-free so it can run in a worker process.
    Returns (result, objective_calls_made, valid). Mirrors the serial
    per-point body in _evaluate_candidates exactly.
    """
    result = PointResult(parameter_point=point, round_index=round_index)
    calls = 0
    valid = True

    for context in search_contexts:
        if not valid:
            break

        ctx_stats = ContextStats(context_name=context.name)

        for rep in range(repetitions_per_point):
            obj = objective_function(point, context, rep)
            calls += 1

            if obj.validity_status != "valid":
                valid = False
                result.validity_status = "invalid"
                break

            if violates_hard_metric_rules(obj.performance_metrics, metric_rules):
                valid = False
                result.validity_status = "invalid"
                break

            normalized = normalize_metrics(obj.performance_metrics, metric_rules)
            score = weighted_metric_score(normalized, metric_weights)

            ctx_stats.scores.append(score)
            ctx_stats.metric_results.append(obj.performance_metrics)

            result.all_scores.append(score)
            result.all_metric_results.append(obj.performance_metrics)
            result.total_sample_count += obj.sample_count

        ctx_stats.mean = safe_mean(ctx_stats.scores)
        ctx_stats.noise = safe_std(ctx_stats.scores)
        ctx_stats.worst_case = percentile(ctx_stats.scores, 10)

        result.context_results.append(ctx_stats)

    return result, calls, valid


class NoiseAwareRobustRegionSearch:
    """
    Universal implementation of Noise-Aware Robust Region Search.

    The optimizer searches for stable parameter regions, not isolated best points.

    Usage:

        optimizer = NoiseAwareRobustRegionSearch(
            objective_function=my_objective,
            parameter_space=[ParameterSpec("x", -5, 5)],
            search_contexts=[SearchContext("ctx1")],
            holdout_contexts=[SearchContext("holdout")],
            metric_rules={"score": MetricRule(good_value=1, bad_value=0)},
            metric_weights={"score": 1.0},
        )

        result = optimizer.run()
    """

    def __init__(
        self,
        objective_function: ObjectiveFn,
        parameter_space: Sequence[ParameterSpec],
        search_contexts: Sequence[SearchContext],
        holdout_contexts: Sequence[SearchContext],
        metric_rules: Dict[str, MetricRule],
        metric_weights: Dict[str, float],
        parameter_constraints: Optional[Sequence[ConstraintFn]] = None,
        config: Optional[NARRSConfig] = None,
        sampler: Optional[BaseSampler] = None,
        validate: bool = True,
    ) -> None:
        if validate:
            validate_setup(
                parameter_space,
                search_contexts,
                holdout_contexts,
                metric_rules,
                metric_weights,
            )
        self.objective_function = objective_function
        self.parameter_specs = {p.name: p for p in parameter_space}
        self.search_contexts = list(search_contexts)
        self.holdout_contexts = list(holdout_contexts)
        self.metric_rules = dict(metric_rules)
        self.metric_weights = dict(metric_weights)
        self.parameter_constraints = list(parameter_constraints or [])
        self.config = config or NARRSConfig()
        self.rng = random.Random(self.config.random_seed)
        self.sampler = sampler or HybridSampler(
            initial_strategy="latin_hypercube",
            exploration_strategy="latin_hypercube",
        )

        self.total_evaluations = 0
        self.holdout_evaluations = 0
        self.all_discovered_regions: List[Region] = []
        self.all_point_results: List[PointResult] = []

        self.rejected_invalid_points: List[Any] = []
        self.rejected_noisy_points: List[PointResult] = []
        self.rejected_low_evidence_points: List[PointResult] = []
        self.rejected_spikes: List[PointResult] = []
        self.rejected_fragile_peaks: List[PointResult] = []
        self.rejected_small_regions: List[Region] = []
        self.rejected_isolated_points: List[PointResult] = []

        self.early_stopped = False
        self.compute_budget_reached = False
        self.search_rounds_completed = 0
        self.last_neighbor_radius = 0.0
        self.last_min_points_per_region = 0
        self._previous_best_region_ranges: Optional[Dict[str, Tuple[float, float]]] = None
        self._stable_boundary_rounds = 0

    def run(self) -> Dict[str, Any]:
        candidate_points = self.sampler.generate_initial(
            self.parameter_specs,
            self.config,
            self.rng,
        )

        previous_best_region_score: Optional[float] = None
        low_improvement_rounds = 0

        best_regions: List[Region] = []

        for round_index in range(self.config.search_rounds):
            self.search_rounds_completed = round_index + 1

            if self.total_evaluations >= self.config.total_compute_budget:
                self.compute_budget_reached = True
                break

            spacing = estimate_weighted_grid_spacing(candidate_points, self.parameter_specs)
            neighbor_radius = self.config.neighbor_radius_multiplier * spacing
            self.last_neighbor_radius = neighbor_radius

            min_points = self.config.minimum_points_per_region
            if min_points is None:
                min_points = max(3, len(self.parameter_specs) + 1)
            self.last_min_points_per_region = min_points

            if self.config.verbose:
                print(
                    f"[NARRS] Round {round_index + 1}/{self.config.search_rounds} | "
                    f"candidates={len(candidate_points)} | radius={neighbor_radius:.4f}"
                )

            results = self._evaluate_candidates(candidate_points, round_index)
            self.all_point_results.extend(results)

            self._apply_minimum_evidence_checks(results)
            self._adaptive_retest(results)
            self._compute_noise_adjusted_scores(results)

            self._compute_neighborhood_stats(results, neighbor_radius)
            self._detect_spikes(results)
            self._compute_context_stability(results)
            self._compute_robust_scores(results)
            surviving_points = self._reject_bad_points(results)

            robust_regions, isolated = dbscan_style_cluster(
                surviving_points,
                self.parameter_specs,
                neighbor_radius,
                min_points,
            )
            self.rejected_isolated_points.extend(isolated)

            surviving_regions = self._reject_and_score_regions(robust_regions)

            if not surviving_regions:
                if self.config.verbose:
                    print("[NARRS] No surviving regions this round. Restarting with fresh global candidates.")

                candidate_points = self.sampler.generate_initial(
                    self.parameter_specs,
                    self.config,
                    self.rng,
                )
                continue

            best_regions = self._select_best_regions(surviving_regions)
            self.all_discovered_regions.extend(best_regions)

            current_best_score = max(r.region_score for r in best_regions)

            if previous_best_region_score is not None:
                denom = abs(previous_best_region_score) if previous_best_region_score != 0 else 1.0
                improvement = (current_best_score - previous_best_region_score) / denom

                if improvement < self.config.early_stopping_minimum_improvement:
                    low_improvement_rounds += 1
                else:
                    low_improvement_rounds = 0

            current_ranges = region_ranges(best_regions[0], self.parameter_specs)
            if self._region_boundaries_are_stable(current_ranges):
                self._stable_boundary_rounds += 1
            else:
                self._stable_boundary_rounds = 0
            self._previous_best_region_ranges = current_ranges

            high_confidence_search_region = (
                best_regions[0].region_score >= self.config.high_confidence_region_score
                and best_regions[0].region_noise <= self.config.high_confidence_max_region_noise
                and best_regions[0].region_context_stability <= self.config.high_confidence_max_context_instability
            )

            if (
                low_improvement_rounds >= self.config.early_stopping_patience
                or self._stable_boundary_rounds >= self.config.early_stopping_patience
                or (high_confidence_search_region and low_improvement_rounds > 0)
            ):
                self.early_stopped = True
                break

            previous_best_region_score = current_best_score

            if round_index < self.config.search_rounds - 1:
                candidate_points = self._generate_next_candidates(
                    best_regions,
                    len(candidate_points),
                    round_index,
                )

        return self._finalize()

    def _region_boundaries_are_stable(self, current_ranges: Dict[str, Tuple[float, float]]) -> bool:
        """Return True when the current best region's boundaries barely changed."""
        if self._previous_best_region_ranges is None:
            return False

        max_change = 0.0
        for name, spec in self.parameter_specs.items():
            old_lo, old_hi = self._previous_best_region_ranges.get(name, (spec.min_value, spec.max_value))
            new_lo, new_hi = current_ranges.get(name, (spec.min_value, spec.max_value))
            span = spec.max_value - spec.min_value or 1.0
            max_change = max(max_change, abs(new_lo - old_lo) / span, abs(new_hi - old_hi) / span)

        return max_change <= self.config.region_boundary_stability_tolerance

    def _point_is_inside_previous_promising_region(self, point: ParamDict) -> bool:
        if self._previous_best_region_ranges is None:
            return False
        for name, (lo, hi) in self._previous_best_region_ranges.items():
            if point[name] < lo or point[name] > hi:
                return False
        return True

    def _point_is_near_previous_region_boundary(self, point: ParamDict) -> bool:
        if self._previous_best_region_ranges is None:
            return False
        for name, spec in self.parameter_specs.items():
            lo, hi = self._previous_best_region_ranges[name]
            span = spec.max_value - spec.min_value or 1.0
            tolerance = self.config.boundary_retest_tolerance * span
            near_lower = abs(point[name] - lo) <= tolerance
            near_upper = abs(point[name] - hi) <= tolerance
            inside_padded = (lo - tolerance) <= point[name] <= (hi + tolerance)
            if inside_padded and (near_lower or near_upper):
                return True
        return False

    def _parameter_point_is_valid(self, point: ParamDict) -> bool:
        for fn in self.parameter_constraints:
            try:
                if not fn(point):
                    return False
            except Exception:
                return False
        return True

    def _evaluate_candidates(self, candidate_points: Sequence[ParamDict], round_index: int) -> List[PointResult]:
        if self.config.n_jobs != 1:
            parallel = self._evaluate_candidates_parallel(candidate_points, round_index)
            if parallel is not None:
                return parallel
            # objective wasn't picklable; fall through to the serial path below

        results: List[PointResult] = []

        for point in candidate_points:
            if self.total_evaluations >= self.config.total_compute_budget:
                self.compute_budget_reached = True
                break

            if not self._parameter_point_is_valid(point):
                self.rejected_invalid_points.append(point)
                continue

            result = PointResult(parameter_point=point, round_index=round_index)

            valid = True

            for context in self.search_contexts:
                if not valid:
                    break

                ctx_stats = ContextStats(context_name=context.name)

                for rep in range(self.config.repetitions_per_point):
                    if self.total_evaluations >= self.config.total_compute_budget:
                        self.compute_budget_reached = True
                        break

                    obj = self.objective_function(point, context, rep)
                    self.total_evaluations += 1

                    if obj.validity_status != "valid":
                        valid = False
                        result.validity_status = "invalid"
                        self.rejected_invalid_points.append(point)
                        break

                    if violates_hard_metric_rules(obj.performance_metrics, self.metric_rules):
                        valid = False
                        result.validity_status = "invalid"
                        self.rejected_invalid_points.append(point)
                        break

                    normalized = normalize_metrics(obj.performance_metrics, self.metric_rules)
                    score = weighted_metric_score(normalized, self.metric_weights)

                    ctx_stats.scores.append(score)
                    ctx_stats.metric_results.append(obj.performance_metrics)

                    result.all_scores.append(score)
                    result.all_metric_results.append(obj.performance_metrics)
                    result.total_sample_count += obj.sample_count

                ctx_stats.mean = safe_mean(ctx_stats.scores)
                ctx_stats.noise = safe_std(ctx_stats.scores)
                ctx_stats.worst_case = percentile(ctx_stats.scores, 10)

                result.context_results.append(ctx_stats)

            if not valid or not result.all_scores:
                result.validity_status = "invalid"
                result.reject("invalid")
                results.append(result)
                continue

            self._update_point_summary(result)
            results.append(result)

        return results

    def _evaluate_candidates_parallel(
        self, candidate_points: Sequence[ParamDict], round_index: int
    ) -> Optional[List[PointResult]]:
        """
        Evaluate candidates across worker processes. Returns None to signal the
        caller should fall back to serial (objective not picklable). Upper-bound
        budgeting guarantees the run never exceeds total_compute_budget.
        """
        try:
            pickle.dumps(self.objective_function)
        except Exception:
            warnings.warn(
                "n_jobs > 1 but objective_function is not picklable "
                "(define it at module level to enable parallelism); running serially.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        cost_per_point = max(1, len(self.search_contexts) * self.config.repetitions_per_point)
        remaining = self.config.total_compute_budget - self.total_evaluations
        if remaining <= 0:
            self.compute_budget_reached = True
            return []

        to_run = list(candidate_points)
        max_points = remaining // cost_per_point
        if max_points < len(to_run):
            to_run = to_run[:max_points]
            self.compute_budget_reached = True
        if not to_run:
            self.compute_budget_reached = True
            return []

        valid_points: List[ParamDict] = []
        for point in to_run:
            if self._parameter_point_is_valid(point):
                valid_points.append(point)
            else:
                self.rejected_invalid_points.append(point)

        max_workers = os.cpu_count() if self.config.n_jobs < 0 else self.config.n_jobs
        results: List[PointResult] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_one_point,
                    point,
                    round_index,
                    self.search_contexts,
                    self.metric_rules,
                    self.metric_weights,
                    self.config.repetitions_per_point,
                    self.objective_function,
                )
                for point in valid_points
            ]
            collected = [f.result() for f in futures]

        for result, calls, valid in collected:
            self.total_evaluations += calls
            if not valid or not result.all_scores:
                result.validity_status = "invalid"
                result.reject("invalid")
                self.rejected_invalid_points.append(result.parameter_point)
                results.append(result)
                continue
            self._update_point_summary(result)
            results.append(result)

        return results

    def _update_point_summary(self, result: PointResult) -> None:
        result.point_mean = safe_mean(result.all_scores)
        result.point_noise = safe_std(result.all_scores)
        result.point_worst_case = percentile(result.all_scores, 10)
        result.point_best_case = percentile(result.all_scores, 90)
        result.confidence_interval = confidence_interval(result.all_scores)

    def _apply_minimum_evidence_checks(self, results: Sequence[PointResult]) -> None:
        for r in results:
            if r.validity_status != "valid":
                continue

            if r.total_sample_count < self.config.minimum_sample_size:
                r.add_label("low evidence")
                self.rejected_low_evidence_points.append(r)

            if confidence_interval_width(r.confidence_interval) > self.config.maximum_confidence_interval_width:
                r.add_label("uncertain")

            if r.point_noise > self.config.maximum_allowed_noise:
                r.add_label("too noisy")
                self.rejected_noisy_points.append(r)

    def _adaptive_retest(self, results: Sequence[PointResult]) -> None:
        valid_results = [r for r in results if r.validity_status == "valid" and r.all_scores]
        if not valid_results:
            return

        sorted_by_mean = sorted(valid_results, key=lambda r: r.point_mean, reverse=True)
        top_cutoff_index = max(1, int(0.20 * len(sorted_by_mean)))
        top_ids = {id(r) for r in sorted_by_mean[:top_cutoff_index]}

        for r in valid_results:
            if self.total_evaluations >= self.config.total_compute_budget:
                self.compute_budget_reached = True
                return

            extra = 0

            if id(r) in top_ids:
                extra += 2

            if "uncertain" in r.labels:
                extra += 2

            if r.point_noise > 0.5 * self.config.maximum_allowed_noise:
                extra += 2

            if self._point_is_inside_previous_promising_region(r.parameter_point):
                extra += 2

            if self._point_is_near_previous_region_boundary(r.parameter_point):
                extra += 1

            extra = min(extra, self.config.max_extra_repetitions)

            current_reps = len(r.all_scores)
            allowed_extra = max(0, self.config.max_total_repetitions_per_point - current_reps)
            extra = min(extra, allowed_extra)

            if extra <= 0:
                continue

            for rep in range(extra):
                if self.total_evaluations >= self.config.total_compute_budget:
                    self.compute_budget_reached = True
                    break

                # Retest across rotating contexts to avoid only reinforcing one context.
                context = self.search_contexts[rep % len(self.search_contexts)]
                seed = 10_000 + len(r.all_scores) + rep

                obj = self.objective_function(r.parameter_point, context, seed)
                self.total_evaluations += 1

                if obj.validity_status != "valid":
                    r.validity_status = "invalid"
                    r.reject("invalid during adaptive retest")
                    break

                if violates_hard_metric_rules(obj.performance_metrics, self.metric_rules):
                    r.validity_status = "invalid"
                    r.reject("hard metric rule violation during adaptive retest")
                    break

                normalized = normalize_metrics(obj.performance_metrics, self.metric_rules)
                score = weighted_metric_score(normalized, self.metric_weights)

                r.all_scores.append(score)
                r.all_metric_results.append(obj.performance_metrics)
                r.total_sample_count += obj.sample_count

                # Also append to matching context stats.
                for ctx_stat in r.context_results:
                    if ctx_stat.context_name == context.name:
                        ctx_stat.scores.append(score)
                        ctx_stat.metric_results.append(obj.performance_metrics)
                        ctx_stat.mean = safe_mean(ctx_stat.scores)
                        ctx_stat.noise = safe_std(ctx_stat.scores)
                        ctx_stat.worst_case = percentile(ctx_stat.scores, 10)
                        break

            self._update_point_summary(r)

    def _compute_noise_adjusted_scores(self, results: Sequence[PointResult]) -> None:
        for r in results:
            r.noise_adjusted_score = (
                r.point_mean
                - self.config.noise_penalty_weight * r.point_noise
                + self.config.worst_case_weight * r.point_worst_case
            )

    def _compute_neighborhood_stats(self, results: Sequence[PointResult], neighbor_radius: float) -> None:
        for r in results:
            if r.validity_status != "valid":
                continue

            neighbors = find_nearby_points(r, results, self.parameter_specs, neighbor_radius)
            r.neighbors = neighbors

            neighbor_scores = [n.noise_adjusted_score for n in neighbors if n.validity_status == "valid"]

            if not neighbor_scores:
                r.add_label("weak neighborhood evidence")
                r.neighborhood_mean = 0.0
                r.neighborhood_noise = 0.0
                r.neighborhood_worst_case = 0.0
                continue

            r.neighborhood_mean = safe_mean(neighbor_scores)
            r.neighborhood_noise = safe_std(neighbor_scores)
            r.neighborhood_worst_case = min(neighbor_scores)

            r.spike_size = r.noise_adjusted_score - r.neighborhood_mean
            if r.spike_size > 0:
                r.spike_penalty = self.config.spike_penalty_weight * r.spike_size
            else:
                r.spike_penalty = 0.0

    def _detect_spikes(self, results: Sequence[PointResult]) -> None:
        if not results:
            return

        valid = [r for r in results if r.validity_status == "valid"]
        if not valid:
            return

        mean_score = safe_mean([r.noise_adjusted_score for r in valid])
        std_score = safe_std([r.noise_adjusted_score for r in valid])

        for r in valid:
            high_score = r.noise_adjusted_score > mean_score + std_score
            much_lower_neighbors = r.neighborhood_mean < r.noise_adjusted_score - std_score
            nearby_fail = r.neighborhood_mean < self.config.minimum_neighborhood_mean

            if high_score and much_lower_neighbors and nearby_fail:
                r.add_label("isolated spike")
                self.rejected_spikes.append(r)

            if r.neighborhood_noise > self.config.maximum_neighborhood_noise:
                r.add_label("fragile peak")
                self.rejected_fragile_peaks.append(r)

    def _compute_context_stability(self, results: Sequence[PointResult]) -> None:
        for r in results:
            if r.validity_status != "valid":
                continue

            context_means = [c.mean for c in r.context_results if c.scores]
            r.context_instability = safe_std(context_means)
            r.context_worst_case = min(context_means) if context_means else 0.0
            r.context_failure_count = sum(
                1 for m in context_means if m < self.config.minimum_acceptable_worst_case_score
            )

            if len(context_means) > 1:
                best_index = max(range(len(context_means)), key=lambda i: context_means[i])
                best_context = context_means[best_index]
                other_contexts = [
                    value for i, value in enumerate(context_means)
                    if i != best_index
                ]
                other_mean = safe_mean(other_contexts)

                if (
                    best_context >= self.config.minimum_acceptable_worst_case_score
                    and other_mean < self.config.minimum_acceptable_worst_case_score
                    and (best_context - other_mean) >= self.config.single_context_overfit_gap
                ):
                    r.add_label("single context overfit")

    def _compute_robust_scores(self, results: Sequence[PointResult]) -> None:
        for r in results:
            r.robust_score = (
                r.neighborhood_mean
                + self.config.worst_case_weight * r.context_worst_case
                - self.config.noise_penalty_weight * r.point_noise
                - self.config.neighborhood_penalty_weight * r.neighborhood_noise
                - self.config.context_penalty_weight * r.context_instability
                - r.spike_penalty
            )

    def _reject_bad_points(self, results: Sequence[PointResult]) -> List[PointResult]:
        surviving: List[PointResult] = []

        for r in results:
            if r.validity_status != "valid":
                r.reject("invalid")
            if r.total_sample_count < self.config.minimum_sample_size:
                r.reject("low evidence")
            if r.point_noise > self.config.maximum_allowed_noise:
                r.reject("too noisy")
            if confidence_interval_width(r.confidence_interval) > self.config.maximum_confidence_interval_width:
                r.reject("confidence interval too wide")
            if r.neighborhood_mean < self.config.minimum_neighborhood_mean:
                r.reject("neighborhood mean too low")
            if r.neighborhood_noise > self.config.maximum_neighborhood_noise:
                r.reject("neighborhood noise too high")
            if r.spike_penalty > self.config.maximum_spike_penalty:
                r.reject("spike penalty too high")
            if "isolated spike" in r.labels:
                r.reject("isolated spike")
            if "fragile peak" in r.labels:
                r.reject("fragile peak")
            if r.context_worst_case < self.config.minimum_acceptable_worst_case_score:
                r.reject("context worst case too low")
            if r.context_failure_count > self.config.maximum_context_failure_count:
                r.reject("too many context failures")
            if "single context overfit" in r.labels:
                r.reject("single context overfit")
            if "weak neighborhood evidence" in r.labels:
                r.reject("weak neighborhood evidence")

            if not r.rejected:
                surviving.append(r)

        return surviving

    def _reject_and_score_regions(self, regions: Sequence[Region]) -> List[Region]:
        surviving = []

        for region in regions:
            self._score_region(region)

            if len(region.points) < self.config.minimum_region_size:
                region.reject("region too small")

            if region.region_width < self.config.minimum_region_width:
                region.reject("region too narrow")

            if region.region_density < self.config.minimum_region_density:
                region.reject("region too thin")

            if region.rejected:
                self.rejected_small_regions.append(region)
            else:
                surviving.append(region)

        return surviving

    def _score_region(self, region: Region) -> None:
        robust_scores = [p.robust_score for p in region.points]
        point_noises = [p.point_noise for p in region.points]
        context_instabilities = [p.context_instability for p in region.points]

        region.region_average_score = safe_mean(robust_scores)
        region.region_stability = safe_std(robust_scores)
        region.region_worst_case = min(robust_scores) if robust_scores else 0.0
        region.region_noise = safe_mean(point_noises)
        region.region_context_stability = safe_mean(context_instabilities)
        region.region_width = region_width(region, self.parameter_specs)
        region.region_density = region_density(region, self.parameter_specs)
        region.region_metric_profile = average_metric_profile(
            [m for p in region.points for m in p.all_metric_results]
        )

        region.region_score = (
            region.region_average_score
            + self.config.worst_case_weight * region.region_worst_case
            + self.config.region_size_weight * size_bonus(region.region_width)
            - region.region_stability
            - region.region_noise
            - region.region_context_stability
        )

    def _select_best_regions(self, regions: Sequence[Region]) -> List[Region]:
        sorted_regions = sorted(regions, key=lambda r: r.region_score, reverse=True)
        if not sorted_regions:
            return []

        n_fraction = max(1, int(math.ceil(len(sorted_regions) * self.config.top_region_fraction)))
        n = min(max(n_fraction, 1), self.config.max_regions_to_refine, len(sorted_regions))
        return sorted_regions[:n]

    def _generate_next_candidates(
        self,
        best_regions: Sequence[Region],
        current_candidate_count: int,
        round_index: int,
    ) -> List[ParamDict]:
        return self.sampler.generate_next(
            best_regions=best_regions,
            parameter_specs=self.parameter_specs,
            config=self.config,
            rng=self.rng,
            current_candidate_count=current_candidate_count,
            round_index=round_index,
        )

    def _finalize(self) -> Dict[str, Any]:
        candidate_regions = list(self.all_discovered_regions)

        if not candidate_regions:
            report = self._build_confidence_report(None, [], None, [])
            return {
                "robust_regions": [],
                "best_region": None,
                "recommended_center": None,
                "optional_region_ensemble": [],
                "confidence_report": report,
                "message": "No robust region found during search.",
            }

        self._holdout_validate(candidate_regions)

        surviving = [r for r in candidate_regions if not r.rejected]

        if not surviving:
            report = self._build_confidence_report(None, [], None, [])
            return {
                "robust_regions": [],
                "best_region": None,
                "recommended_center": None,
                "optional_region_ensemble": [],
                "confidence_report": report,
                "message": "No robust region survived holdout validation.",
            }

        for r in surviving:
            # Confidence balances search region score and holdout survival.
            r.confidence_score = (
                r.region_score
                + r.holdout_mean
                + self.config.worst_case_weight * r.holdout_worst_case
                - r.holdout_noise
                - confidence_interval_width(r.holdout_confidence_interval)
            )

        robust_regions = sorted(surviving, key=lambda r: r.confidence_score, reverse=True)
        best_region = robust_regions[0]

        recommended_center = self._choose_recommended_center(best_region)
        ensemble = self._create_region_ensemble(best_region)

        report = self._build_confidence_report(best_region, robust_regions, recommended_center, ensemble)

        return {
            "robust_regions": robust_regions,
            "best_region": best_region,
            "recommended_center": recommended_center,
            "optional_region_ensemble": ensemble,
            "confidence_report": report,
        }

    def _holdout_validate(self, regions: Sequence[Region]) -> None:
        for region in regions:
            test_points = choose_points_inside_region(
                region,
                self.parameter_specs,
                count=self.config.holdout_points_per_region,
                rng=self.rng,
            )

            for point in test_points:
                for ctx_index, context in enumerate(self.holdout_contexts):
                    obj = self.objective_function(point, context, 100_000 + ctx_index)
                    self.holdout_evaluations += 1

                    if obj.validity_status != "valid":
                        continue

                    if violates_hard_metric_rules(obj.performance_metrics, self.metric_rules):
                        continue

                    normalized = normalize_metrics(obj.performance_metrics, self.metric_rules)
                    score = weighted_metric_score(normalized, self.metric_weights)

                    region.holdout_scores.append(score)
                    region.holdout_metric_results.append(obj.performance_metrics)

            region.holdout_mean = safe_mean(region.holdout_scores)
            region.holdout_noise = safe_std(region.holdout_scores)
            region.holdout_worst_case = percentile(region.holdout_scores, 10)
            region.holdout_confidence_interval = confidence_interval(region.holdout_scores)
            region.holdout_metric_profile = average_metric_profile(region.holdout_metric_results)

            if region.holdout_mean < self.config.minimum_acceptable_holdout_score:
                region.reject("holdout mean too low")

            if region.holdout_worst_case < self.config.minimum_acceptable_worst_case_score:
                region.reject("holdout worst case too low")

            if region.holdout_noise > self.config.maximum_holdout_noise:
                # Do not automatically reject; the final confidence score subtracts holdout_noise.
                # This records the warning promised by the pseudocode: high holdout noise lowers confidence.
                if "holdout noise too high" not in region.rejection_reasons:
                    region.rejection_reasons.append("holdout noise too high")

            if confidence_interval_width(region.holdout_confidence_interval) > self.config.maximum_confidence_interval_width:
                region.reject("holdout confidence interval too wide")

    def _choose_recommended_center(self, region: Region) -> ParamDict:
        weighted_center = weighted_geometric_center(region, self.parameter_specs)

        def score_point(p: PointResult) -> float:
            # Prefer stable central points, not necessarily the highest point.
            distance_to_center = weighted_normalized_distance(
                p.parameter_point,
                weighted_center,
                self.parameter_specs,
            )

            return (
                p.robust_score
                - p.point_noise
                - p.context_instability
                - 0.1 * distance_to_center
            )

        best = max(region.points, key=score_point)
        return best.parameter_point

    def _create_region_ensemble(self, region: Region) -> List[ParamDict]:
        sorted_points = sorted(
            region.points,
            key=lambda p: (
                p.robust_score
                - p.point_noise
                - p.context_instability
            ),
            reverse=True,
        )

        points = [p.parameter_point for p in sorted_points[: self.config.ensemble_size]]
        return unique_points(points)

    def _summarize_region(self, region: Region) -> Dict[str, Any]:
        """Return the explicit per-region report fields promised by the pseudocode."""
        point_scores = [p.robust_score for p in region.points]
        point_noise = [p.point_noise for p in region.points]
        neighborhood_means = [p.neighborhood_mean for p in region.points]
        neighborhood_noise = [p.neighborhood_noise for p in region.points]
        context_instability = [p.context_instability for p in region.points]
        context_worst_cases = [p.context_worst_case for p in region.points]

        return {
            "parameter_ranges": region_ranges(region, self.parameter_specs),
            "region_score": region.region_score,
            "confidence_score": region.confidence_score,
            "average_performance": region.region_average_score,
            "original_metric_profile": region.region_metric_profile,
            "noise_level": region.region_noise,
            "confidence_interval": confidence_interval(point_scores),
            "neighborhood_stability": safe_mean(neighborhood_noise),
            "neighborhood_mean": safe_mean(neighborhood_means),
            "context_stability": region.region_context_stability,
            "context_worst_case": min(context_worst_cases) if context_worst_cases else 0.0,
            "holdout_performance": region.holdout_mean,
            "holdout_noise": region.holdout_noise,
            "holdout_confidence_interval": region.holdout_confidence_interval,
            "holdout_metric_profile": region.holdout_metric_profile,
            "worst_case_performance": region.region_worst_case,
            "weighted_region_width": region.region_width,
            "region_density": region.region_density,
            "number_of_points": len(region.points),
            "rejection_reasons": list(region.rejection_reasons),
        }

    def _build_confidence_report(
        self,
        best_region: Optional[Region],
        robust_regions: Sequence[Region],
        recommended_center: Optional[ParamDict],
        ensemble: Sequence[ParamDict],
    ) -> ConfidenceReport:
        warning_signs = []

        if self.compute_budget_reached:
            warning_signs.append("Compute budget was reached.")

        if not robust_regions:
            warning_signs.append("No robust regions survived.")

        if best_region and best_region.holdout_noise > best_region.holdout_mean:
            warning_signs.append("Best region has high holdout noise relative to mean.")

        if best_region and best_region.holdout_noise > self.config.maximum_holdout_noise:
            warning_signs.append("Best region exceeds maximum_holdout_noise; confidence was lowered.")

        if best_region and best_region.region_width < self.config.minimum_region_width * 2:
            warning_signs.append("Best region is only slightly wider than minimum width.")

        if best_region:
            ranges = region_ranges(best_region, self.parameter_specs)
            best_region_summary = self._summarize_region(best_region)
        else:
            ranges = {}
            best_region_summary = {}

        surviving_region_summaries = [self._summarize_region(r) for r in robust_regions]

        raw = {
            "best_region_summary": best_region_summary,
            "surviving_region_summaries": surviving_region_summaries,
            "all_surviving_robust_regions_count": len(robust_regions),
        }

        confidence_rating = "low"
        # Selection accounting: the best region was chosen after evaluating this
        # many points, so a "high" rating must survive a best-of-N (DSR-style)
        # deflation. Use len(robust_regions) instead for a lighter penalty.
        n_trials_considered = len(self.all_point_results)
        deflated_bar = 0.0
        selection_penalty = 0.0
        if best_region:
            n_holdout = len(best_region.holdout_scores)
            have_enough_holdout = n_holdout >= self.config.minimum_holdout_evidence

            if not self.holdout_contexts:
                # No out-of-sample validation was possible. Cap at "medium", and
                # only grant it if the in-sample region itself is strong.
                warning_signs.append(
                    "No holdout contexts provided; confidence is not out-of-sample "
                    "validated and is capped at 'medium'."
                )
                if (
                    best_region.region_score >= self.config.high_confidence_region_score
                    and best_region.region_noise <= self.config.high_confidence_max_region_noise
                ):
                    confidence_rating = "medium"
            elif not have_enough_holdout:
                # Holdout contexts existed but produced too little valid evidence.
                # Zero-filled stats must NOT be trusted -> hold at "low".
                warning_signs.append(
                    f"Only {n_holdout} valid holdout score(s) (need "
                    f"{self.config.minimum_holdout_evidence}); confidence held at 'low'."
                )
            else:
                # Deflate the "high" bar for how many points we searched among.
                # Medium still clears the raw bar; high must clear the deflated one.
                base_bar = self.config.minimum_acceptable_holdout_score
                deflated_bar, selection_penalty = deflated_holdout_threshold(
                    base_bar,
                    best_region.holdout_noise,
                    n_holdout,
                    n_trials_considered,
                )
                meets_noise_and_ci = (
                    best_region.holdout_noise <= min(0.25, self.config.maximum_holdout_noise)
                    and confidence_interval_width(best_region.holdout_confidence_interval)
                    <= self.config.maximum_confidence_interval_width
                )
                if best_region.holdout_mean >= deflated_bar and meets_noise_and_ci:
                    confidence_rating = "high"
                elif best_region.holdout_mean >= base_bar:
                    confidence_rating = "medium"
                    if meets_noise_and_ci and best_region.holdout_mean < deflated_bar:
                        warning_signs.append(
                            f"Selection-adjusted (deflated) holdout bar not met: chose the best "
                            f"of {n_trials_considered} regions, raising the 'high' bar from "
                            f"{base_bar:.3f} to {deflated_bar:.3f} (holdout mean "
                            f"{best_region.holdout_mean:.3f}); rating reduced to 'medium'."
                        )

        return ConfidenceReport(
            best_region_parameter_ranges=ranges,
            recommended_center=recommended_center,
            optional_region_ensemble=list(ensemble),
            best_region_summary=best_region_summary,
            surviving_region_summaries=surviving_region_summaries,
            number_of_surviving_regions=len(robust_regions),
            number_of_tested_points=len(self.all_point_results),
            number_of_invalid_points=len(self.rejected_invalid_points),
            number_of_rejected_spikes=len(self.rejected_spikes),
            number_of_rejected_fragile_peaks=len(self.rejected_fragile_peaks),
            number_of_rejected_noisy_points=len(self.rejected_noisy_points),
            number_of_rejected_low_evidence_points=len(self.rejected_low_evidence_points),
            number_of_rejected_small_regions=len(self.rejected_small_regions),
            number_of_dbscan_rejected_isolated_points=len(self.rejected_isolated_points),
            search_rounds_completed=self.search_rounds_completed,
            early_stopped=self.early_stopped,
            compute_budget_reached=self.compute_budget_reached,
            search_evaluations=self.total_evaluations,
            holdout_evaluations=self.holdout_evaluations,
            total_objective_calls=self.total_evaluations + self.holdout_evaluations,
            parameter_weights_used={name: spec.weight for name, spec in self.parameter_specs.items()},
            neighbor_radius_used=self.last_neighbor_radius,
            dbscan_minimum_points_per_region=self.last_min_points_per_region,
            number_of_trials_considered=n_trials_considered,
            deflated_holdout_threshold=deflated_bar,
            holdout_selection_penalty=selection_penalty,
            warning_signs=warning_signs,
            final_confidence_rating=confidence_rating,
            raw=raw,
        )
