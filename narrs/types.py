from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


ParamDict = Dict[str, float]
MetricDict = Dict[str, float]


@dataclass(frozen=True)
class ParameterSpec:
    """
    Defines one numeric parameter.

    name:
        Parameter name.

    min_value / max_value:
        Search range.

    weight:
        Weight used by weighted normalized distance.
        Higher weight means differences in this parameter matter more.
    """
    name: str
    min_value: float
    max_value: float
    weight: float = 1.0

    def normalize(self, value: float) -> float:
        if self.max_value == self.min_value:
            return 0.0
        return (value - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, value: float) -> float:
        return self.min_value + value * (self.max_value - self.min_value)

    def clamp(self, value: float) -> float:
        return max(self.min_value, min(self.max_value, value))


@dataclass(frozen=True)
class MetricRule:
    """
    Rule for converting a raw metric into a 0-1 score.

    good_value:
        Value considered good.

    bad_value:
        Value considered bad.

    higher_is_better:
        True for metrics like accuracy/profit.
        False for metrics like drawdown/latency/failure rate.

    hard_min / hard_max:
        Optional hard rejection limits.

        Example:
            max_drawdown hard_max = 0.40
            latency hard_max = 200
    """
    good_value: float
    bad_value: float
    higher_is_better: bool = True
    hard_min: Optional[float] = None
    hard_max: Optional[float] = None

    def normalize(self, value: float) -> float:
        if self.good_value == self.bad_value:
            return 0.0

        # NaN and infinity must be caught before the clamp. Python's min/max
        # compare with NaN as False, so max(0.0, min(1.0, nan)) evaluates to 1.0
        # -- a failed calculation would silently score as the best possible
        # result and be ranked top. A metric that could not be computed is the
        # worst case, not the best.
        if value != value or math.isinf(value):
            return 0.0

        if self.higher_is_better:
            score = (value - self.bad_value) / (self.good_value - self.bad_value)
        else:
            score = (self.bad_value - value) / (self.bad_value - self.good_value)

        if score != score:
            return 0.0
        return max(0.0, min(1.0, score))

    def violates_hard_rule(self, value: float) -> bool:
        # A metric that is NaN or infinite is not evidence of anything. Treat it
        # as a hard violation so the observation is discarded rather than folded
        # into an average, where it would either poison or flatter the result.
        if value != value or math.isinf(value):
            return True
        if self.hard_min is not None and value < self.hard_min:
            return True
        if self.hard_max is not None and value > self.hard_max:
            return True
        return False


@dataclass
class ObjectiveResult:
    """
    Return this from your objective function.

    performance_metrics:
        Dict of raw metrics.

    sample_count:
        Amount of evidence this result represents.
        In trading, this might be number of trades.
        In ML, this might be number of validation samples.
        In simulation, this might be number of rollouts.

    validity_status:
        "valid" or "invalid".

    metadata:
        Optional extra information.
    """
    performance_metrics: MetricDict
    sample_count: int = 1
    validity_status: str = "valid"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchContext:
    """
    Represents one evaluation context.

    name:
        Human-readable context name.

    data:
        Anything your objective function needs.
    """
    name: str
    data: Any = None


@dataclass
class ContextStats:
    context_name: str
    scores: List[float] = field(default_factory=list)
    metric_results: List[MetricDict] = field(default_factory=list)
    mean: float = 0.0
    noise: float = 0.0
    worst_case: float = 0.0


@dataclass
class PointResult:
    parameter_point: ParamDict

    all_scores: List[float] = field(default_factory=list)
    all_metric_results: List[MetricDict] = field(default_factory=list)
    context_results: List[ContextStats] = field(default_factory=list)

    point_mean: float = 0.0
    point_noise: float = 0.0
    point_worst_case: float = 0.0
    point_best_case: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    total_sample_count: int = 0
    validity_status: str = "valid"

    noise_adjusted_score: float = 0.0

    neighbors: List["PointResult"] = field(default_factory=list)
    neighborhood_mean: float = 0.0
    neighborhood_noise: float = 0.0
    neighborhood_worst_case: float = 0.0

    spike_size: float = 0.0
    spike_penalty: float = 0.0

    context_instability: float = 0.0
    context_worst_case: float = 0.0
    context_failure_count: int = 0

    robust_score: float = 0.0

    labels: List[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reasons: List[str] = field(default_factory=list)

    round_index: int = 0

    def add_label(self, label: str) -> None:
        if label not in self.labels:
            self.labels.append(label)

    def reject(self, reason: str) -> None:
        self.rejected = True
        if reason not in self.rejection_reasons:
            self.rejection_reasons.append(reason)


@dataclass
class Region:
    points: List[PointResult]

    region_score: float = 0.0
    region_average_score: float = 0.0
    region_stability: float = 0.0
    region_worst_case: float = 0.0
    region_noise: float = 0.0
    region_context_stability: float = 0.0
    region_width: float = 0.0
    region_density: float = 0.0
    region_metric_profile: MetricDict = field(default_factory=dict)

    holdout_scores: List[float] = field(default_factory=list)
    holdout_metric_results: List[MetricDict] = field(default_factory=list)
    holdout_mean: float = 0.0
    holdout_noise: float = 0.0
    holdout_worst_case: float = 0.0
    holdout_confidence_interval: Tuple[float, float] = (0.0, 0.0)
    holdout_metric_profile: MetricDict = field(default_factory=dict)

    confidence_score: float = 0.0
    rejected: bool = False
    rejection_reasons: List[str] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.rejected = True
        if reason not in self.rejection_reasons:
            self.rejection_reasons.append(reason)


@dataclass
class ConfidenceReport:
    best_region_parameter_ranges: Dict[str, Tuple[float, float]]
    recommended_center: Optional[ParamDict]
    optional_region_ensemble: List[ParamDict]

    # Explicit report fields promised by the pseudocode.
    best_region_summary: Dict[str, Any] = field(default_factory=dict)
    surviving_region_summaries: List[Dict[str, Any]] = field(default_factory=list)

    number_of_surviving_regions: int = 0
    number_of_tested_points: int = 0
    number_of_invalid_points: int = 0
    number_of_rejected_spikes: int = 0
    number_of_rejected_fragile_peaks: int = 0
    number_of_rejected_noisy_points: int = 0
    number_of_rejected_low_evidence_points: int = 0
    number_of_rejected_small_regions: int = 0
    number_of_dbscan_rejected_isolated_points: int = 0
    search_rounds_completed: int = 0
    early_stopped: bool = False
    compute_budget_reached: bool = False
    # Objective-call accounting. total_compute_budget gates search_evaluations
    # only; holdout_evaluations are additional real calls during holdout validation.
    search_evaluations: int = 0
    holdout_evaluations: int = 0
    total_objective_calls: int = 0
    parameter_weights_used: Dict[str, float] = field(default_factory=dict)
    neighbor_radius_used: float = 0.0
    dbscan_minimum_points_per_region: int = 0
    # Multiple-testing / selection accounting (DSR-style): number of regions the
    # best was chosen from, and the raise applied to the "high" holdout bar to
    # offset best-of-N selection bias.
    number_of_trials_considered: int = 0
    # Which context-aggregation policy produced the robustness numbers below.
    # Provenance: two runs are only comparable if this matches.
    context_aggregator_used: str = "worst_case"
    deflated_holdout_threshold: float = 0.0
    holdout_selection_penalty: float = 0.0
    warning_signs: List[str] = field(default_factory=list)
    final_confidence_rating: str = "low"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NARRSConfig:
    """
    Main configuration for Noise-Aware Robust Region Search.
    """
    search_rounds: int = 3
    # Grid density is only used by GridSampler or grid-style refinement.
    initial_grid_density: int = 4

    # Candidate count is used by non-grid samplers like random or Latin hypercube.
    initial_candidate_count: int = 64
    
    repetitions_per_point: int = 3
    max_extra_repetitions: int = 7
    max_total_repetitions_per_point: int = 10

    exploration_fraction: float = 0.20
    total_compute_budget: int = 10_000

    # Parallel candidate evaluation. 1 = serial (default). >1 uses that many
    # worker processes; -1 uses all CPU cores. Requires a picklable
    # objective_function (module-level, not a lambda/closure); otherwise NARRS
    # warns and runs serially.
    n_jobs: int = 1

    neighbor_radius_multiplier: float = 1.5
    minimum_points_per_region: Optional[int] = None

    minimum_region_size: int = 3
    minimum_region_width: float = 0.05
    minimum_region_density: float = 1.0

    minimum_sample_size: int = 1
    maximum_allowed_noise: float = 1.0
    maximum_confidence_interval_width: float = 1.0

    minimum_acceptable_worst_case_score: float = 0.0
    minimum_acceptable_holdout_score: float = 0.0
    # Minimum number of valid holdout scores required before a "medium" or "high"
    # rating can be granted. Guards against rating a region confidently when its
    # holdout produced no evidence (zero-filled stats trivially pass the checks).
    minimum_holdout_evidence: int = 5
    maximum_context_failure_count: int = 0

    minimum_neighborhood_mean: float = 0.0
    maximum_neighborhood_noise: float = 1.0
    maximum_spike_penalty: float = 1.0

    # How per-context means are reduced to one robustness number. Either a
    # builtin name ("worst_case", "cvar", "mean_std", "mean") or any object with
    # .aggregate(context_means, weights=None) and .name. Default reproduces the
    # original min-over-contexts behaviour exactly. Kept as a string by default
    # so NARRSConfig stays flat and serialisable -- this config is itself meant
    # to become a search space.
    context_aggregator: Any = "worst_case"
    # Parameter for the parameterised aggregators: tail fraction for "cvar",
    # k for "mean_std". Ignored by "worst_case" and "mean".
    context_aggregator_alpha: float = 0.25

    noise_penalty_weight: float = 1.25
    context_penalty_weight: float = 1.25
    neighborhood_penalty_weight: float = 1.25
    spike_penalty_weight: float = 2.0
    worst_case_weight: float = 0.75
    region_size_weight: float = 0.15

    early_stopping_minimum_improvement: float = 0.02
    early_stopping_patience: int = 2
    region_boundary_stability_tolerance: float = 0.02
    high_confidence_region_score: float = 0.75
    high_confidence_max_region_noise: float = 0.15
    high_confidence_max_context_instability: float = 0.15

    boundary_retest_tolerance: float = 0.03
    single_context_overfit_gap: float = 0.20
    maximum_holdout_noise: float = 1.0

    top_region_fraction: float = 0.30
    max_regions_to_refine: int = 3

    boundary_fraction: float = 0.10

    holdout_points_per_region: int = 12
    ensemble_size: int = 5

    random_seed: int = 42
    verbose: bool = False
