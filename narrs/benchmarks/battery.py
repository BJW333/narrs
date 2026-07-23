"""
The benchmark battery: run everything on the same landscape, same budget, and
print an auditable report card.

Two things happen here that are worth stating plainly.

**Budgets are matched to NARRS, not the other way round.** NARRS runs first, its
actual objective-call count is recorded, and every baseline is then given that
same number of calls. Any comparison where the methods spent different amounts
of compute is not evidence of anything.

**The battery is also the training substrate.** Each NARRS run emits one record
of ``(landscape features -> predicted confidence -> what actually happened out
of sample)``. That is the labelled corpus a self-tuning loop would calibrate
against and a meta-learning layer would learn priors from. The benchmark you
need in order to *trust* the optimizer and the data you need in order to
*improve* it are the same artifact, which is why this is worth building before
either of those loops.

Nothing here writes to disk unless you pass ``flywheel_path``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..aggregators import resolve_aggregator
from ..core import NoiseAwareRobustRegionSearch
from ..metrics import cvar, deflated_sharpe, median, pbo_cscv, sharpe
from ..types import NARRSConfig, ParamDict, SearchContext
from . import baselines as B
from .problems import candidate_pool, normalized_score, performance_matrix


@dataclass
class MethodOutcome:
    """One method's chosen configuration, judged out of sample."""

    name: str
    picked: str
    point: Dict[str, float]
    in_sample_mean: float
    oos_mean: float
    oos_cvar: float
    oos_decay: float
    survives: bool
    dsr: float
    n_evals: int
    n_trials: int
    confidence_rating: str = ""
    aggregator: str = ""


def _oos_scores(problem, point: ParamDict, contexts: Sequence[SearchContext], reps: int) -> List[float]:
    return [
        normalized_score(problem, point, context, rep)
        for context in contexts
        for rep in range(reps)
    ]


def _judge(
    problem,
    name: str,
    point: Optional[ParamDict],
    n_evals: int,
    n_trials: int,
    search_contexts: Sequence[SearchContext],
    holdout_contexts: Sequence[SearchContext],
    reps: int,
    confidence_rating: str = "",
    aggregator: str = "",
) -> Optional[MethodOutcome]:
    if point is None:
        return None

    is_scores = _oos_scores(problem, point, search_contexts, reps)
    oos = _oos_scores(problem, point, holdout_contexts, reps)
    is_mean = sum(is_scores) / len(is_scores)
    oos_mean = sum(oos) / len(oos)
    tail = cvar(oos, 0.25)

    return MethodOutcome(
        name=name,
        picked=problem.region_kind(point),
        point={k: round(v, 4) for k, v in point.items()},
        in_sample_mean=is_mean,
        oos_mean=oos_mean,
        oos_cvar=tail,
        oos_decay=is_mean - oos_mean,
        survives=tail > problem.survival_threshold,
        dsr=deflated_sharpe(oos, n_trials=max(n_trials, 1))["dsr"],
        n_evals=n_evals,
        n_trials=n_trials,
        confidence_rating=confidence_rating,
        aggregator=aggregator,
    )


def _narrs_features(result: Dict[str, Any]) -> Dict[str, float]:
    """Landscape/search features for the flywheel record.

    Deliberately only things observable *before* you know the answer -- these are
    meant to predict survival, so anything derived from holdout truth would leak.
    """
    report = result["confidence_report"]
    region = result.get("best_region")
    features = {
        "number_of_tested_points": report.number_of_tested_points,
        "number_of_surviving_regions": report.number_of_surviving_regions,
        "number_of_rejected_spikes": report.number_of_rejected_spikes,
        "search_rounds_completed": report.search_rounds_completed,
        "number_of_trials_considered": report.number_of_trials_considered,
    }
    if region is not None:
        features.update(
            {
                "region_score": round(region.region_score, 6),
                "region_noise": round(region.region_noise, 6),
                "region_context_stability": round(region.region_context_stability, 6),
                "region_width": round(region.region_width, 6),
                "region_worst_case": round(region.region_worst_case, 6),
                "region_stability": round(region.region_stability, 6),
            }
        )
    return features


def run_battery(
    problem,
    aggregators: Sequence[str] = ("worst_case", "cvar", "mean_std", "mean"),
    aggregator_alpha: float = 0.25,
    config: Optional[NARRSConfig] = None,
    reps_for_scoring: int = 2,
    seed: int = 0,
    compute_pbo: bool = True,
    pbo_pool_size: int = 50,
    pbo_reps: int = 2,
    flywheel_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run NARRS (one run per aggregator) and the baselines on ``problem``.

    ``aggregators`` includes ``"mean"`` on purpose: it is NARRS with its tail
    sensitivity switched off, which isolates how much of the result comes from
    the aggregation policy rather than from the rest of the machinery.
    """
    search_contexts = problem.search_contexts
    holdout_contexts = problem.holdout_contexts
    base_config = config or NARRSConfig(
        search_rounds=3,
        initial_candidate_count=48,
        repetitions_per_point=2,
        total_compute_budget=6000,
        random_seed=seed,
    )

    outcomes: List[MethodOutcome] = []
    flywheel: List[Dict[str, Any]] = []
    narrs_evals: List[int] = []

    # -- NARRS, once per aggregation policy --------------------------------- #
    for agg_name in aggregators:
        cfg = NARRSConfig(**{**vars(base_config)})
        cfg.context_aggregator = agg_name
        cfg.context_aggregator_alpha = aggregator_alpha
        resolved = resolve_aggregator(agg_name, aggregator_alpha)

        optimizer = NoiseAwareRobustRegionSearch(
            objective_function=problem.objective,
            parameter_space=problem.parameter_space,
            search_contexts=search_contexts,
            holdout_contexts=holdout_contexts,
            metric_rules=problem.metric_rules,
            metric_weights=problem.metric_weights,
            config=cfg,
        )
        result = optimizer.run()
        report = result["confidence_report"]
        used = report.total_objective_calls or (
            report.search_evaluations + report.holdout_evaluations
        )
        narrs_evals.append(used)

        outcome = _judge(
            problem,
            name=f"NARRS [{resolved.name}]",
            point=result.get("recommended_center"),
            n_evals=used,
            n_trials=max(report.number_of_trials_considered, 1),
            search_contexts=search_contexts,
            holdout_contexts=holdout_contexts,
            reps=reps_for_scoring,
            confidence_rating=report.final_confidence_rating,
            aggregator=resolved.name,
        )
        if outcome is None:
            if verbose:
                print(f"  NARRS [{resolved.name}]: no region survived -> abstained")
            flywheel.append(
                {
                    "problem": type(problem).__name__,
                    "aggregator": resolved.name,
                    "features": _narrs_features(result),
                    "predicted_confidence": report.final_confidence_rating,
                    "abstained": True,
                    "actual_survived": None,
                }
            )
            continue

        outcomes.append(outcome)
        flywheel.append(
            {
                "problem": type(problem).__name__,
                "aggregator": resolved.name,
                "features": _narrs_features(result),
                "predicted_confidence": report.final_confidence_rating,
                "abstained": False,
                "actual_survived": outcome.survives,
                "actual_landed_on": outcome.picked,
                "actual_oos_cvar": round(outcome.oos_cvar, 6),
                "actual_oos_decay": round(outcome.oos_decay, 6),
            }
        )

    # -- baselines, on the budget NARRS actually used ----------------------- #
    matched_budget = max(narrs_evals) if narrs_evals else 3000
    for fn in B.ALL_BASELINES:
        res = fn(problem, search_contexts, matched_budget, reps=2, seed=seed)
        if res is None:
            continue
        outcome = _judge(
            problem,
            name=res["name"],
            point=res["point"],
            n_evals=res["n_evals"],
            n_trials=res["n_trials"],
            search_contexts=search_contexts,
            holdout_contexts=holdout_contexts,
            reps=reps_for_scoring,
        )
        if outcome:
            outcomes.append(outcome)

    # -- PBO of the selection policy on this landscape ---------------------- #
    pbo_naive = pbo_robust = None
    if compute_pbo:
        anchors: List[ParamDict] = []
        if hasattr(problem, "_tc"):
            anchors.append(dict(problem._tc))
        if hasattr(problem, "_pc"):
            anchors.append(dict(problem._pc))
        if hasattr(problem, "_sc"):
            anchors.append(dict(problem._sc))
        if hasattr(problem, "_decoys"):
            anchors.extend(dict(d) for d in problem._decoys[:20])

        pool = candidate_pool(problem, n=pbo_pool_size, seed=seed + 99, include=anchors)
        matrix = performance_matrix(problem, pool, search_contexts, reps=pbo_reps)

        def _mean_metric(xs):
            return sum(xs) / len(xs) if xs else 0.0

        def _cvar_metric(xs):
            return cvar(xs, 0.25)

        # The evaluation metric is fixed to whatever "good out of sample" means
        # on this landscape, and only the *selection* rule varies. Scoring a
        # tail-aware selector by out-of-sample mean would mark it down for
        # declining to maximise a quantity it is deliberately not maximising,
        # which measures nothing.
        evaluate = _cvar_metric if getattr(problem, "evaluation_metric", "mean") == "cvar" else _mean_metric
        pbo_naive = pbo_cscv(matrix, n_splits=8, is_metric=_mean_metric, oos_metric=evaluate)
        pbo_robust = pbo_cscv(matrix, n_splits=8, is_metric=_cvar_metric, oos_metric=evaluate)

    out = {
        "problem": type(problem).__name__,
        "headline": problem.headline,
        "evaluation_metric": getattr(problem, "evaluation_metric", "mean"),
        "robust_label": problem.robust_label,
        "trap_label": problem.trap_label,
        "survival_threshold": problem.survival_threshold,
        "matched_budget_evals": matched_budget,
        "outcomes": outcomes,
        "pbo_naive": pbo_naive,
        "pbo_robust": pbo_robust,
        "flywheel_records": flywheel,
    }

    if flywheel_path:
        with open(flywheel_path, "a") as handle:
            for record in flywheel:
                handle.write(json.dumps(record) + "\n")

    if verbose:
        print_report(out)
    return out


def print_report(out: Dict[str, Any]) -> None:
    width = 116
    print("=" * width)
    print(f"{out['problem']}   --   headline metric: {out['headline']}")
    print(
        f"survival bar: OOS CVaR@25% > {out['survival_threshold']:.2f} "
        f"(i.e. beats doing nothing)   |   budget per method: {out['matched_budget_evals']} objective calls"
    )
    if out["pbo_naive"] is not None:
        print(
            f"PBO, judged on OOS {out['evaluation_metric']}:  select by mean = {out['pbo_naive']['pbo']:.2f}"
            f"   vs   select by CVaR = {out['pbo_robust']['pbo']:.2f}"
            "    (lower is better; 0.5 = coin flip)"
        )
    print("-" * width)
    print(
        f"{'method':28s} {'picked':10s} {'IS mean':>8s} {'OOS mean':>9s} "
        f"{'OOS CVaR':>9s} {'decay':>7s} {'survives':>9s} {'DSR':>6s} {'rating':>8s}"
    )
    print("-" * width)
    for o in out["outcomes"]:
        print(
            f"{o.name:28s} {o.picked:10s} {o.in_sample_mean:8.3f} {o.oos_mean:9.3f} "
            f"{o.oos_cvar:9.3f} {o.oos_decay:7.3f} {('YES' if o.survives else 'no'):>9s} "
            f"{o.dsr:6.2f} {o.confidence_rating:>8s}"
        )
    print("=" * width)

    robust = [o for o in out["outcomes"] if o.picked == out["robust_label"]]
    trap = [o for o in out["outcomes"] if o.picked == out["trap_label"]]
    print(f"landed on {out['robust_label']:>10s} : {', '.join(o.name for o in robust) or 'none'}")
    print(f"landed on {out['trap_label']:>10s} : {', '.join(o.name for o in trap) or 'none'}")
    if robust and trap:
        gap = (
            sum(o.oos_cvar for o in robust) / len(robust)
            - sum(o.oos_cvar for o in trap) / len(trap)
        )
        print(f"out-of-sample tail (CVaR@25%) advantage of the robust pick: {gap:+.3f}")
    print("=" * width)
    print()


def run_all(seed: int = 0, flywheel_path: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """Run the full battery on both landscapes."""
    from .problems import DecoyOverfittingProblem, PlateauSpikeProblem

    results = {}
    for problem in (PlateauSpikeProblem(), DecoyOverfittingProblem()):
        results[type(problem).__name__] = run_battery(
            problem, seed=seed, flywheel_path=flywheel_path, verbose=verbose
        )
    return results


__all__ = ["run_battery", "run_all", "print_report", "MethodOutcome"]
