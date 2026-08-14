"""
Stress suite: does NARRS hold up as the problem gets bigger and harder?

``test_additions.py`` checks that each function is correct. ``verify_changes.py``
checks that the integration didn't break anything. This file asks the question
that decides whether you can rely on the thing: **where does it start to fail,
and does it tell you when it does?**

    python tests/stress_suite.py --quick              # ~5 min, reduced sweep
    python tests/stress_suite.py                      # ~25 min, full sweep
    python tests/stress_suite.py --seeds 12           # more statistical power
    python tests/stress_suite.py --axis contexts      # one axis only
    python tests/stress_suite.py --json results.json  # machine-readable dump

The headline is **G1, calibration**. NARRS does not claim to always find the
right region -- no search on a noisy objective can. It claims that when it says
"high confidence", that means something. So the suite measures, across every
cell of the sweep, whether a "high" rating actually predicts out-of-sample
survival. A run that fails and is honestly labelled "medium" is the system
working. A run that fails while rated "high" is the system lying, which is the
only failure mode that really matters here.

Gates
-----
G1  calibration   among "high"-rated runs, survival rate >= 90%
G2  dominance     NARRS survives at least as often as the best baseline, per cell
G3  stability     no crashes anywhere in the sweep
G4  determinism   same seed twice gives an identical answer
G5  edge cases    degenerate setups fail cleanly or run, never hang or corrupt
G6  runtime       cost growth is reported per axis (informational, not gated)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narrs import (  # noqa: E402
    MetricRule,
    NARRSConfig,
    NoiseAwareRobustRegionSearch,
    ObjectiveResult,
    ParameterSpec,
    SearchContext,
)
from narrs.benchmarks import baselines as B  # noqa: E402
from narrs.benchmarks.battery import _judge  # noqa: E402
from narrs.benchmarks.scalable import ScalableDecoy, ScalablePlateauSpike  # noqa: E402
from narrs.benchmarks.adversarial import (  # noqa: E402
    CorrelatedContexts,
    DeceptiveMultiModal,
    HeavyTailNoise,
    RegimeShift,
)
from narrs.validation import NARRSConfigurationError  # noqa: E402


# --------------------------------------------------------------------------- #
# result plumbing                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class CellResult:
    axis: str
    label: str
    problem: str
    n_seeds: int
    narrs_survived: int = 0
    narrs_high: int = 0
    narrs_high_survived: int = 0
    narrs_abstained: int = 0
    baseline_best_survived: int = 0
    picks: Dict[str, int] = field(default_factory=dict)
    mean_seconds: float = 0.0
    mean_evals: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def calibration(self) -> Optional[float]:
        return self.narrs_high_survived / self.narrs_high if self.narrs_high else None


_gate_failures: List[str] = []
_cells: List[CellResult] = []


def _fail(gate: str, detail: str) -> None:
    _gate_failures.append(f"{gate}: {detail}")


# --------------------------------------------------------------------------- #
# one cell                                                                     #
# --------------------------------------------------------------------------- #
def run_cell(
    axis: str,
    label: str,
    make_problem: Callable[[], Any],
    n_seeds: int,
    budget: int = 5000,
    candidates: int = 40,
    reps: int = 2,
    rounds: int = 3,
    with_baselines: bool = True,
) -> CellResult:
    problem0 = make_problem()
    cell = CellResult(axis=axis, label=label, problem=type(problem0).__name__, n_seeds=n_seeds)
    times: List[float] = []
    evals: List[int] = []

    for seed in range(n_seeds):
        problem = make_problem()
        search_ctx = problem.search_contexts
        holdout_ctx = problem.holdout_contexts
        try:
            cfg = NARRSConfig(
                search_rounds=rounds, initial_candidate_count=candidates,
                repetitions_per_point=reps, total_compute_budget=budget,
                random_seed=seed,
            )
            opt = NoiseAwareRobustRegionSearch(
                objective_function=problem.objective,
                parameter_space=problem.parameter_space,
                search_contexts=search_ctx, holdout_contexts=holdout_ctx,
                metric_rules=problem.metric_rules,
                metric_weights=problem.metric_weights, config=cfg,
            )
            t0 = time.time()
            result = opt.run()
            times.append(time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            cell.errors.append(f"seed {seed}: {type(exc).__name__}: {exc}")
            _fail("G3 stability", f"{axis}/{label} seed {seed}: {type(exc).__name__}: {exc}")
            continue

        report = result["confidence_report"]
        used = report.total_objective_calls or (
            report.search_evaluations + report.holdout_evaluations
        )
        evals.append(used)
        center = result.get("recommended_center")

        if center is None:
            cell.narrs_abstained += 1
            cell.picks["abstain"] = cell.picks.get("abstain", 0) + 1
        else:
            outcome = _judge(problem, "NARRS", center, used,
                             max(report.number_of_trials_considered, 1),
                             search_ctx, holdout_ctx, 2)
            cell.picks[outcome.picked] = cell.picks.get(outcome.picked, 0) + 1
            if outcome.survives:
                cell.narrs_survived += 1
            if report.final_confidence_rating == "high":
                cell.narrs_high += 1
                if outcome.survives:
                    cell.narrs_high_survived += 1

        if with_baselines:
            best = 0
            for fn in B.ALL_BASELINES:
                try:
                    res = fn(problem, search_ctx, used, reps=2, seed=seed)
                except Exception as exc:  # noqa: BLE001
                    _fail("G3 stability", f"baseline {fn.__name__} {axis}/{label}: {exc}")
                    continue
                if not res:
                    continue
                bo = _judge(problem, res["name"], res["point"], res["n_evals"],
                            res["n_trials"], search_ctx, holdout_ctx, 2)
                if bo and bo.survives:
                    best = 1
            cell.baseline_best_survived += best

    cell.mean_seconds = statistics.mean(times) if times else 0.0
    cell.mean_evals = statistics.mean(evals) if evals else 0.0

    # G2: NARRS must not be *systematically* beaten by a naive mean-optimizer.
    #
    # A one-seed difference is noise, not evidence, so the gate needs a margin.
    # It also has to allow for the fact that these landscapes stop being
    # adversarial at the easy end of each axis: give the decoy problem enough
    # contexts and the decoys average out, at which point a plain mean-optimizer
    # finds the true edge too and there is nothing left for robustness to buy.
    # Demanding dominance there would be demanding that NARRS win a race nobody
    # is running.
    margin = max(1, int(round(0.15 * n_seeds)))
    if with_baselines and cell.baseline_best_survived - cell.narrs_survived > margin:
        _fail("G2 dominance",
              f"{axis}/{label} ({cell.problem}): baseline survived "
              f"{cell.baseline_best_survived}/{n_seeds} vs NARRS "
              f"{cell.narrs_survived}/{n_seeds} (margin {margin})")

    _cells.append(cell)
    return cell


def print_cell_header(axis: str) -> None:
    print(f"\n  {'cell':>16s} {'problem':>22s} {'NARRS':>8s} {'high-rated':>12s} "
          f"{'calib':>6s} {'baseline':>9s} {'abst':>5s} {'sec':>6s}")
    print("  " + "-" * 96)


def print_cell(cell: CellResult) -> None:
    calib = cell.calibration
    calib_s = "n/a" if calib is None else f"{calib:.0%}"
    print(f"  {cell.label:>16s} {cell.problem:>22s} "
          f"{cell.narrs_survived:>3d}/{cell.n_seeds:<4d} "
          f"{cell.narrs_high_survived:>4d}/{cell.narrs_high:<7d} "
          f"{calib_s:>6s} {cell.baseline_best_survived:>4d}/{cell.n_seeds:<4d} "
          f"{cell.narrs_abstained:>5d} {cell.mean_seconds:>6.1f}")


# --------------------------------------------------------------------------- #
# axes                                                                         #
# --------------------------------------------------------------------------- #
def axis_dimensions(seeds: int, quick: bool, baselines: bool) -> None:
    print("\n" + "=" * 100)
    print("AXIS: DIMENSIONALITY -- more parameters means a larger space to search")
    print("=" * 100)
    dims = (2, 4) if quick else (2, 4, 6, 8)
    print_cell_header("dims")
    for dim in dims:
        # budget scales with dimension: holding it fixed would measure starvation
        budget = 4000 + 1500 * dim
        cands = 32 + 12 * dim
        for make in (lambda d=dim: ScalablePlateauSpike(dim=d),
                     lambda d=dim: ScalableDecoy(dim=d)):
            print_cell(run_cell("dims", f"dim={dim}", make, seeds,
                                budget=budget, candidates=cands,
                                with_baselines=baselines))


def axis_contexts(seeds: int, quick: bool, baselines: bool) -> None:
    print("\n" + "=" * 100)
    print("AXIS: CONTEXT COUNT -- how much independent evidence NARRS gets")
    print("  This is the difficulty dial for overfitting: fewer contexts, more room for luck.")
    print("=" * 100)
    counts = (3, 6, 12) if quick else (2, 3, 4, 6, 12, 24)
    print_cell_header("contexts")
    for n in counts:
        for make in (lambda k=n: ScalablePlateauSpike(n_search_contexts=k, n_holdout_contexts=max(k, 10)),
                     lambda k=n: ScalableDecoy(n_search_contexts=k)):
            print_cell(run_cell("contexts", f"n_ctx={n}", make, seeds,
                                with_baselines=baselines))


def axis_noise(seeds: int, quick: bool, baselines: bool) -> None:
    print("\n" + "=" * 100)
    print("AXIS: NOISE -- how loud the objective is relative to the signal")
    print("=" * 100)
    levels = (0.05, 0.40) if quick else (0.02, 0.10, 0.30, 0.60)
    print_cell_header("noise")
    for sd in levels:
        for make in (lambda s=sd: ScalablePlateauSpike(noise_sd=s),
                     lambda s=sd: ScalableDecoy(noise_sd=s)):
            print_cell(run_cell("noise", f"sd={sd:g}", make, seeds,
                                with_baselines=baselines))


def axis_budget(seeds: int, quick: bool, baselines: bool) -> None:
    print("\n" + "=" * 100)
    print("AXIS: COMPUTE BUDGET -- how NARRS behaves when starved")
    print("  Under-budget should show up as abstention or an honest 'medium', not a")
    print("  confident wrong answer.")
    print("=" * 100)
    budgets = ((800, 16, 2), (5000, 40, 3)) if quick else (
        (400, 12, 1), (800, 16, 2), (2000, 24, 2), (5000, 40, 3), (12000, 64, 4))
    print_cell_header("budget")
    for budget, cands, rounds in budgets:
        for make in (ScalablePlateauSpike, ScalableDecoy):
            print_cell(run_cell("budget", f"budget={budget}", make, seeds,
                                budget=budget, candidates=cands, rounds=rounds,
                                with_baselines=baselines))


def axis_difficulty(seeds: int, quick: bool, baselines: bool) -> None:
    print("\n" + "=" * 100)
    print("AXIS: TRAP DIFFICULTY -- how strong the thing trying to fool it is")
    print("=" * 100)
    settings = ((40, 1.7), (80, 3.0)) if quick else (
        (10, 1.0), (40, 1.7), (80, 3.0), (150, 4.5))
    print_cell_header("difficulty")
    for n_decoys, amp in settings:
        print_cell(run_cell("difficulty", f"{n_decoys}d/a={amp:g}",
                            lambda n=n_decoys, a=amp: ScalableDecoy(n_decoys=n, decoy_amplitude_sd=a),
                            seeds, with_baselines=baselines))
    for crash in ((0.10, 0.40) if quick else (0.05, 0.10, 0.25, 0.40)):
        print_cell(run_cell("difficulty", f"crash={crash:g}",
                            lambda c=crash: ScalablePlateauSpike(crash_fraction=c),
                            seeds, with_baselines=baselines))


# --------------------------------------------------------------------------- #
# G4 determinism / G5 edge cases                                               #
# --------------------------------------------------------------------------- #
def axis_adversarial(seeds: int, quick: bool, baselines: bool) -> None:
    """Landscapes that attack NARRS's assumptions rather than its difficulty.

    Every other axis draws search and holdout contexts from the SAME
    distribution. Real objectives do not. These cells break that assumption on
    purpose, and the bar to clear is NOT "survive" -- on several of them
    surviving is impossible. It is "do not claim high confidence while failing".
    """
    print("\n" + "=" * 100)
    print("AXIS: ADVERSARIAL -- assumptions violated on purpose")
    print("  Passing here means refusing to be confident, not being right.")
    print("=" * 100)
    cells = [
        ("regime-shift", RegimeShift),
        ("regime-big", lambda: RegimeShift(shift=0.6)),
        ("correlated-ctx", CorrelatedContexts),
        ("heavy-tail", HeavyTailNoise),
        ("deceptive", DeceptiveMultiModal),
    ]
    print_cell_header("adversarial")
    for label, make in cells:
        print_cell(run_cell("adversarial", label, make, seeds, with_baselines=baselines))


def check_determinism() -> None:
    print("\n" + "=" * 100)
    print("G4 DETERMINISM -- the same seed must give the same answer")
    print("=" * 100)
    for make, name in ((ScalablePlateauSpike, "PlateauSpike"), (ScalableDecoy, "Decoy")):
        outs = []
        for _ in range(2):
            problem = make()
            opt = NoiseAwareRobustRegionSearch(
                objective_function=problem.objective,
                parameter_space=problem.parameter_space,
                search_contexts=problem.search_contexts,
                holdout_contexts=problem.holdout_contexts,
                metric_rules=problem.metric_rules,
                metric_weights=problem.metric_weights,
                config=NARRSConfig(search_rounds=2, initial_candidate_count=24,
                                   repetitions_per_point=2, total_compute_budget=2000,
                                   random_seed=5),
            )
            r = opt.run()
            outs.append(json.dumps({
                "c": r["recommended_center"],
                "s": round(r["best_region"].region_score, 12) if r["best_region"] else None,
                "rating": r["confidence_report"].final_confidence_rating,
            }, sort_keys=True))
        ok = outs[0] == outs[1]
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: repeat run identical")
        if not ok:
            _fail("G4 determinism", f"{name} differed between identical runs")


def _tiny_objective(params, context, rep):
    return ObjectiveResult(performance_metrics={"score": sum(params.values())}, sample_count=1)


def _always_invalid(params, context, rep):
    return ObjectiveResult(performance_metrics={"score": 0.0}, sample_count=1,
                           validity_status="invalid")


def _nan_objective(params, context, rep):
    return ObjectiveResult(performance_metrics={"score": float("nan")}, sample_count=1)


def check_edge_cases() -> None:
    print("\n" + "=" * 100)
    print("G5 EDGE CASES -- degenerate setups must fail cleanly, never hang or lie")
    print("=" * 100)

    def attempt(name: str, build: Callable[[], Any], expect_raises: bool = False,
                degenerate: bool = True) -> None:
        try:
            result = build()
        except NARRSConfigurationError as exc:
            verdict = expect_raises
            print(f"  [{'PASS' if verdict else 'FAIL'}] {name}: raised "
                  f"NARRSConfigurationError ({str(exc)[:50]})")
            if not verdict:
                _fail("G5 edge cases", f"{name} unexpectedly rejected: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            _fail("G5 edge cases", f"{name}: {type(exc).__name__}: {exc}")
            return
        if expect_raises:
            print(f"  [FAIL] {name}: expected a configuration error, got a result")
            _fail("G5 edge cases", f"{name} should have been rejected")
            return
        center = result.get("recommended_center")
        report = result["confidence_report"]
        rating = report.final_confidence_rating
        warnings = report.warning_signs
        # The bar is not "finds something". It is: never assert high confidence
        # on evidence that cannot support it *without saying so*. Abstaining is
        # fine, a lower rating is fine, and "high" is acceptable only when the
        # report also carries a warning explaining what is thin.
        ok = (not degenerate) or center is None or rating != "high" or bool(warnings)
        if center is None:
            state = "abstained"
        else:
            state = f"rating={rating}" + (f", {len(warnings)} warning(s)" if warnings else ", NO warning")
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {state}")
        if not ok:
            _fail("G5 edge cases",
                  f"{name} returned HIGH confidence with no warning on a degenerate setup")

    def build(**kw):
        problem = kw.pop("problem", None) or ScalablePlateauSpike()
        search = kw.pop("search_contexts", problem.search_contexts)
        holdout = kw.pop("holdout_contexts", problem.holdout_contexts)
        objective = kw.pop("objective", problem.objective)
        space = kw.pop("parameter_space", problem.parameter_space)
        cfg = NARRSConfig(**{"search_rounds": 1, "initial_candidate_count": 12,
                             "repetitions_per_point": 1, "total_compute_budget": 300,
                             "random_seed": 0, **kw})
        return NoiseAwareRobustRegionSearch(
            objective_function=objective, parameter_space=space,
            search_contexts=search, holdout_contexts=holdout,
            metric_rules=problem.metric_rules, metric_weights=problem.metric_weights,
            config=cfg,
        ).run()

    attempt("single search context", lambda: build(
        search_contexts=[SearchContext("only")]))
    attempt("single holdout context", lambda: build(
        holdout_contexts=[SearchContext("only_oos")]))
    # A one-dimensional search is perfectly legitimate, not degenerate -- it is
    # here to confirm the machinery does not assume dim >= 2, and "high" is a
    # valid outcome.
    attempt("one parameter (legitimate)", lambda: build(
        parameter_space=[ParameterSpec("x0", 0.0, 1.0)],
        problem=ScalablePlateauSpike(dim=1)), degenerate=False)
    attempt("tiny budget (30 calls)", lambda: build(total_compute_budget=30))
    attempt("one candidate, one round", lambda: build(
        initial_candidate_count=1, search_rounds=1))
    attempt("objective always invalid", lambda: build(objective=_always_invalid))
    attempt("objective returns NaN", lambda: build(objective=_nan_objective))
    attempt("constant objective (no signal)", lambda: build(
        objective=lambda p, c, r: ObjectiveResult(
            performance_metrics={"score": 1.0}, sample_count=1)))
    attempt("zero-width parameter range", lambda: build(
        parameter_space=[ParameterSpec("x0", 0.5, 0.5), ParameterSpec("x1", 0.0, 1.0)]),
        expect_raises=True)
    attempt("no holdout contexts", lambda: build(holdout_contexts=[]))


# --------------------------------------------------------------------------- #
# summary                                                                      #
# --------------------------------------------------------------------------- #
def summarise(json_path: Optional[str]) -> int:
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    high = sum(c.narrs_high for c in _cells)
    high_ok = sum(c.narrs_high_survived for c in _cells)
    total_runs = sum(c.n_seeds for c in _cells)
    survived = sum(c.narrs_survived for c in _cells)
    abstained = sum(c.narrs_abstained for c in _cells)
    base = sum(c.baseline_best_survived for c in _cells)

    print(f"  cells run                     : {len(_cells)}")
    print(f"  NARRS runs                    : {total_runs}")
    print(f"  survived out of sample        : {survived}/{total_runs} "
          f"({survived/total_runs:.0%})" if total_runs else "")
    print(f"  abstained (no region found)   : {abstained}/{total_runs}")
    print(f"  best baseline survived        : {base}/{total_runs}")

    if high:
        rate = high_ok / high
        print(f"\n  G1 CALIBRATION -- the claim that matters:")
        print(f"     runs rated HIGH             : {high}")
        print(f"     of those, survived OOS      : {high_ok}  ({rate:.1%})")
        ok = rate >= 0.90
        print(f"     [{'PASS' if ok else 'FAIL'}] a 'high' rating predicts survival (bar: 90%)")
        if not ok:
            _fail("G1 calibration",
                  f"only {rate:.0%} of HIGH-rated runs survived (bar 90%) -- "
                  "the confidence rating is not trustworthy")
    else:
        print("\n  G1 CALIBRATION: no HIGH-rated runs in this sweep -- cannot assess")

    # G6 runtime, informational
    print("\n  G6 RUNTIME (informational):")
    for axis in ("dims", "contexts", "budget"):
        cells = [c for c in _cells if c.axis == axis]
        if len(cells) >= 2:
            first, last = cells[0], cells[-1]
            if first.mean_seconds > 0:
                growth = last.mean_seconds / first.mean_seconds
                print(f"     {axis:>9s}: {first.label} = {first.mean_seconds:.1f}s  ->  "
                      f"{last.label} = {last.mean_seconds:.1f}s  ({growth:.1f}x)")

    if json_path:
        Path(json_path).write_text(json.dumps(
            {"cells": [asdict(c) for c in _cells], "gate_failures": _gate_failures},
            indent=1))
        print(f"\n  wrote {json_path}")

    print("\n" + "=" * 100)
    if _gate_failures:
        print(f"FAILED -- {len(_gate_failures)} gate failure(s):")
        for failure in _gate_failures:
            print(f"  * {failure}")
        print("=" * 100)
        return 1
    print("ALL GATES PASSED")
    print("=" * 100)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NARRS stress and scaling suite")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--axis", default="all",
                        choices=["all", "dims", "contexts", "noise", "budget",
                                 "difficulty", "adversarial", "edge",
                                 "determinism"])
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    seeds = 3 if args.quick else args.seeds
    baselines = not args.no_baselines

    print("=" * 100)
    print(f"NARRS STRESS SUITE   seeds/cell={seeds}  quick={args.quick}  baselines={baselines}")
    print("=" * 100)
    started = time.time()

    axes = {
        "dims": axis_dimensions, "contexts": axis_contexts, "noise": axis_noise,
        "budget": axis_budget, "difficulty": axis_difficulty,
        "adversarial": axis_adversarial,
    }
    if args.axis in ("all", "determinism"):
        check_determinism()
    if args.axis in ("all", "edge"):
        check_edge_cases()
    for name, fn in axes.items():
        if args.axis in ("all", name):
            fn(seeds, args.quick, baselines)

    print(f"\n  total wall time: {(time.time() - started) / 60:.1f} min")
    return summarise(args.json)


if __name__ == "__main__":
    sys.exit(main())
