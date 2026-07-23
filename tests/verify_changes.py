"""
Verification harness: evidence that the additions work and changed nothing else.

Unit tests (``tests/test_additions.py``) check that each new function does what
it says. This file answers the harder question -- does the *integration* hold up
under adversarial checking? Each section below was written to try to falsify a
claim, and two of them succeeded during development (see NOTES at the bottom).

    python tests/verify_changes.py                       # all checks
    python tests/verify_changes.py --original /path/to/old/narrs_package
    python tests/verify_changes.py --seeds 16            # wider sweep, slower

The ``--original`` flag runs the equivalence check against an untouched copy of
the package: identical config, identical seed, and every reported field compared.
Without it that one check is skipped and the rest still run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

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
from narrs.benchmarks.problems import (  # noqa: E402
    DecoyOverfittingProblem,
    PlateauSpikeProblem,
)
from narrs.metrics import pbo_cscv  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. Equivalence: the default path must be untouched                           #
# --------------------------------------------------------------------------- #
AB_PROBE = '''
import json, math, random
from narrs import (NoiseAwareRobustRegionSearch, NARRSConfig, ParameterSpec,
                   MetricRule, ObjectiveResult, SearchContext)

def objective(params, context, rep):
    off = {"a": 10, "b": 20, "c": 30, "h1": 40, "h2": 50}
    rng = random.Random(rep + off.get(context.name, 0))
    x, y = params["x"], params["y"]
    s = max(0.0, 1.0 - 0.12 * ((x - 2.0) ** 2 + 0.5 * (y + 1.0) ** 2))
    s += 0.9 * math.exp(-0.5 * (((x + 3.0) ** 2 + (y - 3.0) ** 2) / 0.02))
    s += rng.gauss(0, 0.05)
    return ObjectiveResult(performance_metrics={"score": s}, sample_count=1)

opt = NoiseAwareRobustRegionSearch(
    objective_function=objective,
    parameter_space=[ParameterSpec("x", -5, 5), ParameterSpec("y", -5, 5, weight=1.5)],
    search_contexts=[SearchContext("a"), SearchContext("b"), SearchContext("c")],
    holdout_contexts=[SearchContext("h1"), SearchContext("h2")],
    metric_rules={"score": MetricRule(good_value=1.0, bad_value=0.0)},
    metric_weights={"score": 1.0},
    config=NARRSConfig(search_rounds=3, total_compute_budget=4000, random_seed=42),
)
r = opt.run(); rep = r["confidence_report"]; br = r["best_region"]
print(json.dumps({
    "recommended_center": r["recommended_center"],
    "n_robust_regions": len(r["robust_regions"]),
    "region_score": round(br.region_score, 12) if br else None,
    "confidence_score": round(br.confidence_score, 12) if br else None,
    "holdout_mean": round(br.holdout_mean, 12) if br else None,
    "region_worst_case": round(br.region_worst_case, 12) if br else None,
    "ensemble": r["optional_region_ensemble"],
    "rating": rep.final_confidence_rating,
    "tested_points": rep.number_of_tested_points,
    "search_evals": rep.search_evaluations,
    "holdout_evals": rep.holdout_evaluations,
    "rejected_spikes": rep.number_of_rejected_spikes,
    "rejected_small_regions": rep.number_of_rejected_small_regions,
    "isolated_rejected": rep.number_of_dbscan_rejected_isolated_points,
    "rounds": rep.search_rounds_completed,
    "trials": rep.number_of_trials_considered,
    "deflated_bar": round(rep.deflated_holdout_threshold, 12),
    "warnings": rep.warning_signs,
    "ranges": rep.best_region_parameter_ranges,
}, sort_keys=True, indent=1))
'''


def check_equivalence(original_path: str | None) -> None:
    print("\n1. EQUIVALENCE -- does a default run still produce the original output?")
    if not original_path:
        print("     (skipped: pass --original /path/to/untouched/package to enable)")
        return
    if not os.path.isdir(os.path.join(original_path, "narrs")):
        record("original package found", False, f"no narrs/ under {original_path}")
        return

    here = str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryDirectory() as tmp:
        probe = os.path.join(tmp, "ab_probe.py")
        Path(probe).write_text(AB_PROBE)
        outs = {}
        for label, path in (("original", original_path), ("modified", here)):
            env = dict(os.environ, PYTHONPATH=path)
            proc = subprocess.run(
                [sys.executable, probe], env=env, capture_output=True, text=True, timeout=1800
            )
            if proc.returncode != 0:
                record(f"{label} run completed", False, proc.stderr.strip().splitlines()[-1:] or "")
                return
            outs[label] = proc.stdout

    same = outs["original"] == outs["modified"]
    n_fields = len(json.loads(outs["original"]))
    record(
        "default run is byte-identical to the original",
        same,
        f"{n_fields} fields compared at 12dp"
        if same
        else "OUTPUT DIFFERS -- the changes altered existing behaviour",
    )


# --------------------------------------------------------------------------- #
# 2. Is the aggregator seam functional, or dead code?                          #
# --------------------------------------------------------------------------- #
_BAD_CONTEXT = "ctx_07"


def _rare_outlier_objective(params, context, rep):
    """Region A is excellent in 23/24 contexts and catastrophic in one.

    A rare severe outlier is exactly where `min` and CVaR must disagree: the
    standard deviation barely moves, but the minimum collapses.
    """
    x = params["x"]
    a = math.exp(-0.5 * ((x - 0.25) / 0.10) ** 2)
    b = math.exp(-0.5 * ((x - 0.75) / 0.10) ** 2)
    va = (-2.0 if context.name == _BAD_CONTEXT else 0.95) * a
    vb = 0.72 * b
    rng = random.Random(hash((context.name, rep, round(x, 6))) % (2 ** 31))
    return ObjectiveResult(
        performance_metrics={"score": va + vb + rng.gauss(0, 0.02)}, sample_count=1
    )


def _run_outlier(aggregator: str, alpha: float, isolate: bool) -> str:
    extra = (
        dict(
            noise_penalty_weight=0.0,
            context_penalty_weight=0.0,
            neighborhood_penalty_weight=0.0,
            spike_penalty_weight=0.0,
        )
        if isolate
        else {}
    )
    cfg = NARRSConfig(
        search_rounds=3, initial_candidate_count=40, repetitions_per_point=2,
        total_compute_budget=8000, random_seed=3,
        context_aggregator=aggregator, context_aggregator_alpha=alpha,
        worst_case_weight=2.0, maximum_context_failure_count=25,
        minimum_acceptable_worst_case_score=-9.0, maximum_allowed_noise=99.0,
        maximum_neighborhood_noise=99.0, maximum_spike_penalty=99.0, **extra,
    )
    opt = NoiseAwareRobustRegionSearch(
        objective_function=_rare_outlier_objective,
        parameter_space=[ParameterSpec("x", 0.0, 1.0)],
        search_contexts=[SearchContext(f"ctx_{i:02d}") for i in range(24)],
        holdout_contexts=[SearchContext(f"oos_{i:02d}") for i in range(8)],
        metric_rules={"score": MetricRule(good_value=1.0, bad_value=-2.0)},
        metric_weights={"score": 1.0},
        config=cfg,
    )
    center = opt.run()["recommended_center"]
    if center is None:
        return "abstain"
    x = center["x"]
    return "risky" if abs(x - 0.25) < 0.15 else "steady" if abs(x - 0.75) < 0.15 else "other"


def check_aggregator_is_live() -> None:
    print("\n2. AGGREGATOR SEAM -- does the config knob actually change the answer?")
    combos = [("worst_case", 0.25), ("cvar", 0.25), ("cvar", 0.50), ("mean", 0.25)]

    isolated = {f"{a}@{al}": _run_outlier(a, al, True) for a, al in combos}
    print(f"     penalties zeroed (seam isolated): {isolated}")
    record(
        "worst_case avoids the rare-catastrophe region when isolated",
        isolated["worst_case@0.25"] == "steady",
    )
    record(
        "mean takes it -- so the seam is live, not dead code",
        isolated["mean@0.25"] == "risky",
    )

    default = {f"{a}@{al}": _run_outlier(a, al, False) for a, al in combos}
    print(f"     default penalty weights:          {default}")
    record(
        "under default weights all aggregators agree (documented interaction)",
        len(set(default.values())) == 1,
        "context_instability and point_noise co-vary with the tail and dominate",
    )


# --------------------------------------------------------------------------- #
# 3. Does the headline result survive across seeds?                            #
# --------------------------------------------------------------------------- #
def check_multi_seed(n_seeds: int) -> None:
    print(f"\n3. SEED ROBUSTNESS -- does the benchmark result hold over {n_seeds} seeds?")
    for Problem in (PlateauSpikeProblem, DecoyOverfittingProblem):
        picks: dict[str, list[str]] = {}
        tails: dict[str, list[float]] = {}
        for seed in range(n_seeds):
            problem = Problem()
            sc, hc = problem.search_contexts, problem.holdout_contexts
            cfg = NARRSConfig(
                search_rounds=3, initial_candidate_count=48, repetitions_per_point=2,
                total_compute_budget=6000, random_seed=seed,
            )
            opt = NoiseAwareRobustRegionSearch(
                objective_function=problem.objective,
                parameter_space=problem.parameter_space,
                search_contexts=sc, holdout_contexts=hc,
                metric_rules=problem.metric_rules,
                metric_weights=problem.metric_weights, config=cfg,
            )
            result = opt.run()
            report = result["confidence_report"]
            budget = report.total_objective_calls or (
                report.search_evaluations + report.holdout_evaluations
            )
            entries = [
                ("NARRS", _judge(problem, "NARRS", result["recommended_center"], budget,
                                 max(report.number_of_trials_considered, 1), sc, hc, 2))
            ]
            for fn in B.ALL_BASELINES:
                res = fn(problem, sc, budget, reps=2, seed=seed)
                if res:
                    entries.append((res["name"].split(" (")[0],
                                    _judge(problem, res["name"], res["point"],
                                           res["n_evals"], res["n_trials"], sc, hc, 2)))
            for name, outcome in entries:
                picks.setdefault(name, []).append(outcome.picked if outcome else "abstain")
                if outcome:
                    tails.setdefault(name, []).append(outcome.oos_cvar)

        bar = Problem().survival_threshold
        print(f"     {Problem.__name__}")
        for name in picks:
            survived = sum(1 for c in tails.get(name, []) if c > bar)
            counts = {k: picks[name].count(k) for k in sorted(set(picks[name]))}
            med = statistics.median(tails[name]) if tails.get(name) else float("nan")
            print(f"       {name:>12s}  survived {survived}/{n_seeds}  "
                  f"median OOS CVaR {med:6.3f}  picks {counts}")

        narrs_surv = sum(1 for c in tails.get("NARRS", []) if c > bar)
        best_base = max(
            (sum(1 for c in tails.get(n, []) if c > bar) for n in picks if n != "NARRS"),
            default=0,
        )
        record(
            f"{Problem.__name__}: NARRS survives more often than every baseline",
            narrs_surv > best_base,
            f"{narrs_surv}/{n_seeds} vs best baseline {best_base}/{n_seeds}",
        )


# --------------------------------------------------------------------------- #
# 4. PBO behaves correctly on landscapes with known answers                    #
# --------------------------------------------------------------------------- #
def check_pbo_properties() -> None:
    print("\n4. PBO CORRECTNESS -- known landscapes with known answers")
    rng = random.Random(0)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731

    identical = [[1.0] * 10 for _ in range(32)]
    got = pbo_cscv(identical, 8, metric=mean)["pbo"]
    record("identical candidates -> 0.50 (indifference, not overfitting)",
           abs(got - 0.5) < 1e-9, f"got {got:.2f}")

    noise = [[rng.gauss(0, 1) for _ in range(20)] for _ in range(64)]
    got = pbo_cscv(noise, 8, metric=mean)["pbo"]
    record("pure noise -> near a coin flip", 0.3 < got < 0.7, f"got {got:.2f}")

    signal = [[rng.gauss(0, 1) for _ in range(9)] + [rng.gauss(8, 1)] for _ in range(64)]
    got = pbo_cscv(signal, 8, metric=mean)["pbo"]
    record("one genuinely dominant candidate -> ~0", got < 0.05, f"got {got:.2f}")

    flip = []
    for t in range(64):
        first = t < 32
        flip.append([(1.0 if first else -1.0) * (j + 1) for j in range(10)])
    got = pbo_cscv(flip, 8, metric=mean)["pbo"]
    record("every candidate reverses out of sample -> high", got > 0.6, f"got {got:.2f}")

    result = pbo_cscv(noise, 8, metric=mean)
    record("enumerates C(8,4)=70 balanced partitions", result["n_partitions"] == 70,
           f"got {result['n_partitions']}")


# --------------------------------------------------------------------------- #
# 5. Parallel execution must not change results                                #
# --------------------------------------------------------------------------- #
PAR_PROBE = '''
import json
from narrs import (NoiseAwareRobustRegionSearch, NARRSConfig, ParameterSpec,
                   MetricRule, SearchContext)
from _par_objective import objective
import sys
n_jobs = int(sys.argv[1]); agg = sys.argv[2]
cfg = NARRSConfig(search_rounds=2, initial_candidate_count=32, repetitions_per_point=2,
                  total_compute_budget=3000, random_seed=7, n_jobs=n_jobs,
                  context_aggregator=agg)
o = NoiseAwareRobustRegionSearch(objective_function=objective,
    parameter_space=[ParameterSpec("x",-5,5), ParameterSpec("y",-5,5)],
    search_contexts=[SearchContext(n) for n in ("a","b","c")],
    holdout_contexts=[SearchContext(n) for n in ("h1","h2")],
    metric_rules={"score": MetricRule(good_value=1.0, bad_value=0.0)},
    metric_weights={"score":1.0}, config=cfg)
r = o.run()
print(json.dumps({"c": r["recommended_center"],
                  "s": round(r["best_region"].region_score,12) if r["best_region"] else None},
                 sort_keys=True))
'''

PAR_OBJECTIVE = '''
import math, random, zlib
from narrs import ObjectiveResult

def _seed(*parts):
    # NOT hash(): Python randomises string hashing per process (PYTHONHASHSEED),
    # so a hash-based objective differs between the two probe runs and the test
    # would report a parallelism bug that is really a test bug.
    return zlib.crc32("|".join(str(p) for p in parts).encode())

def objective(params, context, rep):
    x, y = params["x"], params["y"]
    rng = random.Random(_seed(context.name, rep, round(x, 6), round(y, 6)))
    s = max(0.0, 1.0 - 0.12*((x-2.0)**2 + 0.5*(y+1.0)**2)) + rng.gauss(0, 0.04)
    return ObjectiveResult(performance_metrics={"score": s}, sample_count=1)
'''


def check_parallel_determinism() -> None:
    print("\n5. PARALLELISM -- n_jobs must not change the answer")
    here = str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryDirectory() as tmp:
        Path(os.path.join(tmp, "_par_objective.py")).write_text(PAR_OBJECTIVE)
        probe = os.path.join(tmp, "par_probe.py")
        Path(probe).write_text(PAR_PROBE)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([here, tmp]))
        for agg in ("worst_case", "cvar"):
            outs = []
            for n_jobs in ("1", "-1"):
                proc = subprocess.run(
                    [sys.executable, probe, n_jobs, agg],
                    env=env, capture_output=True, text=True, timeout=1800,
                )
                if proc.returncode != 0:
                    record(f"parallel run ({agg})", False,
                           (proc.stderr.strip().splitlines() or [""])[-1])
                    return
                outs.append(proc.stdout.strip())
            record(f"serial == parallel with aggregator '{agg}'", outs[0] == outs[1])


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="verify the NARRS additions")
    parser.add_argument("--original", default=None,
                        help="path to an untouched copy of the package for the A/B check")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--quick", action="store_true",
                        help="skip the slow multi-seed and parallel checks")
    args = parser.parse_args()

    print("=" * 78)
    print("NARRS -- verification of the added functionality")
    print("=" * 78)

    check_equivalence(args.original)
    check_aggregator_is_live()
    check_pbo_properties()
    if not args.quick:
        check_multi_seed(args.seeds)
        check_parallel_determinism()

    print("\n" + "=" * 78)
    failures = [r for r in _results if r[1] == FAIL]
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    for name, _, detail in failures:
        print(f"  FAILED: {name} {detail}")
    print("=" * 78)
    print(NOTES)
    return 1 if failures else 0


NOTES = """
NOTES -- what this harness caught during development, kept as a record:

  * CMA-ES baseline crashed on some seeds. Truncating a generation to fit the
    remaining budget is invalid: CMA must be told about every solution it asked
    for. Fixed by stopping before a generation that will not fit and reporting
    the evaluations actually spent.

  * PBO returned 1.00 for a landscape of identical candidates. Exact ties were
    counted as certain overfitting; they are indifference. Ties now count as
    half, and the degenerate case correctly returns 0.50.

  * The aggregator does not change the recommendation under default weights.
    It is live (section 2 proves it flips the decision when isolated), but
    context_instability and point_noise respond to the same bad contexts and
    dominate the score. Switching aggregator without also relaxing those weights
    will not make NARRS less conservative.

  * Single-seed results overstated NARRS on the overfitting landscape. Over
    8 seeds it is fooled on some of them. The baselines are fooled on all.

  * The parallelism check itself was wrong at first: its probe objective keyed a
    RNG off hash() of a string, and Python randomises string hashing per process,
    so the two probe runs disagreed for reasons unrelated to n_jobs. Any objective
    you write has the same exposure -- seed from a stable digest, not hash().
"""


if __name__ == "__main__":
    sys.exit(main())
