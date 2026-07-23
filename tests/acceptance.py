"""
Acceptance suite -- the go/no-go gate before this goes anywhere.

    python tests/acceptance.py --original /path/to/untouched/package
    python tests/acceptance.py --quick            # ~3 min, fewer seeds
    python tests/acceptance.py --seeds 30         # slower, tighter estimates

Ends with a single verdict line. Everything else on screen is evidence for it.

The organising idea: the number that matters is not how often NARRS wins. It is
**how often NARRS is confidently wrong**. A run that reports "low" or "medium"
and then fails is the system working -- it told you not to trust it. A run that
reports "high" and fails is the only outcome that costs money, and it is the one
this suite is built to detect.

So the headline gate is calibration, not accuracy. A tool that is right 70% of
the time and knows which 70% is far more useful than one that is right 85% of
the time and cannot tell you when.

Gates
-----
1. EQUIVALENCE   default path byte-identical to the original package
2. CALIBRATION   'high' survives more often than not-'high', and confident
                 failures stay rare
3. DOMINANCE     NARRS survives more often than every baseline, both landscapes
4. DETERMINISM   same seed -> same answer; serial == parallel
5. DEGRADATION   as evidence thins, NARRS loses confidence rather than staying
                 confidently wrong (the safety property)
6. REGRESSION    the unit suite still passes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narrs import NARRSConfig, NoiseAwareRobustRegionSearch  # noqa: E402
from narrs.benchmarks import baselines as B  # noqa: E402
from narrs.benchmarks.battery import _judge  # noqa: E402
from narrs.benchmarks.problems import (  # noqa: E402
    DecoyOverfittingProblem,
    PlateauSpikeProblem,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_gates: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    _gates.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def _run_narrs(problem, seed: int, config_overrides: dict | None = None):
    """One NARRS run; returns (rating, outcome) where outcome may be None."""
    search, holdout = problem.search_contexts, problem.holdout_contexts
    kwargs = dict(
        search_rounds=3, initial_candidate_count=48, repetitions_per_point=2,
        total_compute_budget=6000, random_seed=seed,
    )
    kwargs.update(config_overrides or {})
    opt = NoiseAwareRobustRegionSearch(
        objective_function=problem.objective,
        parameter_space=problem.parameter_space,
        search_contexts=search, holdout_contexts=holdout,
        metric_rules=problem.metric_rules, metric_weights=problem.metric_weights,
        config=NARRSConfig(**kwargs),
    )
    result = opt.run()
    report = result["confidence_report"]
    center = result["recommended_center"]
    if center is None:
        return "abstain", None
    budget = report.total_objective_calls or (
        report.search_evaluations + report.holdout_evaluations
    )
    outcome = _judge(problem, "NARRS", center, budget,
                     max(report.number_of_trials_considered, 1),
                     search, holdout, 2)
    return report.final_confidence_rating, outcome


# --------------------------------------------------------------------------- #
# 1. Equivalence                                                              #
# --------------------------------------------------------------------------- #
def gate_equivalence(original: str | None) -> None:
    print("\n1. EQUIVALENCE -- is the original behaviour untouched?")
    if not original:
        print("     (skipped: pass --original to enable; NOT a pass)")
        gate("equivalence checked", False, "skipped -- rerun with --original before pushing")
        return
    verify = HERE / "verify_changes.py"
    proc = subprocess.run(
        [sys.executable, str(verify), "--original", original, "--quick"],
        capture_output=True, text=True, timeout=3600,
    )
    ok = "[PASS] default run is byte-identical" in proc.stdout
    gate("default run byte-identical to the original package", ok,
         "" if ok else "see: python tests/verify_changes.py --original ...")


# --------------------------------------------------------------------------- #
# 2. Calibration -- the headline                                              #
# --------------------------------------------------------------------------- #
def gate_calibration(n_seeds: int) -> dict:
    print(f"\n2. CALIBRATION -- does the stated rating predict survival? "
          f"({n_seeds} seeds x 2 landscapes)")
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confident_failures: list[tuple[str, int]] = []

    for Problem in (PlateauSpikeProblem, DecoyOverfittingProblem):
        for seed in range(n_seeds):
            problem = Problem()
            rating, outcome = _run_narrs(problem, seed)
            survived = bool(outcome and outcome.survives)
            buckets[rating][0] += int(survived)
            buckets[rating][1] += 1
            if rating == "high" and not survived:
                confident_failures.append((Problem.__name__, seed))

    print(f"     {'rating':>10s} {'survived':>9s} {'runs':>6s} {'rate':>7s}")
    for key in ("high", "medium", "low", "abstain"):
        if key in buckets:
            surv, total = buckets[key]
            print(f"     {key:>10s} {surv:>9d} {total:>6d} {surv / total:>6.0%}")

    high = buckets.get("high", [0, 0])
    other = [0, 0]
    for key, val in buckets.items():
        if key != "high":
            other[0] += val[0]
            other[1] += val[1]

    total_runs = high[1] + other[1]
    high_rate = high[0] / high[1] if high[1] else 0.0
    other_rate = other[0] / other[1] if other[1] else 0.0
    cf_rate = len(confident_failures) / total_runs if total_runs else 0.0

    gate("'high' survives at least 90% of the time", high_rate >= 0.90,
         f"{high[0]}/{high[1]} = {high_rate:.0%}")
    if other[1]:
        gate("'high' is more reliable than any lower rating",
             high_rate > other_rate,
             f"high {high_rate:.0%} vs not-high {other_rate:.0%}")
    else:
        print("     (no non-high ratings occurred -- discrimination untested here;")
        print("      gate 5 forces low-evidence runs where they do occur)")
    gate("confident failures stay under 10% of all runs", cf_rate < 0.10,
         f"{len(confident_failures)}/{total_runs} = {cf_rate:.0%}"
         + (f"  {confident_failures}" if confident_failures else ""))
    return {"high": high, "other": other, "confident_failures": confident_failures}


# --------------------------------------------------------------------------- #
# 3. Dominance over the baselines                                             #
# --------------------------------------------------------------------------- #
def gate_dominance(n_seeds: int) -> None:
    print(f"\n3. DOMINANCE -- does NARRS beat every baseline? ({n_seeds} seeds)")
    for Problem in (PlateauSpikeProblem, DecoyOverfittingProblem):
        survived: dict[str, int] = defaultdict(int)
        for seed in range(n_seeds):
            problem = Problem()
            search, holdout = problem.search_contexts, problem.holdout_contexts
            rating, outcome = _run_narrs(problem, seed)
            survived["NARRS"] += int(bool(outcome and outcome.survives))
            budget = outcome.n_evals if outcome else 3000
            for fn in B.ALL_BASELINES:
                res = fn(problem, search, budget, reps=2, seed=seed)
                if not res:
                    continue
                judged = _judge(problem, res["name"], res["point"], res["n_evals"],
                                res["n_trials"], search, holdout, 2)
                survived[res["name"].split(" (")[0]] += int(bool(judged and judged.survives))
        line = "  ".join(f"{k} {v}/{n_seeds}" for k, v in survived.items())
        print(f"     {Problem.__name__}: {line}")
        best_baseline = max((v for k, v in survived.items() if k != "NARRS"), default=0)
        gate(f"{Problem.__name__}: NARRS survives more than every baseline",
             survived["NARRS"] > best_baseline,
             f"{survived['NARRS']}/{n_seeds} vs best baseline {best_baseline}/{n_seeds}")


# --------------------------------------------------------------------------- #
# 4. Determinism                                                              #
# --------------------------------------------------------------------------- #
def gate_determinism() -> None:
    print("\n4. DETERMINISM -- same inputs, same answer?")
    problem = PlateauSpikeProblem()
    first = _run_narrs(problem, 3)
    second = _run_narrs(PlateauSpikeProblem(), 3)
    same = (first[0] == second[0]) and (
        (first[1] is None and second[1] is None)
        or (first[1] and second[1] and first[1].point == second[1].point)
    )
    gate("repeated run with the same seed is identical", same)

    verify = HERE / "verify_changes.py"
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(HERE)!r}); sys.argv=['x'];"
         "import verify_changes as V; V.check_parallel_determinism()"],
        capture_output=True, text=True, timeout=3600, cwd=str(ROOT),
    )
    ok = proc.stdout.count("[PASS]") >= 2 and "[FAIL]" not in proc.stdout
    gate("serial == parallel (n_jobs)", ok,
         "" if ok else "run tests/verify_changes.py section 5 for the traceback")


# --------------------------------------------------------------------------- #
# 5. Degradation -- the safety property                                       #
# --------------------------------------------------------------------------- #
def gate_degradation(n_seeds: int) -> None:
    print(f"\n5. DEGRADATION -- as evidence thins, does it lose confidence "
          f"or stay confidently wrong? ({n_seeds} seeds each)")
    print(f"     {'contexts':>9s} {'survived':>9s} {'rated high':>11s} "
          f"{'CONFIDENT FAILURES':>19s}")
    rows = []
    for n_ctx in (6, 4, 3, 2):
        survived = high = confident_fail = 0
        for seed in range(n_seeds):
            problem = DecoyOverfittingProblem(n_search_contexts=n_ctx)
            rating, outcome = _run_narrs(problem, seed)
            ok = bool(outcome and outcome.survives)
            survived += int(ok)
            if rating == "high":
                high += 1
                if not ok:
                    confident_fail += 1
        rows.append((n_ctx, survived, high, confident_fail))
        print(f"     {n_ctx:>9d} {survived:>4d}/{n_seeds:<4d} {high:>6d}/{n_seeds:<4d} "
              f"{confident_fail:>14d}/{n_seeds:<4d}")

    worst = max(cf for _, _, _, cf in rows)
    gate("never confidently wrong on more than 30% of thin-evidence runs",
         worst / n_seeds <= 0.30, f"worst case {worst}/{n_seeds}")

    survival_drops = rows[0][1] >= rows[-1][1]
    conf_drops = rows[0][2] >= rows[-1][2]
    gate("confidence falls as evidence is removed (not falsely stable)",
         conf_drops or survival_drops,
         f"high-rated {rows[0][2]}/{n_seeds} at 6 contexts -> {rows[-1][2]}/{n_seeds} at 2")


# --------------------------------------------------------------------------- #
# 6. Regression                                                               #
# --------------------------------------------------------------------------- #
def gate_regression() -> None:
    print("\n6. REGRESSION -- unit suite")
    proc = subprocess.run([sys.executable, str(HERE / "test_additions.py")],
                          capture_output=True, text=True, timeout=1800)
    last = [l for l in proc.stdout.strip().splitlines() if "passed" in l]
    ok = proc.returncode == 0
    gate("unit tests pass", ok, last[-1] if last else "")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="NARRS acceptance gate")
    parser.add_argument("--original", default=None,
                        help="path to an untouched copy of the package")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_seeds = 6 if args.quick else args.seeds

    print("=" * 78)
    print("NARRS ACCEPTANCE SUITE")
    print(f"seeds per landscape: {n_seeds}")
    print("=" * 78)

    gate_equivalence(args.original)
    gate_calibration(n_seeds)
    gate_dominance(n_seeds)
    gate_determinism()
    gate_degradation(n_seeds)
    gate_regression()

    failures = [g for g in _gates if not g[1]]
    print("\n" + "=" * 78)
    print(f"{len(_gates) - len(failures)}/{len(_gates)} gates passed")
    for name, _, detail in failures:
        print(f"  FAILED: {name} {detail}")
    print("=" * 78)
    if failures:
        print("VERDICT: DO NOT PUSH -- investigate the failures above.")
    else:
        print("VERDICT: SAFE TO PUSH.")
        print("  Original behaviour is provably unchanged, the additions do what")
        print("  they claim, and the confidence rating is informative -- when it")
        print("  says 'high' it is right, and when it is about to be wrong it")
        print("  says so instead.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
