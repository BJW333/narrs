"""
How NARRS works, narrated end to end on a small problem you can hold in your head.

Run it:

    python examples/how_it_works.py

This is a teaching script, not a test. It builds a deliberately simple 2-D
problem with a known right answer, runs the real optimizer on it, and stops at
each stage to print what NARRS just did and why -- the raw evaluations, the
robustness scoring, the spike rejection, the holdout validation, the confidence
number. Read it top to bottom once and the rest of the package stops being a
black box.

The problem: two candidate regions.
  * a BROAD, STABLE plateau -- lower peak, but its score barely moves across
    contexts. This is what you want to deploy.
  * a TALL, FRAGILE spike -- higher peak on average, but it collapses in some
    contexts. This is the trap a naive optimizer walks into.
NARRS should recommend the plateau and tell you it is confident.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narrs import (MetricRule, NARRSConfig, NoiseAwareRobustRegionSearch,
                   ObjectiveResult, ParameterSpec, SearchContext)
from narrs.psurvive import estimate_from_region

RULE = "\033[2m" + "-" * 78 + "\033[0m" if sys.stdout.isatty() else "-" * 78


def h(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# the toy landscape                                                           #
# --------------------------------------------------------------------------- #
PLATEAU = (0.30, 0.30)   # broad, stable, height 0.80
SPIKE = (0.72, 0.72)     # tall (2.5) but crashes to -2.0 in ~1 context in 4
CRASH_CONTEXTS = {"regime_2"}


def objective(params, context, rep):
    x, y = params["x"], params["y"]
    dp = math.dist((x, y), PLATEAU)
    plateau = 0.80 * math.exp(-0.5 * (dp / 0.18) ** 2)
    ds = math.dist((x, y), SPIKE)
    payoff = -2.0 if context.name in CRASH_CONTEXTS else 2.5
    spike = payoff * math.exp(-0.5 * (ds / 0.12) ** 2)
    # tiny reproducible noise
    import random, zlib
    # NOT hash(): Python randomises string hashing per process, so hash-seeded
    # noise differs between runs and between parallel workers. crc32 is stable.
    key = f"{context.name}|{rep}|{x:.4f}|{y:.4f}".encode()
    rng = random.Random(zlib.crc32(key))
    return ObjectiveResult(
        performance_metrics={"score": plateau + spike + rng.gauss(0, 0.03)},
        sample_count=1)


def main() -> None:
    h("NARRS, end to end")
    print(__doc__.strip().split("\n\n", 1)[1])

    contexts = [SearchContext(f"regime_{i}") for i in range(4)]
    holdout = [SearchContext(f"oos_{i}") for i in range(6)]

    h("STEP 1  -  the setup you hand NARRS")
    print("  parameters : x in [0,1], y in [0,1]  (the knobs to tune)")
    print("  contexts   : 4 search regimes + 6 held-out regimes")
    print("               one search regime (regime_2) secretly crashes the spike")
    print("  metric     : raw score mapped to 0..1, good=2.5  bad=-2.0")
    print("\n  Ground truth we are about to make NARRS discover on its own:")
    print(f"    plateau at {PLATEAU}  -- lower but stable  (the right answer)")
    print(f"    spike   at {SPIKE}  -- higher mean but crashes (the trap)")

    # score the two known points by hand so the reader sees the tension
    def per_context(pt):
        return [sum(objective({"x": pt[0], "y": pt[1]}, c, r).performance_metrics["score"]
                    for r in range(4)) / 4 for c in contexts]
    pl, sp = per_context(PLATEAU), per_context(SPIKE)
    h("STEP 2  -  why a naive 'pick the highest average' optimizer fails here")
    print(f"  plateau per-context means : {[round(v,2) for v in pl]}")
    print(f"    mean {sum(pl)/len(pl):.2f}   worst {min(pl):.2f}")
    print(f"  spike   per-context means : {[round(v,2) for v in sp]}")
    print(f"    mean {sum(sp)/len(sp):.2f}   worst {min(sp):.2f}")
    print("\n  The spike wins on MEAN and loses badly on WORST-CASE. A standard")
    print("  optimizer maximises the mean and walks straight into the crash.")
    print("  NARRS scores the worst-case (or CVaR) across contexts instead.")

    h("STEP 3  -  run the real optimizer")
    print("  Building NoiseAwareRobustRegionSearch and calling .run() ...")
    opt = NoiseAwareRobustRegionSearch(
        objective_function=objective,
        parameter_space=[ParameterSpec("x", 0.0, 1.0), ParameterSpec("y", 0.0, 1.0)],
        search_contexts=contexts, holdout_contexts=holdout,
        metric_rules={"score": MetricRule(good_value=2.5, bad_value=-2.0)},
        metric_weights={"score": 1.0},
        config=NARRSConfig(search_rounds=3, initial_candidate_count=40,
                           repetitions_per_point=3, total_compute_budget=6000,
                           random_seed=1, context_aggregator="cvar"),
    )
    result = opt.run()
    report = result["confidence_report"]

    h("STEP 4  -  what happened inside the search")
    print(f"  points evaluated        : {report.number_of_tested_points}")
    print(f"  search rounds completed : {report.search_rounds_completed}")
    print(f"  objective calls spent   : {report.total_objective_calls}")
    print("\n  Points NARRS rejected, and why each matters:")
    print(f"    {report.number_of_rejected_spikes:>3d}  fragile spikes  (great in-sample, collapse in one regime)")
    print(f"    {report.number_of_rejected_noisy_points:>3d}  too noisy       (score you can't trust)")
    print(f"    {report.number_of_rejected_low_evidence_points:>3d}  thin evidence   (not enough reps to judge)")
    print(f"    {report.number_of_rejected_small_regions:>3d}  too small       (a knife-edge, not a robust region)")
    print("\n  Rejection is the point. A naive optimizer keeps the tall spike;")
    print("  NARRS throws it out precisely because it isn't robust.")

    center = result["recommended_center"]
    region = result["best_region"]
    h("STEP 5  -  the region it recommends")
    if center is None:
        print("  NARRS abstained -- no region was robust enough. (Not expected here.)")
        return
    landed = "PLATEAU (correct)" if math.dist((center["x"], center["y"]), PLATEAU) < 0.2 \
        else "SPIKE (the trap!)" if math.dist((center["x"], center["y"]), SPIKE) < 0.2 \
        else "somewhere else"
    print(f"  recommended center : x={center['x']:.3f}, y={center['y']:.3f}   -> {landed}")
    print(f"  region score (in-sample) : {region.region_score:.3f}")
    print(f"  holdout mean (out-of-sample): {region.holdout_mean:.3f}")
    print(f"  holdout noise            : {region.holdout_noise:.3f}   (low = stable)")
    print(f"  context instability      : {region.region_context_stability:.3f}   (low = regime-independent)")

    h("STEP 6  -  how much to trust it")
    print(f"  label (high/medium/low)  : {report.final_confidence_rating}")
    est = estimate_from_region(region)
    print(f"  p_survive (calibrated)   : {est.p_survive:.2f}")
    print(f"  fragility index          : {est.fragility_index:.2f}  (0 robust .. 1 fragile)")
    print("\n  p_survive is the probability this region stays good when the regime")
    print("  changes, calibrated against real survival on the benchmark battery.")
    print("  It is driven mostly by out-of-sample noise -- the single best")
    print("  predictor of whether a region holds up live.")
    if report.warning_signs:
        print("\n  warnings NARRS attached to this result:")
        for w in report.warning_signs:
            print(f"    - {w}")

    h("SUMMARY")
    print("  NARRS explored a noisy landscape, refused the tall-but-fragile spike,")
    print("  recommended the broad stable plateau, and told you how much to trust it")
    print("  with a calibrated probability. That is the whole idea: not the highest")
    print("  score, but the score that survives contact with reality.")
    print("\n  Next: examples/run_battery.py (proof vs baselines) or")
    print("  tests/stress_suite.py (where it starts to break).")


if __name__ == "__main__":
    main()
