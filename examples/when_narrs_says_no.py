"""
When NARRS says no.

Every other example shows NARRS succeeding. This one shows the part that
actually makes it trustworthy: REFUSING. The same optimizer, the same config,
three landscapes --

  1. A healthy landscape.       NARRS should say "high" and mean it.
  2. A regime shift.            The optimum MOVES between the search contexts
                                and the holdout contexts -- a strategy fit on
                                2023 deployed into 2024. In-sample robustness
                                is real; it just doesn't transfer. NARRS should
                                cap the rating and say REGIME SHIFT.
  3. Correlated contexts.       Twelve contexts that are near-duplicates
                                carrying ~two contexts of information, with a
                                holdout drawn from regimes the search never
                                saw. Survival here is impossible; the only
                                honest output is refusal.

A "high" rating is only worth something if it is withheld when it should be.
This demo is the evidence that it is. The landscapes come from
narrs/benchmarks/adversarial.py, the same ones the permanent stress-suite axis
runs (tests/stress_suite.py --axis adversarial).

Run it:
    python examples/when_narrs_says_no.py
"""

from narrs import NARRSConfig, NoiseAwareRobustRegionSearch
from narrs.benchmarks.adversarial import CorrelatedContexts, RegimeShift
from narrs.benchmarks.scalable import ScalablePlateauSpike
from narrs.psurvive import estimate_from_region


def h(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_one(problem, seed=0):
    """One NARRS run with the same honest config on any benchmark problem."""
    opt = NoiseAwareRobustRegionSearch(
        objective_function=problem.objective,
        parameter_space=problem.parameter_space,
        search_contexts=problem.search_contexts,
        holdout_contexts=problem.holdout_contexts,
        metric_rules=problem.metric_rules,
        metric_weights=problem.metric_weights,
        config=NARRSConfig(
            search_rounds=3,
            initial_candidate_count=40,
            repetitions_per_point=2,
            total_compute_budget=5000,
            random_seed=seed,
            # Always set the bar: 0.5 = the midpoint of the metric range.
            # Without it, "high" only means STABLE, not GOOD.
            minimum_acceptable_holdout_score=0.5,
        ),
    )
    return opt.run()


def report_one(result):
    rep = result["confidence_report"]
    region = result["best_region"]
    if region is None:
        print("  verdict            : ABSTAINED -- no region worth recommending")
        return
    print(f"  rating             : {rep.final_confidence_rating}")
    print(f"  in-sample mean     : {region.region_in_sample_mean:.3f}")
    print(f"  holdout mean (OOS) : {region.holdout_mean:.3f}")
    decay = region.region_in_sample_mean - region.holdout_mean
    print(f"  in->out decay      : {decay:+.3f}")
    est = estimate_from_region(region)
    print(f"  p_survive          : {est.p_survive:.2f}   (sharp fragility warning)")
    if rep.warning_signs:
        print("  warnings:")
        for w in rep.warning_signs:
            print(f"    - {w}")


def main():
    h("1. HEALTHY LANDSCAPE -- a broad genuine plateau. Expect: high.")
    print("A plateau that is genuinely stable in and out of sample. This is the")
    print("case NARRS is allowed to be confident about, and it should be.\n")
    report_one(run_one(ScalablePlateauSpike()))

    h("2. REGIME SHIFT -- the optimum moves out of sample. Expect: capped + warned.")
    print("Search contexts have their optimum in one place; holdout contexts have")
    print("it somewhere else. NARRS measures the in-sample -> out-of-sample drop")
    print("(both on the same normalised scale) and refuses 'high' when the region")
    print("does not transfer. Watch the decay number and the REGIME SHIFT warning.\n")
    report_one(run_one(RegimeShift(shift=0.55)))

    h("3. CORRELATED CONTEXTS -- fake evidence. Expect: refusal.")
    print("Twelve search contexts that are near-copies of each other (think")
    print("overlapping backtest windows), and a holdout from regimes the search")
    print("never saw. Every statistic computed as if n=12 is a lie; true")
    print("out-of-sample performance here is negative. The only honest answer")
    print("is a refusal -- no 'high' rating, loud warnings.\n")
    report_one(run_one(CorrelatedContexts()))

    h("A NUANCE WORTH NOTICING")
    print("  In case 2 p_survive stays high even though the rating was capped.")
    print("  That is correct behaviour, not a contradiction: the region IS")
    print("  internally stable -- low noise, steady across its own contexts --")
    print("  so the noise-driven fragility model has nothing to flag. What is")
    print("  wrong is that it does not TRANSFER, and that is exactly what the")
    print("  decay gate on the label catches. Two failure modes, two detectors:")
    print("  p_survive hears noisy/fragile, the label's gates see non-transfer.")

    h("WHY THIS MATTERS")
    print("  A 'high' from NARRS is informative precisely because cases 2 and 3")
    print("  do not get one. Across the permanent adversarial suite, the cells")
    print("  where survival is impossible show ZERO high ratings -- refusing")
    print("  confidence it has not earned is the product.")
    print("\n  The irreducible case no backtest can catch: a regime shift AFTER")
    print("  your holdout window. Mitigate it structurally -- make the holdout")
    print("  the FUTURE relative to the search set (walk_forward_contexts in")
    print("  narrs.adapters.trading), so measured decay estimates live decay.")


if __name__ == "__main__":
    main()
