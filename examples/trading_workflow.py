"""
The trading workflow: how NARRS is meant to be wired to a real strategy.

Random seeds vary NOISE. They do not vary slippage, latency, fees, or regime --
so a strategy that "survives ten seeds" has been tested against nothing a
market will actually do to it. This example shows the intended setup:

  1. WALK-FORWARD windows as contexts -- search on the EARLY windows, hold out
     the LAST ones. The holdout is the future relative to the search set, which
     makes the measured in->out decay a real estimate of live decay instead of
     an artifact of shuffling. It is also the only structural defence against
     the one failure no backtest can see: a regime shift after your data ends.
  2. COST STRESS -- the same windows scored under more than one cost world, so
     "robust" includes "robust to worse fills than you hoped for". Contexts are
     the seam: `SearchContext.data` carries the window AND the CostModel, and
     the objective consumes both. (For seed-based cost grids without walk-
     forward, `cost_stress_contexts` builds the whole set in one call.)
  3. A REAL PERFORMANCE BAR -- minimum_acceptable_holdout_score set to what you
     actually require, so "high" means good-and-stable, not just stable.
  4. Reading the report the way you would before risking money.

The "strategy" is a stand-in: a parametrised toy whose edge genuinely decays
over calendar time and genuinely suffers under worse costs. Swap
`simulate_strategy_returns` for your real backtest and everything else stays
the same.

Run it:
    python examples/trading_workflow.py
"""

import math
import random
import zlib

from narrs import (
    MetricRule,
    NARRSConfig,
    NoiseAwareRobustRegionSearch,
    ObjectiveResult,
    ParameterSpec,
    SearchContext,
)
from narrs.adapters.trading import CostModel, returns_to_score, walk_forward_contexts
from narrs.psurvive import estimate_from_region

TOTAL_DAYS = 1200          # ~5 years of daily bars
EDGE_CENTER = {"lookback": 30.0, "threshold": 0.55}


def _rng(*parts):
    # Stable across runs and workers -- never seed from hash() on strings.
    return random.Random(zlib.crc32("|".join(str(p) for p in parts).encode()))


def simulate_strategy_returns(params, window, rep):
    """GROSS daily returns for one (start, end) window. Replace with your backtest."""
    start, end = window
    lookback, threshold = params["lookback"], params["threshold"]
    # The edge: strongest near EDGE_CENTER, fading over calendar time
    # (alpha decay -- the thing walk-forward exists to measure).
    dist = math.sqrt(((lookback - EDGE_CENTER["lookback"]) / 25.0) ** 2
                     + ((threshold - EDGE_CENTER["threshold"]) / 0.35) ** 2)
    base_edge = 0.0022 * math.exp(-0.5 * (dist / 0.55) ** 2)
    rng = _rng("ret", start, end, round(lookback, 3), round(threshold, 3), rep)
    out = []
    for day in range(start, end):
        decay = 1.0 - 0.45 * (day / TOTAL_DAYS)
        out.append(base_edge * decay + rng.gauss(0, 0.01))
    return out


def objective(params, context, rep):
    """Consume what the context carries: a test window and a cost model.

    This is the whole adapter pattern -- `SearchContext.data` is opaque to the
    optimizer, so contexts can carry windows and cost worlds and the search
    machinery never changes.
    """
    data = context.data or {}
    window = data["test"]                      # score on the TEST slice
    cost: CostModel = data["cost"]
    gross = simulate_strategy_returns(params, window, rep)
    net = [cost.apply(g, trades=1.0) for g in gross]
    # Reduce the window to an annualised Sharpe. (returns_to_score also offers
    # "cvar" for a tail-only reduction when bad days are what disqualify.)
    score = returns_to_score(net, metric="sharpe")
    return ObjectiveResult(performance_metrics={"score": score},
                           sample_count=max(1, len(net)))


def main():
    # 1. Time-ordered windows with an embargo gap (embargo=0 quietly inflates
    #    every result on autocorrelated data).
    windows = walk_forward_contexts(n_periods=TOTAL_DAYS, n_windows=6,
                                    test_fraction=0.5, embargo=20)

    # 2. Compose cost stress onto the EARLY windows (search); the LAST windows
    #    become the holdout -- the future -- under realistic base costs.
    cost_worlds = [
        CostModel(slippage_bps=1.0, latency_seconds=5.0),
        CostModel(slippage_bps=4.0, latency_seconds=10.0),   # the bad-fill world
    ]
    search_ctx, holdout_ctx = [], []
    for ctx in windows[:-2]:
        for cost in cost_worlds:
            search_ctx.append(SearchContext(
                name=f"{ctx.name}|slip{cost.slippage_bps:g}",
                data={**ctx.data, "cost": cost},
            ))
    for ctx in windows[-2:]:
        holdout_ctx.append(SearchContext(
            name=f"{ctx.name}|oos",
            data={**ctx.data, "cost": cost_worlds[0]},
        ))

    result = NoiseAwareRobustRegionSearch(
        objective_function=objective,
        parameter_space=[ParameterSpec("lookback", 5, 120),
                         ParameterSpec("threshold", 0.1, 1.5)],
        search_contexts=search_ctx,
        holdout_contexts=holdout_ctx,
        # Sharpe -4..+4 maps to 0..1, so 0.5 on the normalised scale IS
        # Sharpe 0 -- break-even. The bar below therefore demands a genuinely
        # positive risk-adjusted edge on FUTURE windows, not just stability.
        metric_rules={"score": MetricRule(good_value=4.0, bad_value=-4.0)},
        metric_weights={"score": 1.0},
        config=NARRSConfig(
            search_rounds=3,
            initial_candidate_count=40,
            repetitions_per_point=2,
            total_compute_budget=6000,
            random_seed=7,
            # 3. The bar. 0.5 = break-even on the normalised scale above;
            #    demand better than break-even before trusting anything.
            minimum_acceptable_holdout_score=0.55,
        ),
    ).run()

    rep = result["confidence_report"]
    region = result["best_region"]

    print("=" * 78)
    print("READING THE REPORT LIKE MONEY IS ON THE LINE")
    print("=" * 78)

    if region is None:
        print("ABSTAINED -- nothing cleared the bar on FUTURE windows.")
        print("That is a result: this strategy family, under these costs, is not")
        print("deployable. Cheaper than finding out live.")
        return

    c = result["recommended_center"]
    ranges = {k: (round(v[0], 2), round(v[1], 2))
              for k, v in rep.best_region_parameter_ranges.items()}
    print(f"recommended params    : lookback={c['lookback']:.1f}, threshold={c['threshold']:.3f}")
    print(f"parameter ranges      : {ranges}")
    print(f"rating                : {rep.final_confidence_rating}")
    print(f"in-sample mean        : {region.region_in_sample_mean:.3f}")
    print(f"holdout (FUTURE) mean : {region.holdout_mean:.3f}")
    decay = region.region_in_sample_mean - region.holdout_mean
    print(f"walk-forward decay    : {decay:+.3f}   <- your live-decay estimate")
    est = estimate_from_region(region)
    print(f"p_survive             : {est.p_survive:.2f}   (sharp fragility warning)")
    if rep.warning_signs:
        print("warnings:")
        for w in rep.warning_signs:
            print(f"  - {w}")

    print()
    print("The deploy checklist this maps to:")
    print("  rating high AND decay small     -> deploy the REGION (any point in")
    print("                                     the ranges above, not one magic")
    print("                                     point), sized to the tail.")
    print("  rating medium, decay the reason -> the edge is fading; if you still")
    print("                                     deploy, monitor live returns")
    print("                                     against the decay number above")
    print("                                     and pull when it runs past it.")
    print("  abstain / low                   -> the market told you now instead")
    print("                                     of later. Keep the money.")


if __name__ == "__main__":
    main()
