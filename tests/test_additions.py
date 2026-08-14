"""
Tests for everything added on top of the original package.

Run with pytest, or directly:

    python tests/test_additions.py

The first section is the one that matters most: the additions must not change
what NARRS did before. A robustness feature that silently alters existing
results is a regression dressed up as an improvement.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narrs import (
    CVaRAggregator,
    MeanAggregator,
    MeanStdAggregator,
    NARRSConfig,
    NoiseAwareRobustRegionSearch,
    ObjectiveResult,
    MetricRule,
    ParameterSpec,
    SearchContext,
    WorstCaseAggregator,
    cvar,
    deflated_sharpe,
    oos_decay,
    pbo_cscv,
    resolve_aggregator,
    sharpe,
)
from narrs.adapters.trading import (
    CostModel,
    cost_stress_contexts,
    make_trading_objective,
    returns_to_score,
    walk_forward_contexts,
)
from narrs.psurvive import estimate_survival, rating_from_p_survive, fit_logistic
from narrs.benchmarks.problems import (
    DecoyOverfittingProblem,
    PlateauSpikeProblem,
    candidate_pool,
    performance_matrix,
)


# --------------------------------------------------------------------------- #
# Backward compatibility                                                       #
# --------------------------------------------------------------------------- #
def test_default_aggregator_is_exactly_min():
    """The default must reproduce the original min-over-contexts reduction."""
    agg = resolve_aggregator(NARRSConfig().context_aggregator, NARRSConfig().context_aggregator_alpha)
    assert agg.name == "worst_case"
    for _ in range(200):
        values = [random.uniform(-5, 5) for _ in range(random.randint(1, 12))]
        assert agg.aggregate(values) == min(values)


def test_default_config_unchanged_fields():
    """Existing knobs keep their original defaults."""
    c = NARRSConfig()
    assert c.search_rounds == 3
    assert c.worst_case_weight == 0.75
    assert c.noise_penalty_weight == 1.25
    assert c.random_seed == 42
    assert c.context_aggregator == "worst_case"


def test_optimizer_runs_with_no_config_changes():
    """A plain default run still completes and reports the aggregator used."""
    problem = PlateauSpikeProblem(n_search_contexts=4, n_holdout_contexts=4)
    opt = NoiseAwareRobustRegionSearch(
        objective_function=problem.objective,
        parameter_space=problem.parameter_space,
        search_contexts=problem.search_contexts,
        holdout_contexts=problem.holdout_contexts,
        metric_rules=problem.metric_rules,
        metric_weights=problem.metric_weights,
        config=NARRSConfig(search_rounds=1, initial_candidate_count=16,
                           repetitions_per_point=1, total_compute_budget=400),
    )
    result = opt.run()
    assert "confidence_report" in result
    assert result["confidence_report"].context_aggregator_used == "worst_case"


# --------------------------------------------------------------------------- #
# Aggregators                                                                  #
# --------------------------------------------------------------------------- #
def test_cvar_endpoints():
    values = [1.0, 2.0, 3.0, 4.0]
    assert abs(CVaRAggregator(1.0).aggregate(values) - 2.5) < 1e-9   # alpha=1 -> mean
    assert abs(CVaRAggregator(0.5).aggregate(values) - 1.5) < 1e-9   # worst half
    assert abs(CVaRAggregator(1e-9).aggregate(values) - 1.0) < 1e-6  # -> worst case


def test_cvar_never_exceeds_mean_and_never_below_min():
    for _ in range(200):
        values = [random.uniform(-3, 3) for _ in range(random.randint(2, 15))]
        for alpha in (0.1, 0.25, 0.5, 0.9):
            got = CVaRAggregator(alpha).aggregate(values)
            assert min(values) - 1e-9 <= got <= sum(values) / len(values) + 1e-9


def test_tail_aggregators_reject_fat_left_tail():
    steady = [1.0] * 10
    risky = [3.0] * 8 + [-5.0, -5.0]          # higher mean, catastrophic tail
    assert MeanAggregator().aggregate(risky) > MeanAggregator().aggregate(steady)
    assert CVaRAggregator(0.25).aggregate(risky) < CVaRAggregator(0.25).aggregate(steady)
    assert WorstCaseAggregator().aggregate(risky) < WorstCaseAggregator().aggregate(steady)


def test_mean_std_penalises_dispersion():
    tight, loose = [1.0, 1.0, 1.0], [3.0, 1.0, -1.0]
    agg = MeanStdAggregator(1.0)
    assert agg.aggregate(tight) > agg.aggregate(loose)


def test_weights_are_honoured():
    values = [0.0, 1.0]
    assert abs(MeanAggregator().aggregate(values, [3.0, 1.0]) - 0.25) < 1e-9


def test_empty_and_bad_input():
    assert WorstCaseAggregator().aggregate([]) == 0.0
    assert CVaRAggregator(0.5).aggregate([]) == 0.0
    for bad in ("nope", "cvarr"):
        try:
            resolve_aggregator(bad)
            raise AssertionError(f"expected failure for {bad}")
        except ValueError:
            pass
    for bad_alpha in (0.0, -0.1, 1.5):
        try:
            CVaRAggregator(bad_alpha)
            raise AssertionError("expected alpha validation")
        except ValueError:
            pass
    try:
        MeanAggregator().aggregate([1.0, 2.0], [1.0])
        raise AssertionError("expected length mismatch error")
    except ValueError:
        pass


def test_custom_aggregator_object_accepted():
    class Median:
        name = "median"

        def aggregate(self, values, weights=None):
            s = sorted(values)
            return s[len(s) // 2] if s else 0.0

    assert resolve_aggregator(Median()).name == "median"
    try:
        resolve_aggregator(object())
        raise AssertionError("expected type error for non-aggregator")
    except TypeError:
        pass


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def test_pbo_detects_pure_noise_selection():
    """All-noise candidates: in-sample winner should be a coin flip out of sample."""
    rng = random.Random(0)
    matrix = [[rng.gauss(0, 1) for _ in range(20)] for _ in range(64)]
    result = pbo_cscv(matrix, n_splits=8, metric=lambda xs: sum(xs) / len(xs))
    assert 0.30 < result["pbo"] < 0.70, result["pbo"]


def test_pbo_low_when_signal_is_real():
    """One candidate is genuinely better everywhere: selection should generalise."""
    rng = random.Random(1)
    matrix = []
    for _ in range(64):
        row = [rng.gauss(0, 1) for _ in range(19)]
        row.append(rng.gauss(6, 1))   # unmistakably better
        matrix.append(row)
    result = pbo_cscv(matrix, n_splits=8, metric=lambda xs: sum(xs) / len(xs))
    assert result["pbo"] < 0.05, result["pbo"]


def test_psurvive_flags_noisy_regions_as_fragile():
    """A clean region scores high; a noisy one scores low. Noise is the signal."""
    clean = estimate_survival(holdout_noise=0.04, context_instability=0.02,
                              in_sample_score=0.85, holdout_mean=0.84)
    noisy = estimate_survival(holdout_noise=0.45, context_instability=0.12,
                              in_sample_score=0.85, holdout_mean=0.60)
    # Thresholds match the SHARP (class-balanced) coefficient regime -- see the
    # DELIBERATE CHOICE note in narrs/psurvive.py. Sharp under-rates survivors a
    # little (clean ~0.74, not ~0.80) in exchange for actually flagging fragile
    # regions (noisy ~0.12, previously a useless 0.57). The property this test
    # protects is the SEPARATION, and it got much stronger.
    assert clean.p_survive > 0.60, clean.p_survive
    assert noisy.p_survive < 0.35, noisy.p_survive
    assert clean.p_survive - noisy.p_survive > 0.30, (clean.p_survive, noisy.p_survive)
    # probabilities stay in range and fragility index is bounded
    for e in (clean, noisy):
        assert 0.0 <= e.p_survive <= 1.0
        assert 0.0 <= e.fragility_index <= 1.0


def test_psurvive_monotonic_in_noise():
    """Increasing holdout noise must never raise p_survive."""
    prev = 1.1
    for noise in (0.02, 0.1, 0.2, 0.35, 0.5, 0.8):
        p = estimate_survival(noise, 0.03, 0.8, 0.78).p_survive
        assert p <= prev + 1e-9, f"p_survive rose with noise at {noise}"
        prev = p


def test_rating_from_p_survive_bands():
    assert rating_from_p_survive(0.95) == "high"
    assert rating_from_p_survive(0.6) == "medium"
    assert rating_from_p_survive(0.2) == "low"


def test_fit_logistic_recovers_a_known_boundary():
    """Sanity: on separable data the fit puts the coefficient in the right direction."""
    rows = []
    for i in range(200):
        noise = (i % 20) / 20.0            # 0..1
        survived = noise < 0.4             # clean survives, noisy fails
        rows.append({"holdout_noise": noise, "survived": survived})
    coeffs = fit_logistic(rows, signal_keys=("holdout_noise",))
    assert coeffs["holdout_noise"] < 0, "more noise should lower survival odds"


def test_nan_never_scores_as_perfect():
    """Regression: MetricRule.normalize(nan) returned 1.0 -- the best possible score.

    Python's min/max compare False against NaN, so max(0.0, min(1.0, nan)) is
    1.0. A failed metric calculation was therefore ranked top and confidently
    recommended. NaN and infinity must be the worst case, and must count as
    hard-rule violations so the observation is discarded rather than averaged in.
    """
    rule = MetricRule(good_value=1.0, bad_value=0.0, higher_is_better=True)
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert rule.normalize(bad) == 0.0, bad
        assert rule.violates_hard_rule(bad) is True, bad
    # the ordinary path is untouched
    assert rule.normalize(0.5) == 0.5
    assert rule.normalize(2.0) == 1.0
    assert rule.normalize(-1.0) == 0.0
    assert rule.violates_hard_rule(0.5) is False

    inverted = MetricRule(good_value=0.0, bad_value=1.0, higher_is_better=False)
    assert inverted.normalize(float("nan")) == 0.0


def test_nan_objective_does_not_produce_a_confident_answer():
    """An objective that always fails must abstain, not recommend something."""
    problem = PlateauSpikeProblem(n_search_contexts=4, n_holdout_contexts=4)

    def nan_objective(params, context, rep):
        return ObjectiveResult(performance_metrics={"score": float("nan")},
                               sample_count=1)

    opt = NoiseAwareRobustRegionSearch(
        objective_function=nan_objective,
        parameter_space=problem.parameter_space,
        search_contexts=problem.search_contexts,
        holdout_contexts=problem.holdout_contexts,
        metric_rules=problem.metric_rules,
        metric_weights=problem.metric_weights,
        config=NARRSConfig(search_rounds=1, initial_candidate_count=12,
                           repetitions_per_point=1, total_compute_budget=300),
    )
    result = opt.run()
    assert result["recommended_center"] is None, "a NaN objective must not yield a pick"


def test_thin_evidence_is_warned_about():
    """A single search context measures no cross-context robustness -- say so."""
    problem = PlateauSpikeProblem(n_search_contexts=4, n_holdout_contexts=4)
    opt = NoiseAwareRobustRegionSearch(
        objective_function=problem.objective,
        parameter_space=problem.parameter_space,
        search_contexts=[SearchContext("only")],
        holdout_contexts=problem.holdout_contexts,
        metric_rules=problem.metric_rules,
        metric_weights=problem.metric_weights,
        config=NARRSConfig(search_rounds=1, initial_candidate_count=12,
                           repetitions_per_point=2, total_compute_budget=400),
    )
    warnings = opt.run()["confidence_report"].warning_signs
    assert any("one search context" in w for w in warnings), warnings


def test_parameter_insensitive_objective_is_flagged():
    """If every point scores the same, the objective probably ignores the params."""
    problem = PlateauSpikeProblem(n_search_contexts=4, n_holdout_contexts=4)
    opt = NoiseAwareRobustRegionSearch(
        objective_function=lambda p, c, r: ObjectiveResult(
            performance_metrics={"score": 1.0}, sample_count=1),
        parameter_space=problem.parameter_space,
        search_contexts=problem.search_contexts,
        holdout_contexts=problem.holdout_contexts,
        metric_rules=problem.metric_rules,
        metric_weights=problem.metric_weights,
        config=NARRSConfig(search_rounds=1, initial_candidate_count=16,
                           repetitions_per_point=1, total_compute_budget=500),
    )
    warnings = opt.run()["confidence_report"].warning_signs
    assert any("insensitive to the parameters" in w for w in warnings), warnings


def test_scalable_problems_hold_their_shape_across_dimensions():
    from narrs.benchmarks.scalable import ScalableDecoy, ScalablePlateauSpike
    for dim in (1, 2, 5, 8):
        p = ScalablePlateauSpike(dim=dim)
        names = [s.name for s in p.parameter_space]
        assert len(names) == dim
        plateau = {n: 0.30 for n in names}
        spike = {n: 0.75 for n in names}
        assert p.region_kind(plateau) == "plateau"
        assert p.region_kind(spike) == "spike"
        ctxs = p.search_contexts
        pm = [sum(p.raw_score(plateau, c, r) for r in range(3)) / 3 for c in ctxs]
        sm = [sum(p.raw_score(spike, c, r) for r in range(3)) / 3 for c in ctxs]
        assert sum(sm) / len(sm) > sum(pm) / len(pm)   # spike lures on mean
        assert min(sm) < min(pm)                        # and loses on the tail

        d = ScalableDecoy(dim=dim, n_decoys=10)
        assert len(d.parameter_space) == dim
        assert d.region_kind({n: 0.28 for n in names}) == "true_edge"


def test_pbo_ties_are_indifference_not_overfitting():
    """Identical candidates must give 0.50, not 1.00.

    Regression: exact ties were counted as certain overfitting, so a landscape
    where every candidate is the same reported maximum overfitting.
    """
    identical = [[1.0] * 10 for _ in range(32)]
    result = pbo_cscv(identical, n_splits=8, metric=lambda xs: sum(xs) / len(xs))
    assert abs(result["pbo"] - 0.5) < 1e-9, result["pbo"]


def test_pbo_high_when_every_candidate_reverses():
    matrix = []
    for t in range(64):
        sign = 1.0 if t < 32 else -1.0
        matrix.append([sign * (j + 1) for j in range(10)])
    result = pbo_cscv(matrix, n_splits=8, metric=lambda xs: sum(xs) / len(xs))
    assert result["pbo"] > 0.6, result["pbo"]


def test_cma_baseline_survives_tiny_budgets():
    """Regression: truncating a CMA generation to fit the budget crashed the run."""
    from narrs.benchmarks.baselines import cma_es
    problem = PlateauSpikeProblem(n_search_contexts=4)
    for budget in (1, 5, 40, 400):
        result = cma_es(problem, problem.search_contexts, budget, reps=1, seed=0)
        if result is None:
            return  # cma not installed
        assert result["point"] is not None
        assert set(result["point"]) == {"x", "y"}


def test_pbo_input_validation():
    for bad in ([], [[1.0]], [[1.0, 2.0], [1.0]]):
        try:
            pbo_cscv(bad)
            raise AssertionError(f"expected rejection of {bad}")
        except ValueError:
            pass


def test_deflated_sharpe_falls_with_more_trials():
    rng = random.Random(3)
    returns = [rng.gauss(0.05, 1.0) for _ in range(250)]
    few = deflated_sharpe(returns, n_trials=2)["dsr"]
    many = deflated_sharpe(returns, n_trials=5000)["dsr"]
    assert many < few


def test_cvar_and_sharpe_helpers():
    assert abs(cvar([1, 2, 3, 4], 0.5) - 1.5) < 1e-9
    assert sharpe([1.0]) == 0.0
    assert sharpe([2.0, 2.0]) == 0.0          # zero dispersion -> undefined, not inf
    decay = oos_decay([1.0, 1.0], [0.5, 0.5])
    assert abs(decay["decay"] - 0.5) < 1e-9
    assert abs(decay["retention"] - 0.5) < 1e-9
    assert oos_decay([-1.0], [0.5])["retention"] is None


# --------------------------------------------------------------------------- #
# Benchmark problems                                                           #
# --------------------------------------------------------------------------- #
def test_landscapes_are_reproducible():
    """Scores must not depend on PYTHONHASHSEED or process identity."""
    p = PlateauSpikeProblem()
    ctx = p.search_contexts[0]
    a = p.raw_score({"x": 0.3, "y": 0.3}, ctx, 0)
    b = p.raw_score({"x": 0.3, "y": 0.3}, ctx, 0)
    assert a == b
    assert PlateauSpikeProblem().raw_score({"x": 0.3, "y": 0.3}, ctx, 0) == a


def test_risk_trap_has_the_intended_shape():
    """Spike must beat plateau on mean and lose badly on the tail."""
    p = PlateauSpikeProblem()
    ctxs = p.search_contexts
    def per_context(pt):
        return [sum(p.raw_score(pt, c, r) for r in range(3)) / 3 for c in ctxs]
    plateau = per_context({"x": 0.30, "y": 0.30})
    spike = per_context({"x": 0.75, "y": 0.75})
    assert sum(spike) / len(spike) > sum(plateau) / len(plateau)   # lures mean-optimizers
    assert min(spike) < min(plateau)                                # punishes the tail
    assert p.region_kind({"x": 0.30, "y": 0.30}) == "plateau"
    assert p.region_kind({"x": 0.75, "y": 0.75}) == "spike"


def test_overfitting_trap_decays_out_of_sample():
    p = DecoyOverfittingProblem()
    ins, oos = p.search_contexts, p.holdout_contexts
    def mean_over(pt, ctxs):
        return sum(p.raw_score(pt, c, 0) for c in ctxs) / len(ctxs)
    best = max(p._decoys, key=lambda d: mean_over(d, ins))
    assert mean_over(best, ins) - mean_over(best, oos) > 0.1     # decoy evaporates
    true_pt = {"x": 0.28, "y": 0.28}
    assert abs(mean_over(true_pt, ins) - mean_over(true_pt, oos)) < 0.1  # edge holds


def test_performance_matrix_shape():
    p = PlateauSpikeProblem(n_search_contexts=5)
    pool = candidate_pool(p, 7, seed=0)
    m = performance_matrix(p, pool, p.search_contexts, reps=2)
    assert len(m) == 10 and all(len(r) == 7 for r in m)
    assert all(0.0 <= v <= 1.0 for row in m for v in row)


def test_survival_threshold_is_derived_not_tuned():
    assert PlateauSpikeProblem().survival_threshold == 0.5
    assert DecoyOverfittingProblem().survival_threshold == 0.5


# --------------------------------------------------------------------------- #
# Trading adapter                                                              #
# --------------------------------------------------------------------------- #
def test_cost_model_reduces_return():
    m = CostModel(slippage_bps=5, fee_bps=2)
    assert m.apply(0.01) < 0.01
    assert abs(m.apply(0.01) - (0.01 - 7e-4)) < 1e-12


def test_latency_only_matters_with_decay():
    slow, fast = CostModel(latency_seconds=60), CostModel(latency_seconds=1)
    assert slow.apply(0.01) == fast.apply(0.01)                       # no decay -> no effect
    assert slow.apply(0.01, edge_decay_per_second=0.01) < fast.apply(0.01, edge_decay_per_second=0.01)


def test_context_builders():
    ctxs = cost_stress_contexts([0, 1], [1.0, 5.0], [7.0])
    assert len(ctxs) == 4
    assert all(isinstance(c.data["cost"], CostModel) for c in ctxs)
    assert len({c.name for c in ctxs}) == 4                            # names unique

    wf = walk_forward_contexts(100, n_windows=5, embargo=2)
    assert len(wf) == 5
    for c in wf:
        assert c.data["train"][1] + c.data["embargo"] <= c.data["test"][0]
    assert walk_forward_contexts(0) == []


def test_returns_to_score_variants():
    returns = [0.01, -0.02, 0.03, 0.005, -0.001]
    assert returns_to_score(returns, "cvar", alpha=0.4) < 0
    assert abs(returns_to_score(returns, "total") - sum(returns)) < 1e-12
    assert returns_to_score([], "sharpe") == 0.0
    assert returns_to_score([0.01, 0.01], "sharpe") == 0.0        # no dispersion
    try:
        returns_to_score(returns, "nonsense")
        raise AssertionError("expected unknown-metric error")
    except ValueError:
        pass


def test_trading_objective_applies_costs():
    def strat(params, seed):
        rng = random.Random(seed)
        return [params["edge"] * 0.001 + rng.gauss(0, 0.002) for _ in range(120)]

    obj = make_trading_objective(strat, metric="sharpe")
    cheap = SearchContext("cheap", {"seed": 1, "cost": CostModel(0.5, 0.5)})
    dear = SearchContext("dear", {"seed": 1, "cost": CostModel(40.0, 20.0)})
    r_cheap = obj({"edge": 2.0}, cheap, 0)
    r_dear = obj({"edge": 2.0}, dear, 0)
    assert r_cheap.performance_metrics["score"] > r_dear.performance_metrics["score"]
    assert r_cheap.sample_count == 120


def test_trading_objective_works_without_cost_model():
    def strat(params, seed):
        return [0.001] * 50
    obj = make_trading_objective(strat, metric="total")
    result = obj({"a": 1.0}, SearchContext("plain", None), 0)
    assert abs(result.performance_metrics["score"] - 0.05) < 1e-9


# --------------------------------------------------------------------------- #
# End-to-end                                                                   #
# --------------------------------------------------------------------------- #
def test_narrs_avoids_the_risk_trap_end_to_end():
    problem = PlateauSpikeProblem()
    opt = NoiseAwareRobustRegionSearch(
        objective_function=problem.objective,
        parameter_space=problem.parameter_space,
        search_contexts=problem.search_contexts,
        holdout_contexts=problem.holdout_contexts,
        metric_rules=problem.metric_rules,
        metric_weights=problem.metric_weights,
        config=NARRSConfig(search_rounds=2, initial_candidate_count=40,
                           repetitions_per_point=2, total_compute_budget=3000,
                           context_aggregator="cvar", context_aggregator_alpha=0.25,
                           random_seed=1),
    )
    result = opt.run()
    center = result["recommended_center"]
    assert center is not None, "expected a surviving region"
    assert problem.region_kind(center) == "plateau", center
    assert result["confidence_report"].context_aggregator_used == "cvar_0.25"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
