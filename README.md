# NARRS — Noise-Aware Robust Region Search

A black-box optimizer that finds **stable, high-performing *regions* of parameter space, not single overfit points.** It's built for noisy objectives — trading backtests, simulations, ML hyperparameters — where the highest-scoring point is often a fragile fluke and what you actually want is a broad basin that keeps working out of sample.

Pure standard library (with an optional `scipy` acceleration). No heavyweight dependencies required.

## Why regions instead of points

On a landscape with a broad noisy plateau and a taller razor-thin spike, a normal optimizer chases the spike — the single best-looking evaluation — and then it evaporates when you re-run it or move to new data. NARRS deliberately rejects that spike and recommends the plateau.

This isn't a heuristic; it's the **flat-minima principle**: solutions in flat, low-curvature regions of the loss surface are more robust to perturbation and generalize better than sharp ones, an idea formalized decades ago as seeking "large regions of connected acceptable minima." NARRS computes exactly those regions and reports how much to trust each one.

## Install

```bash
# from the repo root
pip install -e .

# optional: enable the fast KD-tree neighbor backend (recommended for larger searches)
pip install -e ".[fast]"     # pulls scipy
```

Without `scipy`, NARRS still runs correctly using a pure-Python fallback — just slower on large candidate sets. Requires Python 3.9+.

## Quickstart

```python
import random
from narrs import (
    NoiseAwareRobustRegionSearch,
    NARRSConfig,
    ParameterSpec,
    MetricRule,
    SearchContext,
    ObjectiveResult,
)

# Your noisy objective: score the same params under several "contexts"
# (e.g. time periods / market regimes / random seeds).
def objective(params, context, rep):
    seed_offset = {"2019": 10, "2020": 20, "2021": 30, "oos_a": 40, "oos_b": 50}
    rng = random.Random(rep + seed_offset.get(context.name, 0))
    x, y = params["x"], params["y"]
    # Broad, stable plateau near (2, -1); NARRS should prefer it over any sharp spike.
    score = max(0.0, 1.0 - 0.12 * ((x - 2.0) ** 2 + 0.5 * (y + 1.0) ** 2))
    score += rng.gauss(0, 0.03)                       # observation noise
    return ObjectiveResult(performance_metrics={"score": score}, sample_count=1)

optimizer = NoiseAwareRobustRegionSearch(
    objective_function=objective,
    parameter_space=[ParameterSpec("x", -5, 5), ParameterSpec("y", -5, 5, weight=1.5)],
    search_contexts=[SearchContext("2019"), SearchContext("2020"), SearchContext("2021")],
    holdout_contexts=[SearchContext("oos_a"), SearchContext("oos_b")],
    metric_rules={"score": MetricRule(good_value=1.0, bad_value=0.0, higher_is_better=True)},
    metric_weights={"score": 1.0},
    config=NARRSConfig(
        search_rounds=3,
        total_compute_budget=2000,
        # Scores are normalised 0..1 (0 = your bad_value, 1 = your good_value).
        # Set the score a region must actually CLEAR to be trusted -- see
        # "Stable is not the same as good" below for why leaving this at the
        # 0.0 default is almost never what you want in production.
        minimum_acceptable_holdout_score=0.5,
        n_jobs=1,
        random_seed=42,
    ),
)

result = optimizer.run()
report = result["confidence_report"]

print("Recommended center:", result["recommended_center"])
print("Confidence rating :", report.final_confidence_rating)
print("p_survive         :", report.p_survive, "(fragility warning -- see below)")
print("Objective calls   :", report.total_objective_calls,
      f"(search {report.search_evaluations} + holdout {report.holdout_evaluations})")
```

For a narrated end-to-end walkthrough on a toy landscape, run
`python examples/how_it_works.py`. With `matplotlib` installed,
`python examples/visualize.py` renders the landscape, calibration,
noise-vs-survival, and decay figures.

## Core concepts

- **`objective_function(params, context, rep)`** — you provide this. It scores one parameter point under one context and returns an `ObjectiveResult`. `rep` is the repetition index, so you can inject fresh noise per call.
- **`ParameterSpec(name, min_value, max_value, weight=1.0)`** — one per tunable parameter. `weight` scales how much that dimension counts toward "nearby" when forming regions.
- **`SearchContext(name, data=None)`** — a scenario the strategy is scored against. Multiple contexts (time periods, regimes, seeds) are how NARRS measures *stability across conditions*, not just average performance.
- **holdout contexts** — contexts held out of the search and used only to validate the winning region out of sample. Confidence is capped unless a region survives them.
- **`MetricRule(good_value, bad_value, higher_is_better=True, hard_min=None, hard_max=None)`** — maps a raw metric to a 0–1 score. `hard_min`/`hard_max` are rejection limits (e.g. a max-drawdown ceiling); a point violating one is discarded outright.
- **`metric_weights`** — how to combine multiple normalized metrics into a single score.

## What `run()` returns

A dict:

| key | meaning |
|---|---|
| `recommended_center` | the parameter point at the center of the best stable region (this is your answer) |
| `best_region` | the winning `Region` object |
| `robust_regions` | all surviving regions |
| `optional_region_ensemble` | a small ensemble of strong regions, if you want diversification |
| `confidence_report` | a `ConfidenceReport` — see below |

The **`ConfidenceReport`** carries the recommendation plus everything you need to trust it: `final_confidence_rating`, `p_survive`, `fragility_index`, `best_region_parameter_ranges`, `number_of_surviving_regions`, `warning_signs`, honest call accounting (`search_evaluations`, `holdout_evaluations`, `total_objective_calls`), and the selection-adjustment fields (`number_of_trials_considered`, `deflated_holdout_threshold`, `holdout_selection_penalty`).

## Confidence ratings

The rating is deliberately conservative — it's a statement about out-of-sample trust, not in-sample score. A region is rated **`high`** only if it clears *all* of:

1. **The holdout bar, selection-adjusted.** Its holdout mean beats a Deflated-Sharpe-style bar raised by the expected best-of-N fluke from however much searching was done (`holdout_selection_penalty` shows the haircut).
2. **The noise ceiling — measured twice.** `holdout_noise` must be low, *and* so must `region_noise`, the in-sample spread backed by every search point. The second check exists because a holdout noise reading from a few repetitions on a loud objective can look quiet by pure luck — adversarial testing found exactly this failure, with loud landscapes reading holdout noise well under the cap and then dying out of sample.
3. **No regime shift.** The region's in-sample mean (`region_in_sample_mean`, same normalised scale as `holdout_mean`) may not sit more than `maximum_in_to_out_decay` above its holdout mean. A large drop means the holdout behaves like a different world than the search set — the region may be perfectly stable in each and still not transfer. When this fires you get an explicit `REGIME SHIFT:` warning.
4. **Enough evidence.** At least `minimum_holdout_evidence` valid holdout scores; zero evidence never earns confidence.

**`medium`** means it passed the raw holdout bar but failed one of the stricter gates above — each failure is named in `warning_signs`. **`low`** means insufficient out-of-sample evidence to trust.

### Stable is not the same as good

Scores are normalised to 0..1, where 0 is *your* `bad_value` and 1 is *your* `good_value`. `minimum_acceptable_holdout_score` defaults to **0.0 — which is no performance bar at all**. Left there, a `high` rating asserts only that the region is *stable*, and a region can be rated confidently while scoring below the midpoint of your own metric range: stable and bad. NARRS warns loudly when this happens, but the warning is a backstop, not a substitute. **In production, always set `minimum_acceptable_holdout_score` to the normalised score you actually require** (0.5 is the midpoint between your `bad_value` and `good_value`). The permissive default exists for backward compatibility only.

## p_survive: a sharp fragility warning

Every `ConfidenceReport` also carries `p_survive` and `fragility_index`, from a logistic model fit against *realised* out-of-sample survival on the benchmark landscapes. The signal that separates survivors from failures turned out not to be score — NARRS only recommends regions it already trusts, so scores cluster far above any bar — but **noise**: failed regions carry several times the out-of-sample noise of survivors. The model consumes `effective_noise = max(holdout_noise, region_noise)`, `context_instability`, and the in-to-out `decay_gap`, and is refit against realised survival via `python tests/reliability.py --fit`.

**Read `p_survive` as a fragility warning, not a literal probability.** The installed coefficients are deliberately the *sharp* (class-balanced) fit: at failure-typical noise levels it reads ~0.2 rather than a flattering ~0.6, at the cost of under-rating easy survivors somewhat. That trade is intentional and recorded in `narrs/psurvive.py` — this number is the training gradient for the planned self-tuning layers and an early warning for humans, while the `high`/`medium`/`low` label (with its gates above) carries the calibrated protection. About 90% of recommended regions survive, so a fit optimised for average calibration is precisely one that stays quiet where failures live.

```python
from narrs import estimate_survival, estimate_from_region

est = estimate_from_region(result["best_region"])
est.p_survive          # sharp fragility signal, 0..1
est.fragility_index    # 0 robust .. 1 fragile
est.signals            # the raw inputs, for inspection
```

## Robustness policy: the context aggregator

A point is scored under several contexts, and those per-context results have to
be reduced to one robustness number. That reduction is a policy choice, and it's
pluggable rather than hard-coded.

```python
NARRSConfig(context_aggregator="cvar", context_aggregator_alpha=0.25)
```

| name | reduction | use when |
|---|---|---|
| `worst_case` *(default)* | `min` over contexts | few contexts, or any bad regime is disqualifying |
| `cvar` | mean of the worst `alpha` fraction | many contexts, or the worst one is probably noise |
| `mean_std` | `mean - alpha * std` | you want a symmetric dispersion penalty |
| `mean` | plain mean | no robustness preference (the naive control) |

You can also pass any object with `.aggregate(context_means, weights=None)` and
`.name`. **The default reproduces the original behaviour exactly** — `worst_case`
is `min`, which is what NARRS always did — so existing setups are unaffected.

One caveat worth knowing: CVaR at `alpha=0.25` over 4 contexts takes the worst 1,
so it *is* `min`. The aggregators only separate once the tail holds more than one
context.

## Benchmark battery

```bash
pip install -e ".[bench]"        # optuna + cma for the comparison baselines
python examples/run_battery.py
```

Two synthetic landscapes with a known right answer, because "robust" is a claim
that needs testing against something that punishes the alternative:

**Risk trap** — a broad steady plateau versus a region that pays more *on
average* but collapses in a minority of regimes. Nothing is statistically
unstable; the spike's high mean is entirely real and reproduces out of sample.
It is simply undeployable.

**Overfitting trap** — a modest but genuinely stationary edge among decoy peaks
whose height is pure per-context luck. Average a decoy over a few contexts and it
looks excellent; use fresh contexts and it evaporates.

Every method gets the same number of objective calls — NARRS runs first and the
baselines are given the budget it actually used. Results (seed 1):

| landscape | random | Optuna TPE | CMA-ES | NARRS |
|---|---|---|---|---|
| risk trap | spike | spike | spike | **plateau** |
| overfitting trap | decoy | decoy | decoy | **true edge** |

On the risk trap, out-of-sample tail (CVaR@25%) is **0.64 for NARRS against 0.22–0.24**
for the baselines — all three of which fail the survival bar. On the overfitting
trap the baselines decay **0.20–0.22** from in-sample to out-of-sample while NARRS
decays **0.002**.

PBO is reported per landscape with the evaluation criterion held fixed and only
the *selection rule* varying — scoring a tail-aware selector by out-of-sample
mean would mark it down for declining to maximise something it is deliberately
not maximising. On the risk trap, selecting by in-sample mean gives **PBO 0.81**
against **0.43** for tail-aware selection.

### An honest result about aggregators

`python examples/run_battery.py` also runs an aggregator study, and it does not
say what the "CVaR upgrade" framing would predict:

- **Tail-vs-mean is the decision that matters.** Any tail-aware rule beats `mean`
  by an enormous margin (PBO ~0.14–0.46 vs ~0.80), and the gap widens with more
  contexts.
- **CVaR does not beat plain worst-case when the tail is structural.** If some
  regimes genuinely break the strategy, the sharper rule wins, and `worst_case` —
  the existing default — is the sharpest. `cvar@0.50` gives most of the advantage
  back by averaging real crashes together with good contexts.
- **The ordering inverts when the tail is noise.** With no structurally bad
  regime, `worst_case` becomes the *weakest* rule, because one unlucky context
  dictates the entire score. That is where CVaR's tail-averaging pays.

So there is no globally best aggregator, and CVaR is not a free upgrade. The
choice encodes a belief about whether your bad contexts are signal or
measurement error — which is exactly why it's a pluggable seam with a
conservative default, rather than a replacement.

## Adversarial stress testing

The scaling stress suite varies *difficulty*. The adversarial landscapes in
`narrs/benchmarks/adversarial.py` do something different: each one **violates an
assumption the confidence machinery rests on**, and the bar to clear is not
"survive" — on several of them surviving is impossible. It is *refuse to claim
high confidence while failing*.

| landscape | assumption it attacks |
|---|---|
| `RegimeShift` | search and holdout are exchangeable — the optimum *moves* between them |
| `CorrelatedContexts` | contexts are independent evidence — N near-duplicate contexts carry ~2 contexts of information |
| `HeavyTailNoise` | noise is Gaussian and its sample sd is meaningful — calm until the jump |
| `DeceptiveMultiModal` | in-sample robustness implies out-of-sample robustness — many identical-looking plateaus, one survivor |

```bash
python tests/stress_suite.py --axis adversarial
```

These landscapes are what found the two most dangerous behaviours this package
has had — a `high` rating that meant "stable" while true out-of-sample
performance was negative (the missing performance bar, now warned about and
documented above), and regime-shifted regions rated `high` with saturated
p_survive (a scale bug in the decay signal, now fixed and gated). Both are
covered by the suite permanently: the pass condition is that the money-losing
cells show **zero** high ratings.

The one genuinely irreducible case: a regime shift that happens *after* your
holdout window ends. No backtest can see the future. Mitigate it structurally —
use `walk_forward_contexts` so the holdout *is* the future relative to search
(making the measured decay a real estimate of live decay), and monitor live
performance against that expectation.

## Overfitting metrics

Usable independently of the optimizer, on any returns or performance series:

```python
from narrs.metrics import pbo_cscv, deflated_sharpe, oos_decay

pbo_cscv(performance_matrix, n_splits=8)["pbo"]   # P(in-sample winner is below OOS median)
deflated_sharpe(returns, n_trials=200)["dsr"]     # Sharpe, deflated for selection
oos_decay(in_sample, out_of_sample)               # what evaporated
```

Pure standard library. `pbo_cscv` takes separate `is_metric` and `oos_metric` so
you can ask whether one *selection policy* generalises better than another on the
same landscape.

## Trading adapter (optional)

Core NARRS never imports this; the dependency runs one way only.

```python
from narrs.adapters.trading import CostModel, cost_stress_contexts, make_trading_objective

contexts = cost_stress_contexts(
    base_seeds=range(4),
    slippage_bps=(1.0, 3.0, 6.0),
    latency_seconds=(5.0, 7.0, 10.0),   # 7s default matches Kalshi
)
objective = make_trading_objective(my_strategy_returns, metric="cvar")
```

Contexts carry cost models and walk-forward windows through the existing
`SearchContext.data` hook, so the optimizer is unchanged. The point is that seeds
vary noise but not slippage, latency, or regime — a strategy that survives ten
random seeds has been tested against nothing a market will actually do to it.
`walk_forward_contexts` supports an `embargo` gap, because with autocorrelated
data an embargo of zero quietly inflates every result.

## Tests and verification

```bash
python tests/acceptance.py --original /path/to/old/package       # go/no-go verdict
python tests/test_additions.py                                   # 39 unit tests
python tests/verify_changes.py --original /path/to/old/package   # integration + A/B equivalence
python tests/stress_suite.py --quick                             # scaling sweep + gates
python tests/stress_suite.py --axis adversarial                  # assumption-violation suite
python tests/reliability.py                                      # p_survive calibration (add --fit to refit)
```

`acceptance.py` is the one to run before shipping — it ends in a single
SAFE TO PUSH / DO NOT PUSH line and calls the other suites as sub-steps.

`stress_suite.py` is the exploratory one: it answers *where does this start to
fail*, sweeping dimensionality (2→8), context count (2→24), noise, compute
budget and trap difficulty, with baselines at matched budget in every cell —
plus the adversarial axis above. It overlaps `acceptance.py` on calibration and
dominance by design: acceptance gives a verdict on the shipping configuration,
the stress suite maps the envelope around it.

### The number that actually matters

Not how often NARRS wins, but **how often it is confidently wrong**. A run that
reports `medium` and then fails is the system working: it told you not to trust
it. A run that reports `high` and fails is the only outcome that costs money.

Current acceptance run (20 seeds × 2 landscapes):

| stated rating | survived out of sample |
|---|---|
| `high` | 32/32 — **100%** |
| `medium` | 4/8 — 50% |

Confident failures: **0 of 40 runs**. Across the full 46-cell stress sweep (276
runs), every region rated `high` survived out of sample, while raw survival was
87% — the gap absorbed by honest self-labelling. On the adversarial axis, the
cells where survival is impossible show zero high ratings. The rating is
informative, and that is the actual product claim.

Stress-testing by starving it of evidence (`n_search_contexts` from 6 down to
2), confident failures stay at worst 1 of 20 across the whole range. It degrades
by losing confidence, not by getting confidently wrong.

### What verification caught

Worth reading before trusting any of the numbers above — every one of these was
found by the test layers, fixed, and pinned by a regression test or a permanent
suite cell:

- **A "high" rating could mean stable-but-losing.** With no performance bar set
  (the default), correlated-context landscapes earned `high` with negative true
  out-of-sample performance. Now warned about explicitly; set the bar.
- **Holdout noise can lie at few repetitions.** Loud objectives read quiet by
  luck and rated `high` at 0–50% actual survival. Fixed by the second,
  in-sample noise gate.
- **The decay signal was on the wrong scale.** `region_score` (an unbounded
  composite) was compared against the normalised `holdout_mean`, making the
  regime-shift signal meaningless — it had even fit with a backwards-signed
  coefficient. Fixed via `region_in_sample_mean`; regime shifts now cap the
  rating with an explicit warning.
- **A crashing user constraint looked like a failed search.** A
  `parameter_constraints` function that raised was silently treated as "point
  rejected", so a typo rejected every candidate and surfaced only as "no robust
  region found". Now warns once per distinct error.
- **The aggregator does not change the recommendation under default weights.**
  It is live, but `context_instability` and `point_noise` respond to the same bad
  contexts and dominate the score.
- **Single-seed results overstated NARRS** (6/8 on the overfitting trap over 8
  seeds, not 8/8). Every baseline survives 0/8 on both landscapes.
- **Write objectives with a stable digest, not `hash()`.** Python randomises
  string hashing per process, so a `hash()`-seeded objective silently differs
  across parallel workers and across runs. Use `zlib.crc32` on a formatted key —
  the shipped examples and test objectives all do.

## Configuration

`NARRSConfig` has many knobs; the ones you'll actually touch:

| field | default | what it does |
|---|---|---|
| `search_rounds` | 3 | rounds of search-and-refine |
| `initial_candidate_count` | 64 | candidate points sampled per round (non-grid samplers) |
| `repetitions_per_point` | 3 | noisy re-evaluations per point per context |
| `total_compute_budget` | 10000 | cap on objective calls during **search** |
| `n_jobs` | 1 | parallel workers; `-1` = all cores (see Performance) |
| `random_seed` | 42 | reproducibility |
| `minimum_region_size` | 3 | min points for a cluster to count as a region |
| `minimum_region_width` | 0.05 | min normalized width (rejects razor spikes) |
| `minimum_acceptable_holdout_score` | 0.0 | **the performance bar — set this** (0.0 = no bar; see "Stable is not the same as good") |
| `minimum_holdout_evidence` | 5 | valid holdout scores required for medium/high |
| `maximum_holdout_noise` | 1.0 | holdout-noise ceiling |
| `high_confidence_max_in_sample_noise` | 0.08 | in-sample noise ceiling for `high` (guards against lucky-quiet holdout readings) |
| `maximum_in_to_out_decay` | 0.15 | max in-sample → holdout drop before `high` is refused as a regime shift |
| `context_aggregator` | `worst_case` | robustness reduction policy (see Aggregators) |
| `verbose` | False | per-round logging |

## Performance & reproducibility

- **Speed**: with `scipy` installed, neighbor search uses a KD-tree (`O(n log n)`), which is dramatically faster than the brute-force fallback on large candidate sets while producing identical clusters. Install with `[fast]`.
- **Parallelism**: set `NARRSConfig(n_jobs=-1)` to evaluate candidates across all CPU cores. Results are identical to serial. Your `objective_function` must be picklable (a module-level function, not a lambda/closure); if it isn't, NARRS warns and falls back to serial. Parallelism pays off when each objective call is expensive (real backtests) — for trivial objectives the process overhead makes it slower, which is why the default is `1`.
- **macOS and Windows need a `__main__` guard for `n_jobs > 1`.** Those platforms start workers with `spawn`, not `fork`, and every worker re-imports your top-level script. Without the guard your script re-executes itself in each worker and the run dies with a `RuntimeError` about bootstrapping. Linux uses `fork` and is unaffected, so this only shows up when you move a working script to a Mac:

  ```python
  def main():
      opt = NoiseAwareRobustRegionSearch(..., config=NARRSConfig(n_jobs=-1))
      return opt.run()

  if __name__ == "__main__":      # required on macOS/Windows
      main()
  ```

  The same applies to the module holding your objective: it must be importable by name in a fresh interpreter.
- **Reproducibility**: runs are deterministic from `random_seed`.
- **Honest accounting**: `total_compute_budget` gates search only; the report separately shows `holdout_evaluations` so `total_objective_calls` reflects what you actually paid.

## Validation

Setups are checked at construction. A misconfiguration that would otherwise run silently and produce a misleading result — a typo'd metric-weight key that drops an objective, an inverted or degenerate parameter range, a negative weight, a duplicate parameter name, empty required inputs — raises a clear `NARRSConfigurationError` immediately. Softer issues (no holdout provided, a metric with a rule but no weight and no hard rule) emit warnings. A `parameter_constraints` function that raises is treated as rejecting the point *and* warned about, so a broken filter can't masquerade as an empty search. To skip validation, pass `validate=False` to the constructor.

## Limitations

- **Holdout ≠ purged/embargoed cross-validation.** NARRS validates on held-out contexts, but it can't detect leakage *between* them. On autocorrelated time series, a "high" rating is only as trustworthy as the independence of the contexts you feed it. Separate your holdout periods properly.
- **A regime shift after your holdout window is invisible to any backtest.** NARRS detects and refuses shifts *between* search and holdout, but nothing can validate against a future that hasn't happened. Use `walk_forward_contexts` so measured decay estimates live decay, and monitor deployed performance against it.
- **`p_survive` is a fragility warning, not a calibrated probability.** The sharp fit is installed deliberately (see the note in `narrs/psurvive.py`); the `high`/`medium`/`low` label is the calibrated statement.
- **The confidence haircut is a composite-score correction, not a Sharpe-ratio p-value.** It's the right shape of multiple-testing adjustment for this estimator. For a returns series, use `narrs.metrics.deflated_sharpe` and `narrs.metrics.pbo_cscv` directly instead.
- **The benchmark landscapes are synthetic.** They isolate specific failure modes with a known right answer, which is what makes them useful as tests. They are not evidence about any particular real market.
- **Roadmap**: adaptive evaluation allocation (spend repetitions where ranking is hardest) remains open. Config self-tuning and a cross-run meta-learning layer are the intended next steps — the battery already emits the labelled records both would train on, and the confidence signal they would tune against is now hardened.

## License

MIT — see [LICENSE](LICENSE).
