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
    config=NARRSConfig(search_rounds=3, total_compute_budget=2000, n_jobs=1, random_seed=42),
)

result = optimizer.run()
report = result["confidence_report"]

print("Recommended center:", result["recommended_center"])
print("Confidence rating :", report.final_confidence_rating)
print("Objective calls   :", report.total_objective_calls,
      f"(search {report.search_evaluations} + holdout {report.holdout_evaluations})")
```

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

The **`ConfidenceReport`** carries the recommendation plus everything you need to trust it: `final_confidence_rating`, `best_region_parameter_ranges`, `number_of_surviving_regions`, `warning_signs`, honest call accounting (`search_evaluations`, `holdout_evaluations`, `total_objective_calls`), and the selection-adjustment fields (`number_of_trials_considered`, `deflated_holdout_threshold`, `holdout_selection_penalty`).

## Confidence ratings

The rating is deliberately conservative — it's a statement about out-of-sample trust, not in-sample score:

- **`high`** — the region cleared out-of-sample holdout validation with enough evidence *and* beat a selection-adjusted (Deflated-Sharpe-style) bar that accounts for how much searching was done.
- **`medium`** — passed the raw holdout bar but not the stricter selection-adjusted one, or had no holdout available and was only strong in-sample.
- **`low`** — insufficient out-of-sample evidence to trust. A region with too few valid holdout scores stays here by design (zero evidence never earns confidence).

Two guards are built in:

- **Minimum holdout evidence** (`minimum_holdout_evidence`, default 5): a region needs at least this many valid holdout scores before it can be rated medium/high.
- **Selection (multiple-testing) haircut**: because searching many regions and keeping the best inflates its apparent score by chance, the "high" bar is raised by the expected best-of-N fluke. `holdout_selection_penalty` shows how much.

## Robustness policy: the context aggregator

A point is scored under several contexts, and those per-context results have to
be reduced to one robustness number. That reduction is a policy choice, and it's
now pluggable rather than hard-coded.

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
`.name`. **The default reproduces the previous behaviour exactly** — `worst_case`
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
python tests/test_additions.py                                   # 35 unit tests
python tests/verify_changes.py --original /path/to/old/package   # 14 integration checks
python tests/stress_suite.py --quick                             # scaling sweep + gates
```

`acceptance.py` is the one to run before shipping — it ends in a single
SAFE TO PUSH / DO NOT PUSH line and calls the other two as sub-steps.

`stress_suite.py` is the exploratory one: it answers *where does this start to
fail*, sweeping dimensionality (2→8), context count (2→24), noise, compute
budget and trap difficulty, with baselines at matched budget in every cell. It
overlaps `acceptance.py` on calibration and dominance by design — acceptance
gives a verdict on the shipping configuration, the stress suite maps the
envelope around it.

### The number that actually matters

Not how often NARRS wins, but **how often it is confidently wrong**. A run that
reports `medium` and then fails is the system working: it told you not to trust
it. A run that reports `high` and fails is the only outcome that costs money.

Measured over 24 runs (12 seeds x 2 landscapes):

| stated rating | survived out of sample |
|---|---|
| `high` | 21/22 — **95%** |
| `medium` | 0/2 — **0%** |

The rating is informative, and that is the actual product claim. NARRS lands on
the true edge 9/12 on the overfitting trap while every baseline manages 0–1/12 —
but more importantly, the runs where it is fooled are the runs it declines to
call `high`.

Stress-testing that by starving it of evidence (`n_search_contexts` from 6 down
to 2), confident failures stay at 1–3 of 12 across the whole range. It degrades
by losing confidence, not by getting confidently wrong.

`test_additions.py` checks each new function in isolation. `verify_changes.py`
tries to falsify the integration claims:

1. **Equivalence** — runs an identical setup against an untouched copy of the
   package and compares 19 reported fields at 12 decimal places. Currently
   byte-identical, which is the actual guarantee that nothing existing changed.
2. **Is the aggregator live?** — with the other dispersion penalties zeroed,
   `worst_case` avoids a rare-catastrophe region and `mean` walks into it. The
   seam demonstrably decides the answer.
3. **PBO correctness** — identical candidates give 0.50, pure noise ~0.50, a
   genuinely dominant candidate ~0.00, and a landscape where every candidate
   reverses out of sample gives 0.74.
4. **Seed robustness** — the benchmark result over 8 seeds rather than one.
5. **Parallelism** — `n_jobs=1` and `n_jobs=-1` produce identical output.

### What verification caught

Worth reading before trusting any of the numbers above:

- **The aggregator does not change the recommendation under default weights.**
  It is live, but `context_instability` and `point_noise` respond to the same bad
  contexts and dominate the score. Setting `context_aggregator="cvar"` expecting
  less conservatism will not do much unless you also relax those weights.
- **Single-seed results overstated NARRS.** Over 8 seeds it survives 8/8 on the
  risk trap but **6/8** on the overfitting trap — it is fooled sometimes. Every
  baseline survives 0/8 on both.
- **Two bugs, both fixed and pinned by regression tests**: the CMA-ES baseline
  crashed when a generation was truncated to fit the remaining budget, and PBO
  returned 1.00 instead of 0.50 for a landscape of identical candidates because
  exact ties were counted as certain overfitting.
- **Write objectives with a stable digest, not `hash()`.** Python randomises
  string hashing per process, so a `hash()`-seeded objective silently differs
  across parallel workers and across runs. The verification harness caught this
  in its own test code.

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
| `minimum_acceptable_holdout_score` | 0.0 | floor a region's holdout mean must clear |
| `minimum_holdout_evidence` | 5 | valid holdout scores required for medium/high |
| `maximum_holdout_noise` | 1.0 | holdout-noise ceiling |
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

Setups are checked at construction. A misconfiguration that would otherwise run silently and produce a misleading result — a typo'd metric-weight key that drops an objective, an inverted or degenerate parameter range, a negative weight, a duplicate parameter name, empty required inputs — raises a clear `NARRSConfigurationError` immediately. Softer issues (no holdout provided, a metric with a rule but no weight and no hard rule) emit warnings. To skip validation, pass `validate=False` to the constructor.

## Limitations

- **Holdout ≠ purged/embargoed cross-validation.** NARRS validates on held-out contexts, but it can't detect leakage *between* them. On autocorrelated time series, a "high" rating is only as trustworthy as the independence of the contexts you feed it. Separate your holdout periods properly.
- **The confidence haircut is a composite-score correction, not a Sharpe-ratio p-value.** It's the right shape of multiple-testing adjustment for this estimator. For a returns series, use `narrs.metrics.deflated_sharpe` and `narrs.metrics.pbo_cscv` directly instead.
- **The benchmark landscapes are synthetic.** They isolate two specific failure modes with a known right answer, which is what makes them useful as a test. They are not evidence about any particular real market.
- **Roadmap**: adaptive evaluation allocation (spend repetitions where ranking is hardest) remains open. Config self-tuning and a cross-run meta-learning layer are the intended next steps — the battery already emits the labelled records both would train on.

## License

MIT — see [LICENSE](LICENSE).
