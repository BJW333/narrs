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
- **Reproducibility**: runs are deterministic from `random_seed`.
- **Honest accounting**: `total_compute_budget` gates search only; the report separately shows `holdout_evaluations` so `total_objective_calls` reflects what you actually paid.

## Validation

Setups are checked at construction. A misconfiguration that would otherwise run silently and produce a misleading result — a typo'd metric-weight key that drops an objective, an inverted or degenerate parameter range, a negative weight, a duplicate parameter name, empty required inputs — raises a clear `NARRSConfigurationError` immediately. Softer issues (no holdout provided, a metric with a rule but no weight and no hard rule) emit warnings. To skip validation, pass `validate=False` to the constructor.

## Limitations

- **Holdout ≠ purged/embargoed cross-validation.** NARRS validates on held-out contexts, but it can't detect leakage *between* them. On autocorrelated time series, a "high" rating is only as trustworthy as the independence of the contexts you feed it. Separate your holdout periods properly.
- **The confidence haircut is a composite-score correction, not a Sharpe-ratio p-value.** It's the right shape of multiple-testing adjustment for this estimator, but it doesn't replace a full Deflated-Sharpe / Probability-of-Backtest-Overfitting analysis on a returns series.
- **Roadmap**: adaptive evaluation allocation (spend repetitions where ranking is hardest) and a Probability-of-Backtest-Overfitting (CSCV) estimate are planned to further cut evaluation cost and strengthen the overfitting diagnostics.

## License

MIT — see [LICENSE](LICENSE).
