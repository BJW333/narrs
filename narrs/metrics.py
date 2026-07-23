"""
Overfitting metrics: PBO (via CSCV), Deflated and Probabilistic Sharpe, OOS decay.

These answer a different question from the search itself. The optimizer asks
"which region is best?"; these ask "how much should I believe the answer, given
how many things I tried?" That distinction is the whole point of the report
card, and it is what makes a result presentable to someone who is (rightly)
sceptical of backtests.

Pure standard library -- no numpy required.

The headline number is **PBO**, the Probability of Backtest Overfitting
(Bailey, Borwein, Lopez de Prado & Zhu). Take a performance matrix of
observations by candidate configurations, split the observations into ``S``
contiguous blocks, and for every balanced way of assigning half the blocks to
in-sample and half to out-of-sample: pick the in-sample winner, then see where
that winner ranks out of sample. If the winner is essentially a coin flip
against the OOS median, PBO approaches 0.5 and the selection procedure is
learning noise. PBO near 0 means in-sample ranking carries real information.

Note what is being measured: PBO is a property of the *selection procedure on a
landscape*, not of any single configuration. That is why ``pbo_cscv`` lets you
pass different in-sample and out-of-sample metrics -- selecting by a robust
statistic while scoring realised performance by mean is exactly the experiment
"does robust selection generalise better than naive selection?"
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist
from typing import Callable, List, Optional, Sequence

from .utils import expected_max_under_null

Matrix = Sequence[Sequence[float]]


# --------------------------------------------------------------------------- #
# small statistics helpers (kept local so this module stays import-light)      #
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float], ddof: int = 1) -> float:
    values = list(values)
    n = len(values)
    if n <= ddof:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - ddof))


def sharpe(values: Sequence[float], periods_per_year: Optional[float] = None) -> float:
    """Mean over standard deviation. Annualised only if you pass a period count."""
    values = list(values)
    if len(values) < 2:
        return 0.0
    sd = _std(values)
    if sd <= 0.0:
        return 0.0
    ratio = _mean(values) / sd
    if periods_per_year:
        ratio *= math.sqrt(periods_per_year)
    return ratio


def cvar(values: Sequence[float], alpha: float = 0.25) -> float:
    """Mean of the worst ``alpha`` fraction. The deployment-risk tail."""
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    k = max(1, int(math.ceil(alpha * len(values))))
    return sum(values[:k]) / k


def median(values: Sequence[float]) -> float:
    values = sorted(float(v) for v in values)
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


# --------------------------------------------------------------------------- #
# PBO via CSCV                                                                 #
# --------------------------------------------------------------------------- #
def pbo_cscv(
    performance: Matrix,
    n_splits: int = 8,
    metric: Callable[[Sequence[float]], float] = sharpe,
    is_metric: Optional[Callable[[Sequence[float]], float]] = None,
    oos_metric: Optional[Callable[[Sequence[float]], float]] = None,
) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    Parameters
    ----------
    performance:
        ``T x N`` matrix (rows = observations, columns = candidate
        configurations). Every column must be the same length.
    n_splits:
        Number of contiguous blocks ``S``; forced even and clamped to ``T``.
        All ``C(S, S/2)`` balanced in-sample/out-of-sample partitions are
        enumerated, so keep this modest -- ``S=8`` gives 70 partitions, ``S=16``
        gives 12870.
    metric:
        Default performance statistic, used for both sides unless overridden.
    is_metric / oos_metric:
        Selection statistic and evaluation statistic respectively. Pass
        different ones to compare selection *policies* on the same landscape.

    Returns
    -------
    dict with ``pbo``, ``logits``, ``n_partitions``, ``n_candidates``,
    ``mean_oos_rank`` (the winner's average OOS percentile rank, 1.0 = best).

    Raises
    ------
    ValueError if the matrix is empty, ragged, or has fewer than two candidates
    (PBO is meaningless when there is nothing to select between).
    """
    rows = [list(r) for r in performance]
    if not rows:
        raise ValueError("performance matrix is empty")
    n_candidates = len(rows[0])
    if n_candidates < 2:
        raise ValueError(
            "PBO needs at least 2 candidate configurations to measure selection"
        )
    if any(len(r) != n_candidates for r in rows):
        raise ValueError("performance matrix is ragged: all rows must be the same length")

    n_obs = len(rows)
    is_metric = is_metric or metric
    oos_metric = oos_metric or metric

    n_blocks = n_splits - (n_splits % 2)
    n_blocks = max(2, min(n_blocks, n_obs))
    if n_blocks < 2:
        raise ValueError("need at least 2 observations to split")

    edges = [round(i * n_obs / n_blocks) for i in range(n_blocks + 1)]
    blocks = [list(range(edges[i], edges[i + 1])) for i in range(n_blocks)]
    blocks = [b for b in blocks if b]
    n_blocks = len(blocks)
    half = n_blocks // 2
    if half < 1:
        raise ValueError("not enough observations to form balanced CSCV splits")

    logits: List[float] = []
    ranks: List[float] = []

    for is_blocks in combinations(range(n_blocks), half):
        chosen = set(is_blocks)
        is_rows = [i for b in range(n_blocks) if b in chosen for i in blocks[b]]
        oos_rows = [i for b in range(n_blocks) if b not in chosen for i in blocks[b]]
        if not is_rows or not oos_rows:
            continue

        is_perf = [
            is_metric([rows[i][j] for i in is_rows]) for j in range(n_candidates)
        ]
        oos_perf = [
            oos_metric([rows[i][j] for i in oos_rows]) for j in range(n_candidates)
        ]

        winner = max(range(n_candidates), key=lambda j: is_perf[j])
        # Relative rank of the in-sample winner within the OOS distribution.
        # Ties count as half, so a degenerate all-equal landscape gives 0.5.
        better = sum(1 for v in oos_perf if v < oos_perf[winner])
        equal = sum(1 for v in oos_perf if v == oos_perf[winner])
        rank = (better + 0.5 * equal) / n_candidates
        rank = min(max(rank, 1e-9), 1 - 1e-9)
        ranks.append(rank)
        logits.append(math.log(rank / (1.0 - rank)))

    if not logits:
        raise ValueError("no valid CSCV partitions were produced")

    # PBO = P(the in-sample winner lands at or below the out-of-sample median).
    # A logit of exactly zero means the winner sits *on* the median, which is
    # indifference, not evidence of overfitting -- counting it as a full hit
    # would report PBO 1.0 for a landscape where every candidate is identical.
    # Ties therefore count as half, matching the tie handling in the rank above.
    below = sum(1 for x in logits if x < 0.0)
    tied = sum(1 for x in logits if x == 0.0)
    pbo = (below + 0.5 * tied) / len(logits)
    return {
        "pbo": pbo,
        "logits": logits,
        "n_partitions": len(logits),
        "n_candidates": n_candidates,
        "mean_oos_rank": _mean(ranks),
    }


# --------------------------------------------------------------------------- #
# Sharpe-family corrections                                                    #
# --------------------------------------------------------------------------- #
def probabilistic_sharpe(
    observed_sharpe: float,
    n_obs: int,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true Sharpe > benchmark), correcting for sample length and non-normality.

    Short samples and fat left tails both make a given Sharpe less believable;
    this folds all three into one probability.
    """
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * observed_sharpe + 0.25 * (kurtosis - 1.0) * observed_sharpe ** 2
    if denom <= 0:
        return 0.0
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return NormalDist().cdf(z)


def deflated_sharpe(
    returns: Sequence[float],
    n_trials: int,
    benchmark_sharpe: Optional[float] = None,
) -> dict:
    """Deflated Sharpe Ratio: PSR against a selection-adjusted benchmark.

    Trying ``n_trials`` configurations and keeping the best inflates the winner's
    Sharpe even with no real edge. The benchmark is raised to the level a
    best-of-N fluke would reach, and the observed Sharpe must clear *that*.

    Returns ``sharpe``, ``benchmark``, ``dsr``, ``n_obs``, ``n_trials``.
    ``dsr`` is a probability: above ~0.95 is the usual "survives selection" bar.
    """
    values = [float(v) for v in returns]
    n = len(values)
    observed = sharpe(values)

    if benchmark_sharpe is None:
        sd = _std(values)
        if n > 1 and sd > 0:
            # Variance of the Sharpe estimator across trials, approximated by the
            # sampling variance of a single Sharpe under the null.
            var_sr = (1.0 + 0.5 * observed ** 2) / max(n - 1, 1)
            benchmark_sharpe = expected_max_under_null(n_trials) * math.sqrt(max(var_sr, 0.0))
        else:
            benchmark_sharpe = 0.0

    skew, kurt = _skew_kurtosis(values)
    dsr = probabilistic_sharpe(observed, n, benchmark_sharpe, skew, kurt)
    return {
        "sharpe": observed,
        "benchmark": benchmark_sharpe,
        "dsr": dsr,
        "n_obs": n,
        "n_trials": n_trials,
    }


def _skew_kurtosis(values: Sequence[float]) -> tuple[float, float]:
    values = list(values)
    n = len(values)
    if n < 3:
        return 0.0, 3.0
    m = _mean(values)
    sd = _std(values, ddof=0)
    if sd <= 0:
        return 0.0, 3.0
    skew = sum((v - m) ** 3 for v in values) / (n * sd ** 3)
    kurt = sum((v - m) ** 4 for v in values) / (n * sd ** 4)
    return skew, kurt


def oos_decay(in_sample: Sequence[float], out_of_sample: Sequence[float]) -> dict:
    """How much performance evaporated between selection and validation.

    ``decay`` is the absolute drop in mean; ``retention`` is the fraction kept
    (guarded against a non-positive in-sample mean, where a ratio is meaningless
    and is reported as ``None``).
    """
    is_mean = _mean(in_sample)
    oos_mean = _mean(out_of_sample)
    retention = (oos_mean / is_mean) if is_mean > 0 else None
    return {
        "in_sample_mean": is_mean,
        "out_of_sample_mean": oos_mean,
        "decay": is_mean - oos_mean,
        "retention": retention,
    }


__all__ = [
    "pbo_cscv",
    "deflated_sharpe",
    "probabilistic_sharpe",
    "oos_decay",
    "sharpe",
    "cvar",
    "median",
]
