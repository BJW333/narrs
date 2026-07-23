"""
Trading adapter -- an optional plugin. Core NARRS never imports this.

Everything market-specific lives here and rides on the one seam the core already
exposes: ``SearchContext.data`` is opaque to the optimizer, so a context can
carry a cost model, a walk-forward window, or anything else, and the search
machinery is unchanged. Nothing in this module patches or subclasses the
optimizer; it only builds contexts and objectives that the optimizer already
knows how to consume.

That matters for a specific reason. The value of scoring across contexts comes
entirely from the contexts being *adversarial in the way reality is*. Seeds vary
noise; they do not vary slippage, latency, or regime. A strategy that survives
ten random seeds has been tested against nothing a market will actually do to
it. These builders make contexts that vary the things that kill live strategies.

Pure standard library.

    from narrs.adapters.trading import (
        CostModel, cost_stress_contexts, walk_forward_contexts,
        returns_to_score, make_trading_objective,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..types import ObjectiveResult, ParamDict, SearchContext


# --------------------------------------------------------------------------- #
# Cost / microstructure                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostModel:
    """Per-trade frictions plus latency.

    ``latency_seconds`` defaults to 7.0 to match the Kalshi assumption; change it
    per venue. Latency only bites if you also give the objective an
    ``edge_decay_per_second``, because latency costs you nothing unless your edge
    is decaying while you wait.
    """

    slippage_bps: float = 2.0
    fee_bps: float = 1.0
    latency_seconds: float = 7.0
    fill_probability: float = 1.0

    def apply(
        self,
        gross_return: float,
        trades: float = 1.0,
        edge_decay_per_second: float = 0.0,
    ) -> float:
        """Net a gross per-period return for costs, latency decay and fill risk."""
        cost = (self.slippage_bps + self.fee_bps) * 1e-4 * trades
        decayed = gross_return * math.exp(-edge_decay_per_second * self.latency_seconds)
        return self.fill_probability * (decayed - cost)

    @property
    def label(self) -> str:
        return f"slip{self.slippage_bps:g}bp/fee{self.fee_bps:g}bp/lat{self.latency_seconds:g}s"


def cost_stress_contexts(
    base_seeds: Sequence[int] = (0, 1, 2, 3),
    slippage_bps: Sequence[float] = (1.0, 3.0, 6.0),
    latency_seconds: Sequence[float] = (5.0, 7.0, 10.0),
    fee_bps: float = 1.0,
    fill_probability: float = 1.0,
    prefix: str = "cost",
) -> List[SearchContext]:
    """Cross seeds with a grid of cost regimes: one context per combination.

    A parameter set that still clears the bar under the worst corner of this grid
    is robust to the microstructure it will actually trade in. Note the count
    multiplies fast -- 4 seeds x 3 slippages x 3 latencies is 36 contexts, and
    every context multiplies your objective calls.
    """
    contexts: List[SearchContext] = []
    for seed in base_seeds:
        for slip in slippage_bps:
            for latency in latency_seconds:
                model = CostModel(
                    slippage_bps=slip,
                    fee_bps=fee_bps,
                    latency_seconds=latency,
                    fill_probability=fill_probability,
                )
                contexts.append(
                    SearchContext(
                        name=f"{prefix}_s{seed}_{model.label}",
                        data={"seed": int(seed), "cost": model},
                    )
                )
    return contexts


# --------------------------------------------------------------------------- #
# Walk-forward / regime windows                                                #
# --------------------------------------------------------------------------- #
def walk_forward_contexts(
    n_periods: int,
    n_windows: int = 6,
    test_fraction: float = 0.30,
    embargo: int = 0,
    prefix: str = "wf",
) -> List[SearchContext]:
    """Rolling train/test windows carried as contexts.

    Scoring across these and aggregating with a tail-aware rule measures
    stability *through* regime transitions rather than within one sample. The
    ``embargo`` gap between train and test exists to stop overlapping
    observations leaking across the boundary -- with autocorrelated data, an
    embargo of zero quietly inflates every result.
    """
    if n_periods <= 0 or n_windows <= 0:
        return []

    contexts: List[SearchContext] = []
    step = max(1, n_periods // (n_windows + 1))
    for i in range(n_windows):
        train_start = i * step
        train_end = min(train_start + step, n_periods)
        test_start = min(train_end + embargo, n_periods)
        test_end = min(test_start + max(1, int(step * test_fraction)), n_periods)
        if test_start >= test_end or train_start >= train_end:
            continue
        contexts.append(
            SearchContext(
                name=f"{prefix}_{i:02d}",
                data={
                    "train": (train_start, train_end),
                    "test": (test_start, test_end),
                    "embargo": embargo,
                },
            )
        )
    return contexts


# --------------------------------------------------------------------------- #
# returns -> score                                                             #
# --------------------------------------------------------------------------- #
def returns_to_score(
    returns: Sequence[float],
    metric: str = "sharpe",
    alpha: float = 0.25,
    periods_per_year: float = 252.0,
) -> float:
    """Reduce a returns series to the scalar an objective reports.

    ``sharpe`` annualises; ``sortino`` penalises only downside deviation;
    ``cvar`` is the mean of the worst ``alpha`` fraction of periods and pairs
    naturally with a tail-aware context aggregator -- giving you robustness at
    two levels, within a window and across contexts. ``total`` is the plain sum.
    """
    values = [float(v) for v in returns]
    n = len(values)
    if n == 0:
        return 0.0

    mean = sum(values) / n

    if metric == "total":
        return sum(values)

    if metric == "cvar":
        ordered = sorted(values)
        k = max(1, int(math.ceil(alpha * n)))
        return sum(ordered[:k]) / k

    if n < 2:
        return 0.0

    if metric == "sortino":
        downside = [v for v in values if v < 0]
        if len(downside) < 2:
            return 0.0
        dmean = sum(downside) / len(downside)
        dstd = math.sqrt(sum((v - dmean) ** 2 for v in downside) / (len(downside) - 1))
        if dstd <= 0:
            return 0.0
        return mean / dstd * math.sqrt(periods_per_year)

    if metric != "sharpe":
        raise ValueError(
            f"unknown metric {metric!r}; expected one of "
            "'sharpe', 'sortino', 'cvar', 'total'"
        )

    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    if std <= 0:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


def make_trading_objective(
    strategy_returns: Callable[[ParamDict, int], Sequence[float]],
    metric: str = "sharpe",
    alpha: float = 0.25,
    edge_decay_per_second: float = 0.0,
    trades_per_period: float = 1.0,
    metric_name: str = "score",
    periods_per_year: float = 252.0,
) -> Callable[[ParamDict, SearchContext, int], ObjectiveResult]:
    """Wrap a returns-generating strategy into a NARRS objective.

    ``strategy_returns(params, seed)`` returns gross per-period returns. The
    resulting objective reads any :class:`CostModel` on the context, nets every
    period for costs and latency decay, reduces to ``metric``, and reports it
    with ``sample_count`` set to the number of periods -- so NARRS's own
    minimum-evidence checks see how much data actually backs each score.
    """

    def objective(params: ParamDict, context: SearchContext, rep: int) -> ObjectiveResult:
        data = context.data if isinstance(context.data, dict) else {}
        seed = int(data.get("seed", 0)) * 1000 + int(rep)
        gross = list(strategy_returns(params, seed))

        cost_model: Optional[CostModel] = data.get("cost")
        if cost_model is not None:
            net = [
                cost_model.apply(
                    g,
                    trades=trades_per_period,
                    edge_decay_per_second=edge_decay_per_second,
                )
                for g in gross
            ]
        else:
            net = gross

        score = returns_to_score(
            net, metric=metric, alpha=alpha, periods_per_year=periods_per_year
        )
        return ObjectiveResult(
            performance_metrics={metric_name: score},
            sample_count=max(1, len(net)),
            metadata={"context": context.name},
        )

    return objective


__all__ = [
    "CostModel",
    "cost_stress_contexts",
    "walk_forward_contexts",
    "returns_to_score",
    "make_trading_objective",
]
