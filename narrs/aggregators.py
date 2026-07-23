"""
Pluggable context aggregation.

A point is evaluated under several :class:`SearchContext` objects (time windows,
regimes, seeds, cost assumptions). Those per-context means have to be reduced to
a single robustness number. That reduction is a *policy choice*, and this module
makes it swappable instead of hard-coded.

Historically NARRS reduced with ``min`` (pure worst-case). That is still the
default and remains exactly what you get if you change nothing. The reason to
make it pluggable is that ``min`` has a known failure mode: it is a
single-order-statistic estimator, so with many contexts it is both
over-conservative and *noisy* -- one unlucky context dictates the entire score,
and the score jumps around when you resample contexts. Conditional Value at Risk
(CVaR) at level ``alpha`` -- the mean of the worst ``alpha`` fraction -- keeps
the tail focus but averages over the tail, so it is stable under resampling and
tunable between "average case" and "worst case".

This is the distributionally-robust view: instead of optimizing expected
performance under the empirical context distribution, optimize expected
performance under the worst ``alpha``-fraction of it. ``min`` and ``mean`` are
the two endpoints of that family; CVaR is the dial between them.

Pure standard library.

Choosing one
------------
    worst_case              you have few contexts, or a single bad regime is
                            genuinely unacceptable (default; back-compatible)
    cvar (alpha=0.25)       you have many contexts (cost/latency stress grids,
                            walk-forward windows) and want a stable tail measure
    mean_std (k=1.0)        you want a symmetric dispersion penalty rather than
                            a tail measure
    mean                    no robustness preference -- included mainly as the
                            naive control in the benchmark battery

Note on few contexts: CVaR at alpha=0.25 over 4 contexts takes the worst 1, so
it *is* ``min``. The aggregators only separate once you have enough contexts for
the tail to contain more than one observation. Do not expect a difference at
n_contexts <= 4.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ContextAggregator(Protocol):
    """Reduce per-context mean scores to one robustness number.

    Implement this to plug in your own policy. Higher must always mean better,
    because everything downstream (region scoring, rejection thresholds) assumes
    a higher-is-better scale.
    """

    @property
    def name(self) -> str:  # pragma: no cover - protocol declaration
        ...

    def aggregate(
        self,
        context_means: Sequence[float],
        weights: Optional[Sequence[float]] = None,
    ) -> float:  # pragma: no cover - protocol declaration
        ...


def _clean(
    context_means: Sequence[float],
    weights: Optional[Sequence[float]],
) -> tuple[List[float], List[float]]:
    values = [float(v) for v in context_means]
    if weights is None:
        w = [1.0] * len(values)
    else:
        w = [float(x) for x in weights]
        if len(w) != len(values):
            raise ValueError(
                f"weights length {len(w)} does not match context_means length {len(values)}"
            )
        if any(x < 0 for x in w):
            raise ValueError("weights must be non-negative")
    return values, w


class WorstCaseAggregator:
    """Minimum over contexts. The historical NARRS behaviour and the default.

    Maximally conservative: performance is judged entirely by the single worst
    context. Appropriate when any bad regime is disqualifying, and when you have
    few enough contexts that the minimum is not dominated by noise.
    """

    name = "worst_case"

    def aggregate(
        self,
        context_means: Sequence[float],
        weights: Optional[Sequence[float]] = None,
    ) -> float:
        values, _ = _clean(context_means, weights)
        if not values:
            return 0.0
        return min(values)


class CVaRAggregator:
    """Mean of the worst ``alpha`` fraction of contexts (a.k.a. expected shortfall).

    ``alpha=1.0`` recovers the mean; ``alpha -> 0`` recovers the worst case. The
    tail always contains at least one context, so this never degenerates to
    something less conservative than ``min`` on tiny context sets -- it equals it.

    Weighted case: contexts are sorted worst-first and weight mass is consumed
    until ``alpha`` of the total is used, with the boundary context partially
    counted (the Rockafellar-Uryasev construction). This keeps the measure
    continuous in alpha rather than jumping as contexts cross the cutoff.
    """

    def __init__(self, alpha: float = 0.25) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)

    @property
    def name(self) -> str:
        return f"cvar_{self.alpha:g}"

    def aggregate(
        self,
        context_means: Sequence[float],
        weights: Optional[Sequence[float]] = None,
    ) -> float:
        values, w = _clean(context_means, weights)
        if not values:
            return 0.0

        order = sorted(range(len(values)), key=lambda i: values[i])
        total_weight = sum(w)
        if total_weight <= 0:
            return min(values)

        target = self.alpha * total_weight
        used = 0.0
        acc = 0.0
        for i in order:
            take = min(w[i], target - used)
            if take <= 0:
                break
            acc += take * values[i]
            used += take
            if used >= target:
                break

        if used <= 0:
            # alpha so small that no mass was consumed: fall back to the worst.
            return values[order[0]]
        return acc / used


class MeanStdAggregator:
    """``mean - k * std`` over contexts. A symmetric dispersion penalty.

    Not a tail measure: it punishes upside variability as much as downside. Kept
    because it is the familiar baseline and is cheap to reason about.
    """

    def __init__(self, k: float = 1.0) -> None:
        self.k = float(k)

    @property
    def name(self) -> str:
        return f"mean_std_{self.k:g}"

    def aggregate(
        self,
        context_means: Sequence[float],
        weights: Optional[Sequence[float]] = None,
    ) -> float:
        values, w = _clean(context_means, weights)
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]

        total = sum(w)
        if total <= 0:
            return sum(values) / len(values)
        mean = sum(wi * v for wi, v in zip(w, values)) / total
        var = sum(wi * (v - mean) ** 2 for wi, v in zip(w, values)) / total
        return mean - self.k * math.sqrt(max(var, 0.0))


class MeanAggregator:
    """Plain (weighted) mean. No robustness preference at all.

    This is the naive control: it is what a standard optimizer implicitly uses.
    Selecting with it is what the benchmark battery is designed to punish.
    """

    name = "mean"

    def aggregate(
        self,
        context_means: Sequence[float],
        weights: Optional[Sequence[float]] = None,
    ) -> float:
        values, w = _clean(context_means, weights)
        if not values:
            return 0.0
        total = sum(w)
        if total <= 0:
            return sum(values) / len(values)
        return sum(wi * v for wi, v in zip(w, values)) / total


_BUILTIN = {
    "worst_case": WorstCaseAggregator,
    "worst": WorstCaseAggregator,
    "min": WorstCaseAggregator,
    "cvar": CVaRAggregator,
    "mean_std": MeanStdAggregator,
    "meanstd": MeanStdAggregator,
    "mean": MeanAggregator,
}


def resolve_aggregator(spec: Any, alpha: float = 0.25) -> ContextAggregator:
    """Turn a config value into an aggregator instance.

    Accepts an aggregator object (returned as-is), or a string name. ``alpha`` is
    used as the parameter for the parameterised aggregators (tail fraction for
    ``cvar``, ``k`` for ``mean_std``) so that :class:`NARRSConfig` can stay flat
    and serialisable -- which matters because that config is itself intended to
    be a search space later.
    """
    if spec is None:
        return WorstCaseAggregator()
    if not isinstance(spec, str):
        if hasattr(spec, "aggregate") and hasattr(spec, "name"):
            return spec
        raise TypeError(
            "context_aggregator must be a string name or an object with "
            f".aggregate() and .name; got {type(spec).__name__}"
        )

    key = spec.strip().lower()
    if key not in _BUILTIN:
        raise ValueError(
            f"unknown context_aggregator {spec!r}; "
            f"choose one of {sorted(set(_BUILTIN))} or pass an object"
        )
    cls = _BUILTIN[key]
    if cls is CVaRAggregator:
        return CVaRAggregator(alpha=alpha)
    if cls is MeanStdAggregator:
        return MeanStdAggregator(k=alpha)
    return cls()


__all__ = [
    "ContextAggregator",
    "WorstCaseAggregator",
    "CVaRAggregator",
    "MeanStdAggregator",
    "MeanAggregator",
    "resolve_aggregator",
]
