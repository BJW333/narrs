"""
Dimension- and context-scalable versions of the benchmark landscapes.

``problems.py`` holds the fixed 2-D landscapes used by the report card. Those
are deliberately frozen: their numbers appear in the README and in the
verification harness, so changing them would invalidate results you have already
checked. This module provides parameterised versions for stress testing instead,
so the sweep can vary dimensionality, context count, noise and difficulty
without touching anything already verified.

Distances are normalised by ``sqrt(dim)`` so that a "width" means roughly the
same fraction of the space at every dimensionality. Without that, a fixed-width
Gaussian bump becomes an undetectable needle by 8 dimensions and the sweep would
be measuring the curse of dimensionality rather than anything about NARRS.

Pure standard library.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..types import (
    MetricRule,
    ObjectiveResult,
    ParamDict,
    ParameterSpec,
    SearchContext,
)


def _stable_hash(*parts) -> int:
    """Deterministic across processes -- never ``hash()`` on a string here.

    Python randomises string hashing per process, so a hash-seeded objective
    silently differs between parallel workers and between runs.
    """
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


def _rng(*parts) -> random.Random:
    return random.Random(_stable_hash(*parts))


def _key(params: ParamDict) -> str:
    return ",".join(f"{k}={params[k]:.10g}" for k in sorted(params))


def _ndist(params: ParamDict, center: Dict[str, float]) -> float:
    """Euclidean distance normalised by sqrt(dim), so widths are dimension-free."""
    if not center:
        return 0.0
    total = sum((params[k] - center[k]) ** 2 for k in center)
    return math.sqrt(total / len(center))


def _names(dim: int) -> List[str]:
    return [f"x{i}" for i in range(dim)]


@dataclass
class ScalablePlateauSpike:
    """Risk trap at arbitrary dimension and context count.

    A broad flat plateau with a steady edge, versus a region with a higher mean
    that collapses in a fixed fraction of contexts.
    """

    dim: int = 2
    n_search_contexts: int = 10
    n_holdout_contexts: int = 10
    noise_sd: float = 0.05
    crash_fraction: float = 0.20
    plateau_value: float = 0.30
    spike_value: float = 0.75
    plateau_height: float = 0.85
    plateau_flat_radius: float = 0.16
    plateau_falloff: float = 0.22
    spike_width: float = 0.22
    spike_payoff: float = 3.0
    spike_crash: float = -3.0
    seed: int = 11

    def __post_init__(self) -> None:
        names = _names(self.dim)
        self._pc = {n: self.plateau_value for n in names}
        self._sc = {n: self.spike_value for n in names}

    @property
    def parameter_space(self) -> List[ParameterSpec]:
        return [ParameterSpec(n, 0.0, 1.0) for n in _names(self.dim)]

    @property
    def metric_rules(self) -> Dict[str, MetricRule]:
        return {"score": MetricRule(good_value=3.0, bad_value=-3.0, higher_is_better=True)}

    @property
    def metric_weights(self) -> Dict[str, float]:
        return {"score": 1.0}

    def _contexts(self, prefix: str, n: int) -> List[SearchContext]:
        n_crash = max(1, int(round(self.crash_fraction * n))) if self.crash_fraction > 0 else 0
        rng = random.Random(_stable_hash(self.seed, prefix, "regimes", n))
        crash = set(rng.sample(range(n), n_crash)) if n_crash else set()
        return [SearchContext(name=f"{prefix}_{i:02d}", data={"crash": i in crash})
                for i in range(n)]

    @property
    def search_contexts(self) -> List[SearchContext]:
        return self._contexts("regime", self.n_search_contexts)

    @property
    def holdout_contexts(self) -> List[SearchContext]:
        return self._contexts("oos", self.n_holdout_contexts)

    @property
    def survival_threshold(self) -> float:
        return self.metric_rules["score"].normalize(0.0)

    def raw_score(self, params: ParamDict, context: SearchContext, rep: int) -> float:
        d = _ndist(params, self._pc)
        if d <= self.plateau_flat_radius:
            plateau = self.plateau_height
        else:
            over = d - self.plateau_flat_radius
            plateau = self.plateau_height * math.exp(-0.5 * (over / self.plateau_falloff) ** 2)

        ds = _ndist(params, self._sc)
        prox = math.exp(-0.5 * (ds / self.spike_width) ** 2)
        crashed = bool((context.data or {}).get("crash", False))
        spike = (self.spike_crash if crashed else self.spike_payoff) * prox

        noise = _rng(self.seed, context.name, rep, _key(params)).gauss(0.0, self.noise_sd)
        return plateau + spike + noise

    def objective(self, params: ParamDict, context: SearchContext, rep: int) -> ObjectiveResult:
        return ObjectiveResult(
            performance_metrics={"score": self.raw_score(params, context, rep)},
            sample_count=1,
        )

    def region_kind(self, params: ParamDict) -> str:
        if _ndist(params, self._pc) <= self.plateau_flat_radius + self.plateau_falloff:
            return "plateau"
        if _ndist(params, self._sc) <= 2.0 * self.spike_width:
            return "spike"
        return "neither"

    robust_label = "plateau"
    trap_label = "spike"
    evaluation_metric = "cvar"


@dataclass
class ScalableDecoy:
    """Overfitting trap at arbitrary dimension, context count and decoy density.

    ``n_search_contexts`` is the difficulty dial that matters most: overfitting
    is a small-sample phenomenon, so fewer contexts means more room for a decoy
    to look good by luck.
    """

    dim: int = 2
    n_search_contexts: int = 6
    n_holdout_contexts: int = 24
    n_decoys: int = 40
    decoy_amplitude_sd: float = 1.70
    decoy_width: float = 0.11
    true_value: float = 0.28
    true_height: float = 0.60
    true_width: float = 0.30
    noise_sd: float = 0.04
    seed: int = 7

    def __post_init__(self) -> None:
        names = _names(self.dim)
        self._tc = {n: self.true_value for n in names}
        rng = random.Random(_stable_hash(self.seed, "decoys", self.dim, self.n_decoys))
        self._decoys: List[Dict[str, float]] = []
        attempts = 0
        while len(self._decoys) < self.n_decoys and attempts < 20000:
            attempts += 1
            c = {n: rng.uniform(0.0, 1.0) for n in names}
            if _ndist(c, self._tc) > self.true_width:
                self._decoys.append(c)

    @property
    def parameter_space(self) -> List[ParameterSpec]:
        return [ParameterSpec(n, 0.0, 1.0) for n in _names(self.dim)]

    @property
    def metric_rules(self) -> Dict[str, MetricRule]:
        return {"score": MetricRule(good_value=1.5, bad_value=-1.5, higher_is_better=True)}

    @property
    def metric_weights(self) -> Dict[str, float]:
        return {"score": 1.0}

    @property
    def search_contexts(self) -> List[SearchContext]:
        return [SearchContext(name=f"window_{i:02d}", data={"idx": i})
                for i in range(self.n_search_contexts)]

    @property
    def holdout_contexts(self) -> List[SearchContext]:
        return [SearchContext(name=f"oos_{i:02d}", data={"idx": 1000 + i})
                for i in range(self.n_holdout_contexts)]

    @property
    def survival_threshold(self) -> float:
        return self.metric_rules["score"].normalize(0.0)

    def raw_score(self, params: ParamDict, context: SearchContext, rep: int) -> float:
        dt = _ndist(params, self._tc)
        true_edge = self.true_height * math.exp(-0.5 * (dt / self.true_width) ** 2)

        decoys = 0.0
        for k, center in enumerate(self._decoys):
            d = _ndist(params, center)
            prox = math.exp(-0.5 * (d / self.decoy_width) ** 2)
            if prox > 1e-4:
                amp = _rng(self.seed, "decoy", k, context.name).gauss(0.0, self.decoy_amplitude_sd)
                decoys += amp * prox

        noise = _rng(self.seed, context.name, rep, _key(params)).gauss(0.0, self.noise_sd)
        return true_edge + decoys + noise

    def objective(self, params: ParamDict, context: SearchContext, rep: int) -> ObjectiveResult:
        return ObjectiveResult(
            performance_metrics={"score": self.raw_score(params, context, rep)},
            sample_count=1,
        )

    def region_kind(self, params: ParamDict) -> str:
        if _ndist(params, self._tc) <= self.true_width:
            return "true_edge"
        for center in self._decoys:
            if _ndist(params, center) <= 2.0 * self.decoy_width:
                return "decoy"
        return "neither"

    robust_label = "true_edge"
    trap_label = "decoy"
    evaluation_metric = "mean"


__all__ = ["ScalablePlateauSpike", "ScalableDecoy"]
