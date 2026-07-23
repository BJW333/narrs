"""
Synthetic landscapes with a known right answer, used to test whether a search
procedure is fooled.

Two distinct failure modes get two distinct landscapes, because they are not the
same thing and a single benchmark cannot measure both:

``PlateauSpikeProblem`` -- a **risk trap**. A broad flat plateau with a steady
edge, versus a taller region that pays more *on average* but collapses in a
minority of regimes. Nothing here is statistically unstable: the spike's high
mean is entirely real and reproduces out of sample. It is simply a bad thing to
deploy. Mean-maximising optimizers take it. Tail-aware selection does not. The
metric that separates them is out-of-sample CVaR, not PBO -- PBO would correctly
report ~0 here, because in-sample ranking *does* predict out-of-sample ranking.

``DecoyOverfittingProblem`` -- an **overfitting trap**. A modest but genuinely
stationary edge, surrounded by decoy peaks whose height is pure per-context
noise. Average a decoy over a handful of contexts and it can look excellent by
luck; average it over fresh contexts and it reverts to nothing. This is the
failure CSCV/PBO was designed to detect, and it needs limited context evidence
to appear at all -- with enough contexts the decoys wash out and no honest
procedure overfits.

Both expose the standard NARRS setup (``parameter_space``, ``search_contexts``,
``holdout_contexts``, ``metric_rules``, ``metric_weights``) plus an
``objective`` with the usual ``(params, context, rep)`` signature, so they drop
straight into ``NoiseAwareRobustRegionSearch`` with no adapter.

Pure standard library.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..types import (
    MetricRule,
    ObjectiveResult,
    ParamDict,
    ParameterSpec,
    SearchContext,
)


def _stable_hash(*parts) -> int:
    """Deterministic hash across processes and runs.

    Python randomises ``hash()`` for strings per process (PYTHONHASHSEED), and
    NARRS can evaluate candidates in a process pool, so using the builtin here
    would make results irreproducible and silently inconsistent between workers.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(payload)


def _rng(*parts) -> random.Random:
    return random.Random(_stable_hash(*parts))


def _point_key(params: ParamDict) -> str:
    return ",".join(f"{k}={params[k]:.10g}" for k in sorted(params))


def _dist(params: ParamDict, center: Dict[str, float]) -> float:
    return math.sqrt(sum((params[k] - center[k]) ** 2 for k in center))


# --------------------------------------------------------------------------- #
# Landscape 1: risk trap                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class PlateauSpikeProblem:
    """Broad steady plateau vs a high-mean region that crashes in some regimes.

    The crash is a property of the *context*, not of the repetition: a regime
    either breaks the spike or it does not. That is the honest analogue of a
    strategy that works until a particular market condition appears, and it means
    the danger is visible to any procedure that looks across contexts rather than
    averaging them away.
    """

    plateau_center: Tuple[float, float] = (0.30, 0.30)
    plateau_height: float = 0.85
    plateau_flat_radius: float = 0.16
    plateau_falloff: float = 0.22

    spike_center: Tuple[float, float] = (0.75, 0.75)
    spike_width: float = 0.22
    spike_payoff: float = 3.0
    spike_crash: float = -3.0
    crash_fraction: float = 0.20

    noise_sd: float = 0.05
    n_search_contexts: int = 10
    n_holdout_contexts: int = 10
    seed: int = 11

    def __post_init__(self) -> None:
        self._pc = {"x": self.plateau_center[0], "y": self.plateau_center[1]}
        self._sc = {"x": self.spike_center[0], "y": self.spike_center[1]}

    # -- NARRS setup -------------------------------------------------------- #
    @property
    def parameter_space(self) -> List[ParameterSpec]:
        return [ParameterSpec("x", 0.0, 1.0), ParameterSpec("y", 0.0, 1.0)]

    @property
    def metric_rules(self) -> Dict[str, MetricRule]:
        # Bounds chosen so neither the plateau nor either spike outcome clips:
        # clipping would compress exactly the differences under test.
        return {"score": MetricRule(good_value=3.0, bad_value=-3.0, higher_is_better=True)}

    @property
    def metric_weights(self) -> Dict[str, float]:
        return {"score": 1.0}

    def _contexts(self, prefix: str, n: int) -> List[SearchContext]:
        """Exactly ``crash_fraction`` of contexts are crash regimes.

        Deterministic rather than sampled: if the number of stressed regimes were
        left to chance, a run could contain none and the benchmark would silently
        stop testing what it claims to test.
        """
        n_crash = max(1, int(round(self.crash_fraction * n)))
        rng = random.Random(_stable_hash(self.seed, prefix, "regimes"))
        crash_idx = set(rng.sample(range(n), n_crash))
        return [
            SearchContext(
                name=f"{prefix}_{i:02d}",
                data={"crash": i in crash_idx},
            )
            for i in range(n)
        ]

    @property
    def search_contexts(self) -> List[SearchContext]:
        return self._contexts("regime", self.n_search_contexts)

    @property
    def holdout_contexts(self) -> List[SearchContext]:
        return self._contexts("oos", self.n_holdout_contexts)

    # -- the landscape ------------------------------------------------------ #
    def raw_score(self, params: ParamDict, context: SearchContext, rep: int) -> float:
        d_plateau = _dist(params, self._pc)
        if d_plateau <= self.plateau_flat_radius:
            plateau = self.plateau_height
        else:
            over = d_plateau - self.plateau_flat_radius
            plateau = self.plateau_height * math.exp(-0.5 * (over / self.plateau_falloff) ** 2)

        d_spike = _dist(params, self._sc)
        proximity = math.exp(-0.5 * (d_spike / self.spike_width) ** 2)
        crashed = bool((context.data or {}).get("crash", False))
        spike = (self.spike_crash if crashed else self.spike_payoff) * proximity

        noise = _rng(self.seed, context.name, rep, _point_key(params)).gauss(0.0, self.noise_sd)
        return plateau + spike + noise

    def objective(self, params: ParamDict, context: SearchContext, rep: int) -> ObjectiveResult:
        return ObjectiveResult(
            performance_metrics={"score": self.raw_score(params, context, rep)},
            sample_count=1,
        )


    @property
    def survival_threshold(self) -> float:
        """The normalized score of a raw zero: "did this beat doing nothing?".

        Deriving the bar from the metric rule rather than hand-picking a number
        keeps the benchmark honest -- a threshold tuned until the desired method
        passes would prove nothing.
        """
        return self.metric_rules["score"].normalize(0.0)

    def region_kind(self, params: ParamDict) -> str:
        """Ground truth label for a chosen point: which feature did it land on?"""
        if _dist(params, self._pc) <= self.plateau_flat_radius + self.plateau_falloff:
            return "plateau"
        if _dist(params, self._sc) <= 2.0 * self.spike_width:
            return "spike"
        return "neither"

    robust_label = "plateau"
    trap_label = "spike"
    headline = "out-of-sample tail risk (CVaR)"
    # What "good out of sample" means here. The spike's *mean* is genuinely high
    # and reproduces perfectly; it is the tail that makes it undeployable, so the
    # tail is the criterion any selection rule has to be judged against.
    evaluation_metric = "cvar"


# --------------------------------------------------------------------------- #
# Landscape 2: overfitting trap                                                #
# --------------------------------------------------------------------------- #
@dataclass
class DecoyOverfittingProblem:
    """A real but modest edge, hidden among peaks made of pure per-context luck.

    Each decoy draws a fresh amplitude in every context. Over few contexts some
    decoy will look outstanding; over fresh contexts it reverts to zero. The true
    edge is identical in every context, so it survives.
    """

    true_center: Tuple[float, float] = (0.28, 0.28)
    true_height: float = 0.60
    true_width: float = 0.30

    n_decoys: int = 40
    decoy_width: float = 0.11
    decoy_amplitude_sd: float = 1.70

    noise_sd: float = 0.04
    n_search_contexts: int = 6
    n_holdout_contexts: int = 24
    seed: int = 7

    def __post_init__(self) -> None:
        self._tc = {"x": self.true_center[0], "y": self.true_center[1]}
        rng = random.Random(_stable_hash(self.seed, "decoys"))
        self._decoys: List[Dict[str, float]] = []
        attempts = 0
        while len(self._decoys) < self.n_decoys and attempts < 2000:
            attempts += 1
            c = {"x": rng.uniform(0.0, 1.0), "y": rng.uniform(0.0, 1.0)}
            # Keep decoys clear of the true edge so the edge stays a clean signal.
            if _dist(c, self._tc) > self.true_width:
                self._decoys.append(c)

    @property
    def parameter_space(self) -> List[ParameterSpec]:
        return [ParameterSpec("x", 0.0, 1.0), ParameterSpec("y", 0.0, 1.0)]

    @property
    def metric_rules(self) -> Dict[str, MetricRule]:
        return {"score": MetricRule(good_value=1.5, bad_value=-1.5, higher_is_better=True)}

    @property
    def metric_weights(self) -> Dict[str, float]:
        return {"score": 1.0}

    @property
    def search_contexts(self) -> List[SearchContext]:
        return [
            SearchContext(name=f"window_{i:02d}", data={"idx": i})
            for i in range(self.n_search_contexts)
        ]

    @property
    def holdout_contexts(self) -> List[SearchContext]:
        return [
            SearchContext(name=f"oos_{i:02d}", data={"idx": 1000 + i})
            for i in range(self.n_holdout_contexts)
        ]

    def raw_score(self, params: ParamDict, context: SearchContext, rep: int) -> float:
        d_true = _dist(params, self._tc)
        true_edge = self.true_height * math.exp(-0.5 * (d_true / self.true_width) ** 2)

        decoy_total = 0.0
        for k, center in enumerate(self._decoys):
            d = _dist(params, center)
            proximity = math.exp(-0.5 * (d / self.decoy_width) ** 2)
            if proximity > 1e-4:
                amp = _rng(self.seed, "decoy", k, context.name).gauss(0.0, self.decoy_amplitude_sd)
                decoy_total += amp * proximity

        noise = _rng(self.seed, context.name, rep, _point_key(params)).gauss(0.0, self.noise_sd)
        return true_edge + decoy_total + noise

    def objective(self, params: ParamDict, context: SearchContext, rep: int) -> ObjectiveResult:
        return ObjectiveResult(
            performance_metrics={"score": self.raw_score(params, context, rep)},
            sample_count=1,
        )


    @property
    def survival_threshold(self) -> float:
        """The normalized score of a raw zero: "did this beat doing nothing?".

        Deriving the bar from the metric rule rather than hand-picking a number
        keeps the benchmark honest -- a threshold tuned until the desired method
        passes would prove nothing.
        """
        return self.metric_rules["score"].normalize(0.0)

    def region_kind(self, params: ParamDict) -> str:
        if _dist(params, self._tc) <= self.true_width:
            return "true_edge"
        for center in self._decoys:
            if _dist(params, center) <= 2.0 * self.decoy_width:
                return "decoy"
        return "neither"

    robust_label = "true_edge"
    trap_label = "decoy"
    headline = "probability of backtest overfitting (PBO)"
    # No fat tails here by construction -- decoys fail by reverting to zero, not
    # by crashing. Mean is the honest criterion.
    evaluation_metric = "mean"


# --------------------------------------------------------------------------- #
# shared helpers                                                               #
# --------------------------------------------------------------------------- #
def normalized_score(problem, params: ParamDict, context: SearchContext, rep: int) -> float:
    """Raw objective mapped through the problem's MetricRule into [0, 1].

    This is the same scale NARRS scores on, so comparisons between NARRS and the
    external baselines are like-for-like.
    """
    rule = problem.metric_rules["score"]
    return rule.normalize(problem.raw_score(params, context, rep))


def candidate_pool(
    problem,
    n: int,
    seed: int = 0,
    include: Sequence[ParamDict] = (),
) -> List[ParamDict]:
    """Random candidate configurations, optionally seeded with specific points.

    ``include`` matters for PBO: the pool has to actually contain the features
    that can fool a selection rule, otherwise you measure overfitting on a
    landscape with nothing to overfit to.
    """
    rng = random.Random(seed)
    specs = problem.parameter_space
    pool: List[ParamDict] = [dict(p) for p in include][:n]
    while len(pool) < n:
        pool.append({s.name: rng.uniform(s.min_value, s.max_value) for s in specs})
    return pool


def performance_matrix(
    problem,
    points: Sequence[ParamDict],
    contexts: Sequence[SearchContext],
    reps: int = 1,
) -> List[List[float]]:
    """Observations x candidates matrix of normalized scores, for CSCV/PBO.

    Rows are ordered context-major so that contiguous CSCV blocks correspond to
    contiguous groups of contexts -- which is what makes the split meaningful
    rather than an arbitrary reshuffle.
    """
    matrix: List[List[float]] = []
    for context in contexts:
        for rep in range(reps):
            matrix.append([normalized_score(problem, p, context, rep) for p in points])
    return matrix


__all__ = [
    "PlateauSpikeProblem",
    "DecoyOverfittingProblem",
    "normalized_score",
    "candidate_pool",
    "performance_matrix",
]
