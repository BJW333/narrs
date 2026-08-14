"""
Adversarial landscapes: built to BREAK NARRS, not to be solved by it.

``scalable.py`` varies the difficulty of problems NARRS is designed for. This
module is different in intent: every landscape here attacks a specific
assumption the confidence machinery rests on. The goal is to find the failure
boundary before real money does.

Each class documents the assumption it violates and what a HONEST system should
do about it. In most cases the correct answer is not "find the right region" --
it is "notice that you cannot know, and say so". A landscape here is passed not
by surviving it, but by refusing to claim high confidence when it shouldn't.

  RegimeShift        holdout is drawn from a DIFFERENT landscape than search.
                     Violates: search and holdout are exchangeable.
                     This is the real-world killer -- a strategy fit on 2023
                     deployed into 2024. NARRS validates on holdout, so if the
                     holdout itself has moved, validation is measuring the wrong
                     thing. Honest behaviour: low confidence or abstain.

  CorrelatedContexts N contexts that are near-duplicates of each other.
                     Violates: contexts are independent evidence.
                     Overlapping backtest windows look like 12 samples but carry
                     maybe 2 samples of information. Every noise and stability
                     statistic is computed as if n=12, so confidence intervals
                     are far too tight. Honest behaviour: not high confidence.

  HeavyTailNoise     Student-t / jump noise instead of Gaussian.
                     Violates: noise is well-behaved and its standard deviation
                     is meaningful. With fat tails the sample sd understates the
                     real risk, and a "quiet" sample is quiet only until the jump.
                     Honest behaviour: notice the noise estimate is unstable.

  DeceptiveMultiModal  Many near-equal broad plateaus; only one survives OOS.
                     Violates: robustness in-sample implies robustness OOS.
                     All candidates look equally stable during search, so there
                     is no signal to choose between them -- selection is a
                     coin flip dressed up as analysis. Honest behaviour: flag
                     that the choice was arbitrary.

  ContextOutlier     One context is wildly different from the rest.
                     Tests the aggregator seam directly: worst_case should be
                     dominated by it, mean should ignore it. Not a failure mode
                     so much as a behaviour that must be predictable.

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


def _rng(*parts) -> random.Random:
    """Stable across processes -- never hash() on a string."""
    return random.Random(zlib.crc32("|".join(str(p) for p in parts).encode()))


def _key(params: ParamDict) -> str:
    return ",".join(f"{k}={params[k]:.10g}" for k in sorted(params))


def _ndist(params: ParamDict, center: Dict[str, float]) -> float:
    if not center:
        return 0.0
    return math.sqrt(sum((params[k] - center[k]) ** 2 for k in center) / len(center))


def _names(dim: int) -> List[str]:
    return [f"x{i}" for i in range(dim)]


# --------------------------------------------------------------------------- #
@dataclass
class RegimeShift:
    """The optimum MOVES between search and holdout.

    A region that is genuinely excellent during search is mediocre or bad on
    holdout, not because it was overfit, but because the world changed. No
    amount of in-sample robustness can detect this -- which is the point. The
    only honest response is to not claim high confidence.
    """

    dim: int = 2
    n_search_contexts: int = 10
    n_holdout_contexts: int = 10
    shift: float = 0.35          # how far the optimum moves (0 = no shift)
    noise_sd: float = 0.05
    height: float = 0.85
    width: float = 0.20
    seed: int = 3

    @property
    def parameter_space(self) -> List[ParameterSpec]:
        return [ParameterSpec(n, 0.0, 1.0) for n in _names(self.dim)]

    @property
    def metric_rules(self) -> Dict[str, MetricRule]:
        return {"score": MetricRule(good_value=1.0, bad_value=-1.0)}

    @property
    def metric_weights(self) -> Dict[str, float]:
        return {"score": 1.0}

    @property
    def search_contexts(self) -> List[SearchContext]:
        return [SearchContext(f"in_{i:02d}", data={"oos": False})
                for i in range(self.n_search_contexts)]

    @property
    def holdout_contexts(self) -> List[SearchContext]:
        return [SearchContext(f"oos_{i:02d}", data={"oos": True})
                for i in range(self.n_holdout_contexts)]

    @property
    def survival_threshold(self) -> float:
        return self.metric_rules["score"].normalize(0.0)

    def _center(self, is_oos: bool) -> Dict[str, float]:
        base = 0.30 + (self.shift if is_oos else 0.0)
        return {n: base for n in _names(self.dim)}

    def raw_score(self, params, context, rep) -> float:
        is_oos = bool((context.data or {}).get("oos", False))
        d = _ndist(params, self._center(is_oos))
        peak = self.height * math.exp(-0.5 * (d / self.width) ** 2)
        return peak + _rng(self.seed, context.name, rep, _key(params)).gauss(0, self.noise_sd)

    def objective(self, params, context, rep) -> ObjectiveResult:
        return ObjectiveResult(performance_metrics={"score": self.raw_score(params, context, rep)},
                               sample_count=1)

    def region_kind(self, params) -> str:
        if _ndist(params, self._center(False)) <= self.width:
            return "in_sample_optimum"
        if _ndist(params, self._center(True)) <= self.width:
            return "oos_optimum"
        return "neither"

    robust_label = "oos_optimum"
    trap_label = "in_sample_optimum"
    evaluation_metric = "mean"


# --------------------------------------------------------------------------- #
@dataclass
class CorrelatedContexts:
    """N contexts that are near-duplicates: n=12 that carries n=2 of information.

    Every context is generated from one of ``n_independent`` underlying regimes,
    with only a tiny per-context perturbation. NARRS computes noise and stability
    across all N as if they were independent, so its confidence intervals are far
    too tight. This is what overlapping backtest windows do in practice.
    """

    dim: int = 2
    n_search_contexts: int = 12
    n_holdout_contexts: int = 12
    n_independent: int = 2       # true independent regimes behind the N contexts
    crash_regime_fraction: float = 0.5
    noise_sd: float = 0.04
    seed: int = 5

    @property
    def parameter_space(self) -> List[ParameterSpec]:
        return [ParameterSpec(n, 0.0, 1.0) for n in _names(self.dim)]

    @property
    def metric_rules(self) -> Dict[str, MetricRule]:
        return {"score": MetricRule(good_value=1.5, bad_value=-1.5)}

    @property
    def metric_weights(self) -> Dict[str, float]:
        return {"score": 1.0}

    def _contexts(self, prefix: str, n: int) -> List[SearchContext]:
        out = []
        for i in range(n):
            regime = i % self.n_independent
            out.append(SearchContext(f"{prefix}_{i:02d}",
                                     data={"regime": regime, "idx": i}))
        return out

    @property
    def search_contexts(self) -> List[SearchContext]:
        return self._contexts("win", self.n_search_contexts)

    @property
    def holdout_contexts(self) -> List[SearchContext]:
        # holdout draws from regimes the search set never saw
        out = []
        for i in range(self.n_holdout_contexts):
            regime = self.n_independent + (i % self.n_independent)
            out.append(SearchContext(f"oos_{i:02d}", data={"regime": regime, "idx": i}))
        return out

    @property
    def survival_threshold(self) -> float:
        return self.metric_rules["score"].normalize(0.0)

    def raw_score(self, params, context, rep) -> float:
        regime = (context.data or {}).get("regime", 0)
        d = _ndist(params, {n: 0.35 for n in _names(self.dim)})
        base = 0.9 * math.exp(-0.5 * (d / 0.22) ** 2)
        # regimes flip the sign of the edge; search only ever sees the good ones
        regime_multiplier = 1.0 if regime < self.n_independent else -0.8
        # near-duplicate contexts within a regime: perturbation is tiny
        jitter = _rng(self.seed, "reg", regime, context.name).gauss(0, 0.01)
        noise = _rng(self.seed, context.name, rep, _key(params)).gauss(0, self.noise_sd)
        return base * regime_multiplier + jitter + noise

    def objective(self, params, context, rep) -> ObjectiveResult:
        return ObjectiveResult(performance_metrics={"score": self.raw_score(params, context, rep)},
                               sample_count=1)

    def region_kind(self, params) -> str:
        return "edge" if _ndist(params, {n: 0.35 for n in _names(self.dim)}) <= 0.22 else "neither"

    robust_label = "edge"
    trap_label = "neither"
    evaluation_metric = "mean"


# --------------------------------------------------------------------------- #
@dataclass
class HeavyTailNoise:
    """Student-t noise with occasional jumps: the sample sd understates the risk.

    A Gaussian assumption makes a quiet sample look safe. Here, most reps are
    calm and a rare rep is catastrophic, so ``holdout_noise`` reads low right up
    until it doesn't. Directly attacks the fragility model, which treats noise as
    the primary survival signal.
    """

    dim: int = 2
    n_search_contexts: int = 10
    n_holdout_contexts: int = 10
    tail_df: float = 1.8          # lower = fatter tails (t-distribution dof)
    jump_probability: float = 0.04
    jump_size: float = 2.5
    base_noise: float = 0.04
    seed: int = 9

    @property
    def parameter_space(self) -> List[ParameterSpec]:
        return [ParameterSpec(n, 0.0, 1.0) for n in _names(self.dim)]

    @property
    def metric_rules(self) -> Dict[str, MetricRule]:
        return {"score": MetricRule(good_value=1.5, bad_value=-1.5)}

    @property
    def metric_weights(self) -> Dict[str, float]:
        return {"score": 1.0}

    @property
    def search_contexts(self) -> List[SearchContext]:
        return [SearchContext(f"in_{i:02d}") for i in range(self.n_search_contexts)]

    @property
    def holdout_contexts(self) -> List[SearchContext]:
        return [SearchContext(f"oos_{i:02d}") for i in range(self.n_holdout_contexts)]

    @property
    def survival_threshold(self) -> float:
        return self.metric_rules["score"].normalize(0.0)

    def _heavy_noise(self, rng: random.Random) -> float:
        # t-distributed via normal / sqrt(chi2/df), approximated with gammavariate
        z = rng.gauss(0.0, 1.0)
        chi2 = rng.gammavariate(self.tail_df / 2.0, 2.0)
        t = z / math.sqrt(max(chi2 / self.tail_df, 1e-9))
        noise = self.base_noise * t
        if rng.random() < self.jump_probability:
            noise += rng.choice((-1.0, 1.0)) * self.jump_size
        return noise

    def raw_score(self, params, context, rep) -> float:
        d = _ndist(params, {n: 0.4 for n in _names(self.dim)})
        base = 0.8 * math.exp(-0.5 * (d / 0.25) ** 2)
        return base + self._heavy_noise(_rng(self.seed, context.name, rep, _key(params)))

    def objective(self, params, context, rep) -> ObjectiveResult:
        return ObjectiveResult(performance_metrics={"score": self.raw_score(params, context, rep)},
                               sample_count=1)

    def region_kind(self, params) -> str:
        return "edge" if _ndist(params, {n: 0.4 for n in _names(self.dim)}) <= 0.25 else "neither"

    robust_label = "edge"
    trap_label = "neither"
    evaluation_metric = "cvar"


# --------------------------------------------------------------------------- #
@dataclass
class DeceptiveMultiModal:
    """Many equally-good broad plateaus in-sample; only one holds up OOS.

    There is genuinely no in-sample signal distinguishing the survivor from the
    decoys -- they are identical in mean, spread and width. Any pick is a
    1-in-``n_modes`` coin flip. A system that reports high confidence here is
    claiming knowledge it provably does not have.
    """

    dim: int = 2
    n_modes: int = 5
    n_search_contexts: int = 10
    n_holdout_contexts: int = 10
    noise_sd: float = 0.04
    seed: int = 13

    def __post_init__(self):
        names = _names(self.dim)
        rng = random.Random(zlib.crc32(f"modes|{self.seed}|{self.dim}|{self.n_modes}".encode()))
        self._modes: List[Dict[str, float]] = []
        # Bounded rejection sampling with a relaxing separation requirement.
        # An unbounded "keep trying until they are all far enough apart" loop
        # never terminates when the requested separation is geometrically
        # impossible for n_modes in the box -- which it is at the default
        # settings, because _ndist divides by sqrt(dim). Relax the requirement
        # rather than spin forever.
        separation = 0.28
        attempts = 0
        while len(self._modes) < self.n_modes:
            attempts += 1
            if attempts > 2000:          # give up on this separation, loosen it
                separation *= 0.8
                attempts = 0
                if separation < 0.02:    # accept whatever we have plus fillers
                    while len(self._modes) < self.n_modes:
                        self._modes.append({n: rng.uniform(0.15, 0.85) for n in names})
                    break
            c = {n: rng.uniform(0.15, 0.85) for n in names}
            if all(_ndist(c, m) > separation for m in self._modes):
                self._modes.append(c)
        self._survivor = 0   # only mode 0 survives out of sample

    @property
    def parameter_space(self) -> List[ParameterSpec]:
        return [ParameterSpec(n, 0.0, 1.0) for n in _names(self.dim)]

    @property
    def metric_rules(self) -> Dict[str, MetricRule]:
        return {"score": MetricRule(good_value=1.0, bad_value=-1.0)}

    @property
    def metric_weights(self) -> Dict[str, float]:
        return {"score": 1.0}

    @property
    def search_contexts(self) -> List[SearchContext]:
        return [SearchContext(f"in_{i:02d}", data={"oos": False})
                for i in range(self.n_search_contexts)]

    @property
    def holdout_contexts(self) -> List[SearchContext]:
        return [SearchContext(f"oos_{i:02d}", data={"oos": True})
                for i in range(self.n_holdout_contexts)]

    @property
    def survival_threshold(self) -> float:
        return self.metric_rules["score"].normalize(0.0)

    def raw_score(self, params, context, rep) -> float:
        is_oos = bool((context.data or {}).get("oos", False))
        best = 0.0
        for k, c in enumerate(self._modes):
            d = _ndist(params, c)
            amp = 0.80
            if is_oos and k != self._survivor:
                amp = -0.40          # decoys invert out of sample
            best = max(best, amp * math.exp(-0.5 * (d / 0.16) ** 2)) if amp > 0 else \
                   min(best, amp * math.exp(-0.5 * (d / 0.16) ** 2)) if best == 0 else best
            if is_oos and k != self._survivor:
                best += amp * math.exp(-0.5 * (d / 0.16) ** 2)
        return best + _rng(self.seed, context.name, rep, _key(params)).gauss(0, self.noise_sd)

    def objective(self, params, context, rep) -> ObjectiveResult:
        return ObjectiveResult(performance_metrics={"score": self.raw_score(params, context, rep)},
                               sample_count=1)

    def region_kind(self, params) -> str:
        for k, c in enumerate(self._modes):
            if _ndist(params, c) <= 0.16:
                return "survivor" if k == self._survivor else "decoy_mode"
        return "neither"

    robust_label = "survivor"
    trap_label = "decoy_mode"
    evaluation_metric = "mean"


__all__ = ["RegimeShift", "CorrelatedContexts", "HeavyTailNoise", "DeceptiveMultiModal"]
