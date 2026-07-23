"""
Which context aggregator should you actually use?

The seam in :mod:`narrs.aggregators` makes the reduction over contexts
swappable. This module measures whether swapping it *helps*, because a knob that
changes nothing is worse than no knob -- it invites tuning on noise.

Method: build a pool of candidate configurations, score them over contexts, and
run CSCV. The **selection** statistic is the aggregator under test; the
**evaluation** statistic is held fixed at the thing you actually care about
(out-of-sample tail). The reported number is PBO -- the probability that the
configuration this aggregator picks in-sample lands at or below the median
out-of-sample. Lower is better; 0.5 is a coin flip.

Two regimes are tested separately, because they give opposite answers and
conflating them is how you end up with a superstition:

**Structural tail** -- some contexts genuinely break the strategy (a regime
exists in which it loses). The bad contexts are signal.

**Noise tail** -- no context is structurally bad; per-context estimates are just
noisy. The bad contexts are measurement error.

What the measurements say (see ``README``): the tail-versus-mean decision
dominates everything, and only in the structural regime. Among tail measures,
the sharper the better when the tail is real -- so plain worst-case is hard to
beat, and a high CVaR ``alpha`` actively erodes the benefit by averaging genuine
crashes together with good contexts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..aggregators import resolve_aggregator
from ..metrics import cvar, pbo_cscv
from .problems import PlateauSpikeProblem, candidate_pool, performance_matrix

# (label, builtin name, alpha)
DEFAULT_AGGREGATORS: List[Tuple[str, str, float]] = [
    ("worst_case", "worst_case", 0.25),
    ("cvar@0.25", "cvar", 0.25),
    ("cvar@0.50", "cvar", 0.50),
    ("mean_std k=1", "mean_std", 1.0),
    ("mean", "mean", 0.25),
]


def aggregator_pbo(
    problem,
    aggregators: Sequence[Tuple[str, str, float]] = tuple(DEFAULT_AGGREGATORS),
    pool_size: int = 40,
    n_splits: int = 8,
    seed: int = 5,
) -> Dict[str, float]:
    """PBO of selecting with each aggregator, judged on out-of-sample tail.

    ``reps=1`` so that one matrix row is one context: the aggregator then reduces
    exactly the per-context means it would see inside the optimizer.
    """
    anchors = []
    for attr in ("_pc", "_sc", "_tc"):
        if hasattr(problem, attr):
            anchors.append(dict(getattr(problem, attr)))
    pool = candidate_pool(problem, pool_size, seed=seed, include=anchors)
    matrix = performance_matrix(problem, pool, problem.search_contexts, reps=1)

    def evaluate(xs):
        return cvar(xs, 0.25)

    out: Dict[str, float] = {}
    for label, name, alpha in aggregators:
        agg = resolve_aggregator(name, alpha)
        result = pbo_cscv(
            matrix,
            n_splits=n_splits,
            is_metric=lambda xs, g=agg: g.aggregate(xs),
            oos_metric=evaluate,
        )
        out[label] = result["pbo"]
    return out


def run_study(
    context_counts: Sequence[int] = (12, 32),
    seed: int = 5,
    verbose: bool = True,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Compare aggregators across both tail regimes and several context counts."""
    regimes = {
        "structural tail (real crash regimes)": dict(crash_fraction=0.20, noise_sd=0.30),
        "noise tail (no crash, noisy contexts)": dict(crash_fraction=0.0, noise_sd=0.60),
    }
    labels = [a[0] for a in DEFAULT_AGGREGATORS]
    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    if verbose:
        print("=" * 92)
        print("Aggregator study -- PBO of the selection rule, judged on out-of-sample tail")
        print("lower is better; 0.5 = coin flip")
        print("=" * 92)

    for regime_name, kwargs in regimes.items():
        results[regime_name] = {}
        if verbose:
            print(f"\n{regime_name}")
            header = f"{'contexts':>9s} | " + " ".join(f"{l:>13s}" for l in labels)
            print(header)
            print("-" * len(header))
        for n_ctx in context_counts:
            problem = PlateauSpikeProblem(n_search_contexts=n_ctx, **kwargs)
            row = aggregator_pbo(problem, seed=seed)
            results[regime_name][n_ctx] = row
            if verbose:
                print(f"{n_ctx:>9d} | " + " ".join(f"{row[l]:>13.2f}" for l in labels))

    if verbose:
        print("\n" + "=" * 92)
        print("Reading this table:")
        print("  * Structural tail: every tail-aware rule beats `mean` by a wide margin,")
        print("    and the margin widens with more contexts. That gap is the whole case")
        print("    for robust selection.")
        print("  * Among tail rules, sharper wins when the tail is real. cvar@0.50 averages")
        print("    genuine crashes together with good contexts and gives most of the")
        print("    advantage back, so alpha is a real knob -- set it too high and you have")
        print("    quietly re-invented the mean.")
        print("  * Noise tail: the ordering inverts. With nothing structural to protect")
        print("    against, worst_case is the *weakest* rule -- a single unlucky context")
        print("    dictates the whole score, so it chases measurement error. Here CVaR's")
        print("    tail-averaging is the advantage, and plain mean is fine.")
        print("  * So there is no globally best aggregator. The choice encodes a belief")
        print("    about whether your bad contexts are signal or noise, which is why this")
        print("    is a pluggable seam and not a hard-coded upgrade.")
        print("=" * 92)
    return results


__all__ = ["aggregator_pbo", "run_study", "DEFAULT_AGGREGATORS"]
