"""
p_survive: a calibrated probability that a recommended region survives live.

Design note -- read this before changing anything, because the obvious version
does not work and the reason is instructive.

The intuitive definition of "survives" is "its true out-of-sample performance
clears the acceptance bar". Measured that way, ``p_survive`` is useless: NARRS
only recommends regions it already believes in, so by the time we compute this
the holdout mean sits 5 to 150+ standard errors above the bar. Every tail
probability saturates to 1.0, carrying no information and giving a self-tuning
loop nothing to optimise. (The empirical demonstration is in
``tests/reliability.py`` -- the "clears-the-bar" estimator scores every run at
~1.0 regardless of whether it actually held up.)

The signal that *does* separate the runs that survive from the runs that fail is
**fragility**, and it is visible at decision time. Comparing survivors against
failures on the benchmark: failures have several times the holdout noise and
several times the cross-context instability of survivors, and their in-sample
score decays to nothing out of sample while survivors barely move. None of that
shows up in "distance above the bar"; all of it shows up in how noisy and
regime-dependent the region is.

So ``p_survive`` here is the probability that the region is *robust* -- that its
performance will not decay away when the regime changes -- estimated from the
fragility signals NARRS already computes (holdout noise, context instability,
the in-sample-to-out-of-sample gap), and then **calibrated against realised
survival** so the number means what it says. The mapping from fragility to
probability is not guessed; it is fit and checked by ``tests/reliability.py``,
and the coefficients live here as constants that study produces.

Pure standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Optional

_NORMAL = NormalDist()


# --------------------------------------------------------------------------- #
# calibrated fragility model                                                  #
# --------------------------------------------------------------------------- #
# A logistic model: P(survive) = sigmoid(b0 + b1*context_instability +
# b2*effective_noise + b3*decay_gap). It consumes the RAW signals -- the fitted
# coefficients carry the units. The z_* values reported in `signals` are for
# human inspection only and are NOT model inputs; fitting against them would
# produce coefficients in the wrong units (this was a real bug).
# The defaults below are the coefficients fit by tests/reliability.py on the
# benchmark battery. Re-running that study with --fit prints updated values.
# They are deliberately conservative: an unseen fragility pattern pulls the
# probability toward the base rate, not toward certainty.
_DEFAULT_COEFFS = {
    # Fitted by tests/reliability.py --fit on the benchmark battery (140 runs,
    # in-sample ECE 0.033). holdout_noise dominates: once the out-of-sample noise
    # is known, instability and decay add little, which matches the finding that
    # fragility is mostly "how noisy is this region out of sample". Re-fit if the
    # landscapes or the search change.
    # Fitted by tests/reliability.py --fit (unweighted). Held-out ECE ~0.06:
    # honest AS A PROBABILITY at the top and bottom of its range.
    # KNOWN LIMIT: between roughly 0.45 and 0.75 it is overconfident -- at the
    # average failure's noise it still reads ~0.6. Treat that band as "unknown",
    # not as a real 60% chance. A class-balanced refit
    # (fit_logistic(..., balance_classes=True)) moves the 50% crossover from
    # noise 0.52 to 0.18 and reads 0.20 at the average failure, but then
    # under-rates survivors too (ECE ~0.20). Sharpness and calibration trade off
    # here; the calibrated fit is the default because this value is named, and
    # read, as a probability.
    "intercept": 1.673,
    "context_instability": -0.431,
    "effective_noise": -3.222,
    "decay_gap": 0.761,
}

# Scales that turn a raw signal into its z (roughly "how many bad-units").
_SCALES = {
    "context_instability": 0.10,   # instability of ~0.10 is one unit of concern
    "holdout_noise": 0.15,         # holdout noise of ~0.15 is one unit
    "decay_gap": 0.50,             # a half-point score drop is one unit
}


@dataclass
class SurvivalEstimate:
    p_survive: float
    fragility_index: float          # 0 (robust) .. 1 (fragile), pre-calibration
    signals: Dict[str, float]
    method: str = "calibrated_fragility"


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def estimate_survival(
    holdout_noise: float,
    context_instability: float,
    in_sample_score: float,
    holdout_mean: float,
    region_noise: float = 0.0,
    coeffs: Optional[Dict[str, float]] = None,
) -> SurvivalEstimate:
    """Probability the region is robust enough to survive a regime change.

    Parameters
    ----------
    holdout_noise : spread of the region's out-of-sample scores.
    context_instability : how much the region's score varies across contexts.
    in_sample_score : the region's score during search.
    holdout_mean : the region's mean score on held-out contexts.

    The three fragility signals are scaled and combined through a logistic model
    whose coefficients were calibrated against realised survival. The raw
    weighted sum is also returned as a ``fragility_index`` for inspection.
    """
    c = coeffs or _DEFAULT_COEFFS
    decay_gap = max(0.0, in_sample_score - holdout_mean)

    # holdout_noise alone under-reads when the objective is loud and few reps were
    # averaged -- the same failure that let sd=0.6 landscapes be rated "high".
    # region_noise is the in-sample spread backed by every search point. Take the
    # larger: a region is only as trustworthy as its NOISIEST credible estimate.
    effective_noise = max(holdout_noise, region_noise)

    # The logistic is fit directly on the raw signals (the fitted coefficients
    # carry the units), so a region whose holdout noise is genuinely high is
    # driven to a low probability instead of being clamped near the intercept.
    # z_* are reported for inspection only; they do not enter the model.
    z_instab = context_instability / _SCALES["context_instability"]
    z_noise = holdout_noise / _SCALES["holdout_noise"]
    z_decay = decay_gap / _SCALES["decay_gap"]

    log_odds = (
        c["intercept"]
        + c["context_instability"] * context_instability
        + c["effective_noise"] * effective_noise
        + c["decay_gap"] * decay_gap
    )
    p = _sigmoid(log_odds)

    # A bounded fragility index for humans, independent of the fitted logistic:
    # 0 means all three signals are clean, 1 means all are at/above one bad-unit.
    fragility = min(1.0, (min(z_instab, 3) + min(z_noise, 3) + min(z_decay, 3)) / 9.0)

    return SurvivalEstimate(
        p_survive=p,
        fragility_index=fragility,
        signals={
            "holdout_noise": holdout_noise,
            "region_noise": region_noise,
            "effective_noise": effective_noise,
            "context_instability": context_instability,
            "decay_gap": decay_gap,
            "z_instability": z_instab, "z_noise": z_noise, "z_decay": z_decay,
        },
    )


def estimate_from_region(region, coeffs: Optional[Dict[str, float]] = None
                         ) -> SurvivalEstimate:
    """Convenience wrapper: pull the signals off a NARRS Region object."""
    return estimate_survival(
        holdout_noise=getattr(region, "holdout_noise", 0.0),
        context_instability=getattr(region, "region_context_stability", 0.0),
        # region_in_sample_mean, NOT region_score: the latter is a composite on a
        # different scale, so subtracting holdout_mean from it produced a
        # meaningless "decay" that mostly tracked how good the region was.
        in_sample_score=getattr(region, "region_in_sample_mean", 0.0),
        holdout_mean=getattr(region, "holdout_mean", 0.0),
        region_noise=getattr(region, "region_noise", 0.0),
        coeffs=coeffs,
    )


def rating_from_p_survive(p: float, high: float = 0.80, medium: float = 0.50) -> str:
    """Map the probability back to the existing high/medium/low vocabulary.

    Lets the number coexist with the current label API. Thresholds default here
    but should be set from the reliability study's operating point.
    """
    if p >= high:
        return "high"
    if p >= medium:
        return "medium"
    return "low"


def fit_logistic(
    rows,
    signal_keys=("context_instability", "effective_noise", "decay_gap"),
    iterations: int = 500,
    lr: float = 0.3,
    balance_classes: bool = True,
) -> Dict[str, float]:
    """Fit the logistic coefficients from (signals, survived) rows.

    Plain-stdlib gradient descent -- no numpy. ``rows`` is a list of dicts each
    with the ``signal_keys`` and a boolean ``survived``. Used by
    ``tests/reliability.py --fit`` to regenerate ``_DEFAULT_COEFFS``.
    """
    keys = list(signal_keys)
    w = {k: 0.0 for k in keys}
    b = 0.0
    n = len(rows)
    if n == 0:
        return {"intercept": b, **w}

    # Roughly 90% of recommended regions survive. An unweighted fit is therefore
    # dominated by survivors: it maximises overall likelihood by keeping the
    # intercept high and the slope shallow, which is exactly the observed
    # mid-band overconfidence -- at the average FAILURE's noise the model still
    # reported ~0.6. Weighting each class by its inverse frequency makes the rare
    # failures count as much in aggregate as the common survivals, which steepens
    # the curve where the decision actually gets made. Set balance_classes=False
    # to recover the plain maximum-likelihood fit.
    n_surv = sum(1 for r in rows if r["survived"]) or 1
    n_fail = (n - n_surv) or 1
    if balance_classes:
        w_surv, w_fail = 0.5 * n / n_surv, 0.5 * n / n_fail
    else:
        w_surv = w_fail = 1.0

    for _ in range(iterations):
        gb = 0.0
        gw = {k: 0.0 for k in keys}
        total_weight = 0.0
        for r in rows:
            survived = bool(r["survived"])
            weight = w_surv if survived else w_fail
            z = b + sum(w[k] * r[k] for k in keys)
            err = (_sigmoid(z) - (1.0 if survived else 0.0)) * weight
            gb += err
            for k in keys:
                gw[k] += err * r[k]
            total_weight += weight
        b -= lr * gb / total_weight
        for k in keys:
            w[k] -= lr * gw[k] / total_weight
    return {"intercept": b, **w}


__all__ = [
    "SurvivalEstimate",
    "estimate_survival",
    "estimate_from_region",
    "rating_from_p_survive",
    "fit_logistic",
]
