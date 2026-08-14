"""
Reliability: is p_survive calibrated, and what are its coefficients?

Two jobs, one harness.

``--fit`` collects (fragility signals -> did it survive) across many seeds and
landscapes, fits the logistic model in ``narrs.psurvive``, and prints the
coefficients to paste into ``_DEFAULT_COEFFS``. This is how the probability is
grounded in data rather than guessed.

Without ``--fit`` it evaluates the *current* coefficients: it produces a
reliability curve (predicted probability vs realised survival, binned) and the
expected calibration error. A calibrated model sits on the diagonal -- when it
says 0.7, about 70% survive.

    python tests/reliability.py --fit            # calibrate, prints coefficients
    python tests/reliability.py                  # evaluate current calibration
    python tests/reliability.py --seeds 40        # more data

The landscapes are chosen to span the full fragility range on purpose: easy
cells where everything survives, and hard/noisy/thin cells where a meaningful
fraction fails. Without failures there is nothing to calibrate against, so the
harness deliberately includes cells NARRS gets wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narrs import NARRSConfig, NoiseAwareRobustRegionSearch  # noqa: E402
from narrs.psurvive import estimate_survival, fit_logistic  # noqa: E402
from narrs.benchmarks.battery import _judge  # noqa: E402
from narrs.benchmarks.scalable import ScalableDecoy, ScalablePlateauSpike  # noqa: E402

# Landscapes spanning easy -> hard, so survival actually varies.
_LANDSCAPES = [
    ("PlateauSpike", lambda: ScalablePlateauSpike()),
    ("PlateauSpike/noisy", lambda: ScalablePlateauSpike(noise_sd=0.45)),
    ("PlateauSpike/dim6", lambda: ScalablePlateauSpike(dim=6)),
    ("Decoy", lambda: ScalableDecoy()),
    ("Decoy/thin", lambda: ScalableDecoy(n_search_contexts=3)),
    ("Decoy/hard", lambda: ScalableDecoy(n_decoys=90, decoy_amplitude_sd=3.2)),
    ("Decoy/thin+hard", lambda: ScalableDecoy(n_search_contexts=2, n_decoys=90,
                                              decoy_amplitude_sd=3.2)),
]


def _collect(n_seeds: int, verbose: bool) -> List[Dict]:
    rows: List[Dict] = []
    for pname, make in _LANDSCAPES:
        n_here = 0
        for seed in range(n_seeds):
            problem = make()
            sc, hc = problem.search_contexts, problem.holdout_contexts
            budget = 5000 + 1200 * (len(problem.parameter_space) - 2)
            cfg = NARRSConfig(search_rounds=3, initial_candidate_count=40,
                              repetitions_per_point=2, total_compute_budget=budget,
                              random_seed=seed)
            opt = NoiseAwareRobustRegionSearch(
                objective_function=problem.objective,
                parameter_space=problem.parameter_space,
                search_contexts=sc, holdout_contexts=hc,
                metric_rules=problem.metric_rules,
                metric_weights=problem.metric_weights, config=cfg)
            result = opt.run()
            region = result.get("best_region")
            if region is None or not region.holdout_scores:
                continue

            est = estimate_survival(
                holdout_noise=region.holdout_noise,
                context_instability=region.region_context_stability,
                in_sample_score=region.region_score,
                holdout_mean=region.holdout_mean,
                region_noise=region.region_noise,
            )
            report = result["confidence_report"]
            budget_used = report.total_objective_calls or (
                report.search_evaluations + report.holdout_evaluations)
            outcome = _judge(problem, "N", result["recommended_center"], budget_used,
                             max(report.number_of_tested_points, 1), sc, hc, 2)
            rows.append({
                "problem": pname, "seed": seed,
                "context_instability": est.signals["context_instability"],
                "holdout_noise": est.signals["holdout_noise"],
                "effective_noise": est.signals["effective_noise"],
                "decay_gap": est.signals["decay_gap"],
                "z_instability": est.signals["z_instability"],
                "z_noise": est.signals["z_noise"],
                "z_decay": est.signals["z_decay"],
                "p_survive": est.p_survive,
                "fragility": est.fragility_index,
                "survived": bool(outcome.survives),
            })
            n_here += 1
        if verbose:
            done = [r for r in rows if r["problem"] == pname]
            surv = sum(r["survived"] for r in done)
            print(f"  {pname:<20s} {n_here:>3d} runs, {surv} survived")
    return rows


def _reliability_curve(pairs: Sequence[Tuple[float, bool]], n_bins: int
                       ) -> Tuple[List[dict], float]:
    bins: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, o in pairs:
        bins[min(n_bins - 1, int(p * n_bins))].append((p, o))
    total = len(pairs)
    out, ece = [], 0.0
    for i, b in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not b:
            out.append({"range": (lo, hi), "n": 0})
            continue
        pred = sum(p for p, _ in b) / len(b)
        obs = sum(1 for _, o in b if o) / len(b)
        out.append({"range": (lo, hi), "n": len(b), "predicted": pred, "observed": obs})
        ece += (len(b) / total) * abs(pred - obs)
    return out, ece


def _print_curve(rows: List[dict], ece: float) -> None:
    print(f"\n  RELIABILITY  (ECE = {ece:.3f}, lower is better)")
    print(f"    {'bin':>10s} {'n':>4s} {'predicted':>10s} {'observed':>9s}  reliability")
    for r in rows:
        if r["n"] == 0:
            continue
        lo, hi = r["range"]
        bar = int(round(r["observed"] * 20))
        gap = r["predicted"] - r["observed"]
        flag = "  overconfident" if gap > 0.15 else ("  underconfident" if gap < -0.15 else "")
        print(f"    {lo:.1f}-{hi:.1f} {r['n']:>4d} {r['predicted']:>10.2f} "
              f"{r['observed']:>9.2f}  {'#'*bar:<20s}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--fit", action="store_true", help="fit and print coefficients")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print("=" * 74)
    print(f"p_survive RELIABILITY  ({args.seeds} seeds x {len(_LANDSCAPES)} landscapes)")
    print("=" * 74)
    rows = _collect(args.seeds, verbose=True)
    if not rows:
        print("no usable runs")
        return 1

    total = len(rows)
    surv = sum(r["survived"] for r in rows)
    print(f"\n  {total} runs, {surv} survived ({surv/total:.0%} base rate)")

    if args.fit:
        coeffs = fit_logistic(rows)
        print("\n  FITTED COEFFICIENTS -- paste into narrs/psurvive._DEFAULT_COEFFS:")
        print("    _DEFAULT_COEFFS = {")
        print(f"        \"intercept\": {coeffs['intercept']:.3f},")
        print(f"        \"context_instability\": {coeffs['context_instability']:.3f},")
        print(f"        \"effective_noise\": {coeffs['effective_noise']:.3f},")
        print(f"        \"decay_gap\": {coeffs['decay_gap']:.3f},")
        print("    }")
        # evaluate the freshly fitted model in-sample as a sanity check
        from narrs.psurvive import _sigmoid
        pairs = [(_sigmoid(coeffs["intercept"] + coeffs["context_instability"] * r["context_instability"]
                           + coeffs["effective_noise"] * r["effective_noise"] + coeffs["decay_gap"] * r["decay_gap"]),
                  r["survived"]) for r in rows]
        _, ece = _reliability_curve(pairs, args.bins)
        print(f"\n  in-sample ECE of the fitted model: {ece:.3f}")
    else:
        curve, ece = _reliability_curve([(r["p_survive"], r["survived"]) for r in rows], args.bins)
        _print_curve(curve, ece)
        print("\n" + "=" * 74)
        verdict = "well calibrated" if ece < 0.10 else (
            "acceptable" if ece < 0.20 else "NEEDS REFIT (run with --fit)")
        print(f"VERDICT: {verdict}  (ECE {ece:.3f})")
        print("=" * 74)

    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows}, indent=1))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
