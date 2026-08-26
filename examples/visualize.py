"""
Visualize the CONFIDENCE side of NARRS -- trust, fragility, noise, decay.
(For the search/landscape story -- what was explored and why the plateau won --
use examples/visualize_narrs.py; the two are complementary.)

Saves PNGs you can put in a README or an audit.

    python examples/visualize.py                 # writes to ./narrs_figures/
    python examples/visualize.py --out /some/dir

Four figures, each answering a question a skeptic would actually ask:

  1. landscape.png    -- WHERE does it search and what does it pick? The 2-D
     objective as a heatmap, every evaluated point, and the recommended region
     drawn on top. Shows it avoiding the fragile spike and settling on the
     stable plateau.

  2. calibration.png  -- p_survive versus realised survival across ~150 runs.
     NOTE: the installed fit is deliberately SHARP (failures weighted up), so
     expect points BELOW the diagonal in the mid-range -- it under-promises on
     survivors to make fragile regions actually read as fragile. When it says 0.7,
     about 70% survive.

  3. noise_vs_survival.png -- WHAT drives failure? Each run plotted by its
     out-of-sample noise, coloured by whether it survived. The single clearest
     predictor: survivors cluster at low noise, failures at high.

  4. decay.png        -- what does a GOOD vs BAD region look like across
     contexts? In-sample score versus per-context out-of-sample score for a
     robust region and a fragile one, side by side.

Needs matplotlib (`pip install matplotlib`). Everything else is the package
itself. Figures 2 and 3 re-run a small survival sweep, so allow a few minutes.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
except ImportError:
    print("This script needs matplotlib:  pip install matplotlib")
    sys.exit(1)

from narrs import (MetricRule, NARRSConfig, NoiseAwareRobustRegionSearch,
                   ObjectiveResult, ParameterSpec, SearchContext)
from narrs.psurvive import estimate_survival
from narrs.benchmarks.battery import _judge
from narrs.benchmarks.scalable import ScalableDecoy, ScalablePlateauSpike

# a quiet, legible style -- no gradients, no chartjunk
INK = "#1a1a2e"
PLATEAU_C = "#2a9d8f"
SPIKE_C = "#e76f51"
GRID = "#d9d9e3"


# --------------------------------------------------------------------------- #
# figure 1: the landscape and what NARRS picked                               #
# --------------------------------------------------------------------------- #
_PLATEAU = (0.30, 0.30)
_SPIKE = (0.72, 0.72)
_CRASH = {"regime_2"}


def _toy_objective(params, context, rep):
    x, y = params["x"], params["y"]
    dp = math.dist((x, y), _PLATEAU)
    plateau = 0.80 * math.exp(-0.5 * (dp / 0.18) ** 2)
    ds = math.dist((x, y), _SPIKE)
    payoff = -2.0 if context.name in _CRASH else 2.5
    spike = payoff * math.exp(-0.5 * (ds / 0.12) ** 2)
    import random, zlib
    key = f"{context.name}|{rep}|{x:.4f}|{y:.4f}".encode()   # NOT hash(): see how_it_works.py
    rng = random.Random(zlib.crc32(key))
    return ObjectiveResult(performance_metrics={"score": plateau + spike + rng.gauss(0, 0.03)},
                           sample_count=1)


def _mean_field(x, y):
    """Average objective over contexts (what a mean-optimizer sees)."""
    total = 0.0
    for cname in ("regime_0", "regime_1", "regime_2", "regime_3"):
        c = SearchContext(cname)
        total += _toy_objective({"x": x, "y": y}, c, 0).performance_metrics["score"]
    return total / 4


def fig_landscape(out: Path) -> None:
    contexts = [SearchContext(f"regime_{i}") for i in range(4)]
    holdout = [SearchContext(f"oos_{i}") for i in range(6)]
    opt = NoiseAwareRobustRegionSearch(
        objective_function=_toy_objective,
        parameter_space=[ParameterSpec("x", 0.0, 1.0), ParameterSpec("y", 0.0, 1.0)],
        search_contexts=contexts, holdout_contexts=holdout,
        metric_rules={"score": MetricRule(good_value=2.5, bad_value=-2.0)},
        metric_weights={"score": 1.0},
        config=NARRSConfig(search_rounds=3, initial_candidate_count=40,
                           repetitions_per_point=3, total_compute_budget=6000,
                           random_seed=1, context_aggregator="cvar"))
    result = opt.run()

    n = 120
    xs = [i / (n - 1) for i in range(n)]
    field = [[_mean_field(x, y) for x in xs] for y in xs]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                   cmap="RdYlGn", vmin=-1, vmax=1.5, aspect="auto", alpha=0.9)
    fig.colorbar(im, ax=ax, label="mean score across contexts", shrink=0.85)

    # every evaluated point
    pts = [(r.parameter_point["x"], r.parameter_point["y"]) for r in opt.all_point_results]
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=9, c=INK,
                   alpha=0.35, linewidths=0, label=f"evaluated ({len(pts)})")

    # ground-truth markers
    ax.scatter(*_PLATEAU, marker="*", s=380, c=PLATEAU_C, edgecolor="white",
               linewidth=1.5, zorder=5, label="plateau (stable, correct)")
    ax.scatter(*_SPIKE, marker="X", s=240, c=SPIKE_C, edgecolor="white",
               linewidth=1.5, zorder=5, label="spike (fragile trap)")

    center = result["recommended_center"]
    region = result["best_region"]
    if center and region:
        w = max(region.region_width, 0.03)
        ax.add_patch(Ellipse((center["x"], center["y"]), 2 * w, 2 * w,
                             fill=False, edgecolor=INK, linewidth=2.5, zorder=6))
        ax.scatter(center["x"], center["y"], marker="o", s=90, c="white",
                   edgecolor=INK, linewidth=2, zorder=7, label="NARRS recommendation")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Where NARRS searches and what it picks\n"
                 "green = high mean, but the spike (X) crashes in one regime",
                 fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "landscape.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out/'landscape.png'}")


# --------------------------------------------------------------------------- #
# survival sweep shared by figures 2 & 3                                       #
# --------------------------------------------------------------------------- #
def _survival_sweep(seeds: int):
    land = [
        ("PlateauSpike", lambda: ScalablePlateauSpike()),
        ("PS/noisy", lambda: ScalablePlateauSpike(noise_sd=0.45)),
        ("Decoy", lambda: ScalableDecoy()),
        ("Decoy/thin", lambda: ScalableDecoy(n_search_contexts=3)),
        ("Decoy/hard", lambda: ScalableDecoy(n_decoys=90, decoy_amplitude_sd=3.2)),
        ("Decoy/thin+hard", lambda: ScalableDecoy(n_search_contexts=2, n_decoys=90,
                                                  decoy_amplitude_sd=3.2)),
    ]
    rows = []
    for _, make in land:
        for seed in range(seeds):
            problem = make()
            sc, hc = problem.search_contexts, problem.holdout_contexts
            budget = 5000 + 1200 * (len(problem.parameter_space) - 2)
            opt = NoiseAwareRobustRegionSearch(
                objective_function=problem.objective,
                parameter_space=problem.parameter_space,
                search_contexts=sc, holdout_contexts=hc,
                metric_rules=problem.metric_rules,
                metric_weights=problem.metric_weights,
                config=NARRSConfig(search_rounds=3, initial_candidate_count=40,
                                   repetitions_per_point=2, total_compute_budget=budget,
                                   random_seed=seed))
            result = opt.run()
            region = result.get("best_region")
            if region is None or not region.holdout_scores:
                continue
            est = estimate_survival(region.holdout_noise, region.region_context_stability,
                                    region.region_score, region.holdout_mean)
            report = result["confidence_report"]
            used = report.total_objective_calls or (report.search_evaluations + report.holdout_evaluations)
            outcome = _judge(problem, "N", result["recommended_center"], used,
                             max(report.number_of_tested_points, 1), sc, hc, 2)
            rows.append({"p": est.p_survive, "noise": region.holdout_noise,
                         "survived": outcome.survives})
    return rows


def fig_calibration(out: Path, rows) -> None:
    n_bins = 8
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        bins[min(n_bins - 1, int(r["p"] * n_bins))].append(r["survived"])

    xs, ys, sizes = [], [], []
    ece = 0.0
    for i, b in enumerate(bins):
        if not b:
            continue
        pred = (i + 0.5) / n_bins
        obs = sum(b) / len(b)
        xs.append(pred); ys.append(obs); sizes.append(len(b))
        ece += (len(b) / len(rows)) * abs(pred - obs)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1.5, label="perfect calibration")
    ax.scatter(xs, ys, s=[20 + 8 * n for n in sizes], c=PLATEAU_C,
               edgecolor=INK, linewidth=1, alpha=0.85, zorder=5)
    for x, y, n in zip(xs, ys, sizes):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("p_survive  (what NARRS predicts)")
    ax.set_ylabel("actual survival rate")
    ax.set_title(f"Is the probability honest?\n"
                 f"dashed line = calibrated; sharp fit sits below it mid-range by design   (ECE = {ece:.3f})", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out/'calibration.png'}  (ECE {ece:.3f})")


def fig_noise_vs_survival(out: Path, rows) -> None:
    surv = [r for r in rows if r["survived"]]
    fail = [r for r in rows if not r["survived"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter([r["noise"] for r in surv], [r["p"] for r in surv], s=45,
               c=PLATEAU_C, edgecolor="white", linewidth=0.6, alpha=0.85,
               label=f"survived ({len(surv)})")
    ax.scatter([r["noise"] for r in fail], [r["p"] for r in fail], s=55,
               c=SPIKE_C, edgecolor="white", linewidth=0.6, alpha=0.9,
               marker="X", label=f"failed ({len(fail)})")
    ax.set_xlabel("out-of-sample noise of the chosen region")
    ax.set_ylabel("p_survive")
    ax.set_title("Noise is what predicts failure\n"
                 "failures (orange X) cluster at high out-of-sample noise", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "noise_vs_survival.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out/'noise_vs_survival.png'}")


def fig_decay(out: Path) -> None:
    """A robust region vs a fragile one, scored across contexts."""
    robust = ScalableDecoy()
    fragile = ScalableDecoy(n_search_contexts=2, n_decoys=90, decoy_amplitude_sd=3.2)

    def run(problem, seed):
        sc, hc = problem.search_contexts, problem.holdout_contexts
        opt = NoiseAwareRobustRegionSearch(
            objective_function=problem.objective, parameter_space=problem.parameter_space,
            search_contexts=sc, holdout_contexts=hc, metric_rules=problem.metric_rules,
            metric_weights=problem.metric_weights,
            config=NARRSConfig(search_rounds=3, initial_candidate_count=40,
                               repetitions_per_point=2, total_compute_budget=5000,
                               random_seed=seed))
        r = opt.run()
        reg = r["best_region"]
        return reg

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (reg, title, color) in zip(
            axes, [(run(robust, 0), "robust region", PLATEAU_C),
                   (run(fragile, 3), "fragile region", SPIKE_C)]):
        if reg is None or not reg.holdout_scores:
            ax.text(0.5, 0.5, "no region", ha="center")
            continue
        scores = reg.holdout_scores
        ax.axhline(reg.region_score, color=INK, linestyle="--", linewidth=1.3,
                   label=f"in-sample score {reg.region_score:.2f}")
        ax.plot(range(len(scores)), scores, "o-", color=color, linewidth=1.5,
                label=f"per-context OOS (noise {reg.holdout_noise:.2f})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("held-out context")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel("score")
    fig.suptitle("A robust region holds across contexts; a fragile one scatters", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "decay.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out/'decay.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="narrs_figures")
    ap.add_argument("--seeds", type=int, default=25, help="seeds per landscape for the sweep figures")
    ap.add_argument("--skip-sweep", action="store_true", help="only the landscape + decay figures (fast)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {out}/")

    fig_landscape(out)
    fig_decay(out)
    if not args.skip_sweep:
        print("  running survival sweep for calibration figures (a few minutes)...")
        rows = _survival_sweep(args.seeds)
        fig_calibration(out, rows)
        fig_noise_vs_survival(out, rows)

    print("done.")


if __name__ == "__main__":
    main()
