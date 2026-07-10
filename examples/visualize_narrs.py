"""
NARRS visualizer, refined version.

Run from inside your narrs_package folder:

    python3.10 visualize_narrs_refined.py

This version is designed to be easier to understand than the previous plots.

It separates the pictures into:

    FIGURE 1: FINAL ANSWER / WHAT THE ALGORITHM CHOSE
        This is the main plot. It shows the chosen robust region and the
        recommended center. Use this figure to judge the final result.

    FIGURE 2: WHY THE ALGORITHM PREFERS THE PLATEAU
        This compares the broad stable plateau against the sharp fake spike.
        It explains why one tiny high point is less trustworthy than a region.

    FIGURE 3: SEARCH PROCESS ONLY / NOT FINAL GOOD-BAD LABELS
        This shows where each search round tested points. Round colors only
        mean WHEN the point was tested, not whether it was accepted.

    FIGURE 4: ROBUSTNESS SCORE MAP
        This colors every tested point by robust score after penalties for
        noise, instability, weak neighbors, and sharp spikes.

The toy objective has:
    - broad stable plateau near (x=2, y=-1)
    - sharp fake spike near (x=-3, y=3)

The algorithm should prefer the broad stable plateau.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Make imports work whether this file is run from:
#   narrs_package/
#   narrs_package/examples/
#   somewhere else with narrs_package as parent
THIS_FILE = Path(__file__).resolve()
POSSIBLE_PACKAGE_ROOTS = [
    THIS_FILE.parent,
    THIS_FILE.parent.parent,
    Path.cwd(),
]

for root in POSSIBLE_PACKAGE_ROOTS:
    if (root / "narrs").exists():
        sys.path.insert(0, str(root))
        break

from narrs import (
    NoiseAwareRobustRegionSearch,
    NARRSConfig,
    ParameterSpec,
    MetricRule,
    ObjectiveResult,
    SearchContext,
)


# ---------------------------------------------------------------------
# 1. Fixed deterministic context seeds
# ---------------------------------------------------------------------
# Do not use Python's hash(context.name), because hash randomization can
# make results change across different Python runs.

CONTEXT_OFFSETS = {
    "normal": 100,
    "hard": 200,
    "shifted": 300,
    "holdout_1": 400,
    "holdout_2": 500,
}


# ---------------------------------------------------------------------
# 2. Fake objective
# ---------------------------------------------------------------------

def broad_plateau_score(x: float, y: float) -> float:
    """
    Broad stable hill centered around x=2, y=-1.

    This is the area the algorithm should prefer.
    """
    plateau_distance = ((x - 2.0) ** 2 + 0.8 * (y + 1.0) ** 2)
    return max(0.0, 1.0 - 0.10 * plateau_distance)


def sharp_spike_score(x: float, y: float) -> float:
    """
    Sharp fake spike centered around x=-3, y=3.

    This can look great at one point but falls apart nearby.
    """
    spike_distance = ((x + 3.0) ** 2 + (y - 3.0) ** 2)
    return max(0.0, 1.2 - 2.5 * spike_distance)


def true_surface_score(x: float, y: float) -> float:
    """
    Clean, noise-free score surface used only for plotting.
    The optimizer does not get to see this clean surface directly.
    """
    return max(broad_plateau_score(x, y), sharp_spike_score(x, y))


def objective(params, context, random_seed):
    """
    Noisy objective function passed into NARRS.

    In a real use case, this function would run your backtest, ML training,
    simulation, pricing test, etc.
    """
    context_offset = CONTEXT_OFFSETS.get(context.name, 999)
    rng = random.Random(random_seed + context_offset)

    x = params["x"]
    y = params["y"]

    clean_score = true_surface_score(x, y)

    context_penalty = {
        "normal": 0.00,
        "hard": 0.08,
        "shifted": 0.04,
        "holdout_1": 0.03,
        "holdout_2": 0.06,
    }.get(context.name, 0.0)

    noise = rng.gauss(0, 0.04)

    raw_score = clean_score - context_penalty + noise
    raw_score = max(0.0, min(1.0, raw_score))

    return ObjectiveResult(
        performance_metrics={"score": raw_score},
        sample_count=1,
        validity_status="valid",
    )


# ---------------------------------------------------------------------
# 3. Optimizer setup
# ---------------------------------------------------------------------

parameter_space = [
    ParameterSpec("x", -5, 5, weight=1.0),
    ParameterSpec("y", -5, 5, weight=1.0),
]

search_contexts = [
    SearchContext("normal"),
    SearchContext("hard"),
    SearchContext("shifted"),
]

holdout_contexts = [
    SearchContext("holdout_1"),
    SearchContext("holdout_2"),
]

metric_rules = {
    "score": MetricRule(
        good_value=1.0,
        bad_value=0.0,
        higher_is_better=True,
    )
}

metric_weights = {"score": 1.0}

config = NARRSConfig(
    search_rounds=3,
    initial_grid_density=7,

    repetitions_per_point=4,
    max_extra_repetitions=8,
    max_total_repetitions_per_point=12,

    exploration_fraction=0.20,
    total_compute_budget=9000,

    minimum_region_size=5,
    minimum_region_width=0.05,
    minimum_region_density=1.0,

    minimum_sample_size=1,
    maximum_allowed_noise=0.50,
    maximum_confidence_interval_width=1.0,

    minimum_acceptable_worst_case_score=0.20,
    minimum_acceptable_holdout_score=0.20,

    minimum_neighborhood_mean=0.35,
    maximum_neighborhood_noise=1.0,
    maximum_spike_penalty=1.0,

    neighbor_radius_multiplier=1.50,

    holdout_points_per_region=8,
    
    random_seed=42,
    verbose=True,
)


optimizer = NoiseAwareRobustRegionSearch(
    objective_function=objective,
    parameter_space=parameter_space,
    search_contexts=search_contexts,
    holdout_contexts=holdout_contexts,
    metric_rules=metric_rules,
    metric_weights=metric_weights,
    config=config,
)

result = optimizer.run()

best_region = result["best_region"]
recommended_center = result["recommended_center"]
report = result["confidence_report"]
all_points = optimizer.all_point_results

rejected_points = [p for p in all_points if p.rejected]
surviving_points = [p for p in all_points if not p.rejected]
best_region_points = best_region.points if best_region is not None else []


# ---------------------------------------------------------------------
# 4. Terminal explanation
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("NARRS VISUALIZATION: READ THIS FIRST")
print("=" * 78)
print("Fake objective design:")
print("  - Broad stable plateau near (x=2, y=-1)")
print("  - Sharp fake spike near (x=-3, y=3)")
print()
print("The algorithm should NOT chase the highest single point.")
print("It should find a REGION where nearby values also work.")
print()
print("Most important outputs:")
print("  1. Recommended center = one safe representative parameter setting")
print("  2. Best region ranges = the stable parameter zone around that center")
print("  3. Figure 1 = the main final-decision plot")
print()
print("Final result:")
print("  Recommended center:", recommended_center)
print("  Best region ranges:", report.best_region_parameter_ranges)
print("  Confidence rating:", report.final_confidence_rating)
print()
print("Counts:")
print("  Tested points:", len(all_points))
print("  Rejected / unstable points:", len(rejected_points))
print("  Surviving stable points:", len(surviving_points))
print("  Points inside final best region:", len(best_region_points))
print()
print("Warning signs:")
if report.warning_signs:
    for warning in report.warning_signs:
        print("  -", warning)
else:
    print("  - None")
print("=" * 78 + "\n")


# ---------------------------------------------------------------------
# 5. Plot helpers
# ---------------------------------------------------------------------

def make_surface_grid(x_min=-5, x_max=5, y_min=-5, y_max=5, steps=180):
    xs = [x_min + (x_max - x_min) * i / (steps - 1) for i in range(steps)]
    ys = [y_min + (y_max - y_min) * i / (steps - 1) for i in range(steps)]

    z = []
    plateau_z = []
    spike_z = []

    for y in ys:
        z_row = []
        p_row = []
        s_row = []
        for x in xs:
            z_row.append(true_surface_score(x, y))
            p_row.append(broad_plateau_score(x, y))
            s_row.append(sharp_spike_score(x, y))
        z.append(z_row)
        plateau_z.append(p_row)
        spike_z.append(s_row)

    return xs, ys, z, plateau_z, spike_z


def get_best_region_box():
    if best_region is None or not report.best_region_parameter_ranges:
        return None

    ranges = report.best_region_parameter_ranges
    x0, x1 = ranges["x"]
    y0, y1 = ranges["y"]

    # Make very thin regions visible on the plot.
    min_visible_width = 0.18
    min_visible_height = 0.18

    width = max(x1 - x0, min_visible_width)
    height = max(y1 - y0, min_visible_height)

    if x1 - x0 < min_visible_width:
        cx = (x0 + x1) / 2
        x0 = cx - width / 2

    if y1 - y0 < min_visible_height:
        cy = (y0 + y1) / 2
        y0 = cy - height / 2

    return x0, y0, width, height


def draw_best_region_box(ax):
    box = get_best_region_box()
    if box is None:
        return

    x0, y0, width, height = box

    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fill=False,
        linewidth=3,
        linestyle="--",
        edgecolor="black",
        label="Final robust region boundary",
    )
    ax.add_patch(rect)


def draw_decision_summary(ax):
    if best_region is None:
        summary = "No robust region survived."
    else:
        ranges = report.best_region_parameter_ranges
        x_range = ranges.get("x", ("?", "?"))
        y_range = ranges.get("y", ("?", "?"))
        summary = (
            "FINAL ANSWER\n"
            f"recommended center: x={recommended_center['x']:.3f}, y={recommended_center['y']:.3f}\n"
            f"robust x range: {x_range[0]:.3f} to {x_range[1]:.3f}\n"
            f"robust y range: {y_range[0]:.3f} to {y_range[1]:.3f}\n"
            f"confidence: {report.final_confidence_rating}\n\n"
            "How to read this plot:\n"
            "star = parameter setting to use\n"
            "black box = robust zone\n"
            "hollow circles = final-region evidence"
        )

    ax.text(
        0.02,
        0.02,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", alpha=0.85),
    )


def draw_known_centers(ax):
    ax.scatter([2], [-1], s=180, marker="P", label="True plateau center in toy demo")
    ax.scatter([-3], [3], s=180, marker="D", label="Sharp spike center in toy demo")

    ax.annotate(
        "Broad stable plateau\nwide area = trustworthy",
        xy=(2, -1),
        xytext=(2.65, -2.35),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )

    ax.annotate(
        "Sharp spike\none tiny area = suspicious",
        xy=(-3, 3),
        xytext=(-4.85, 2.05),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )


xs, ys, z, plateau_z, spike_z = make_surface_grid()


# ---------------------------------------------------------------------
# FIGURE 1: Final decision view
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 9))

heat = ax.imshow(
    z,
    origin="lower",
    extent=[-5, 5, -5, 5],
    aspect="auto",
    alpha=0.32,
)
plt.colorbar(heat, ax=ax, label="Clean fake objective score")

# Rejected points.
if rejected_points:
    ax.scatter(
        [p.parameter_point["x"] for p in rejected_points],
        [p.parameter_point["y"] for p in rejected_points],
        marker="x",
        s=55,
        alpha=0.35,
        label="Rejected points: failed noise/stability/region checks",
    )

# Surviving points that are NOT in the final best region.
best_region_ids = {id(p) for p in best_region_points}
surviving_not_best = [p for p in surviving_points if id(p) not in best_region_ids]

if surviving_not_best:
    ax.scatter(
        [p.parameter_point["x"] for p in surviving_not_best],
        [p.parameter_point["y"] for p in surviving_not_best],
        s=45,
        alpha=0.45,
        label="Passed checks, but not chosen final region",
    )

# Final best-region points.
if best_region_points:
    ax.scatter(
        [p.parameter_point["x"] for p in best_region_points],
        [p.parameter_point["y"] for p in best_region_points],
        s=170,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Final-region points: evidence for chosen robust zone",
    )

# Best-region box.
draw_best_region_box(ax)

# Recommended center.
if recommended_center:
    ax.scatter(
        [recommended_center["x"]],
        [recommended_center["y"]],
        s=450,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label="Recommended center: safe setting to use",
        zorder=10,
    )

draw_known_centers(ax)
draw_decision_summary(ax)

ax.set_title("FIGURE 1 — FINAL ANSWER: Chosen Robust Region + Recommended Center")
ax.set_xlabel("Parameter x (toy parameter 1)")
ax.set_ylabel("Parameter y (toy parameter 2)")
ax.grid(True)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# FIGURE 2: Why broad plateau beats sharp spike
# ---------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
heat1 = ax.imshow(
    plateau_z,
    origin="lower",
    extent=[-5, 5, -5, 5],
    aspect="auto",
    alpha=0.75,
)
plt.colorbar(heat1, ax=ax, label="Plateau score")
ax.scatter([2], [-1], s=220, marker="P", label="Broad plateau center")
draw_best_region_box(ax)
if recommended_center:
    ax.scatter([recommended_center["x"]], [recommended_center["y"]], s=350, marker="*", label="Recommended center: safe setting to use")
ax.set_title("FIGURE 2A — Broad Plateau: Many Nearby Values Work")
ax.set_xlabel("Parameter x (toy parameter 1)")
ax.set_ylabel("Parameter y (toy parameter 2)")
ax.grid(True)
ax.legend()

ax = axes[1]
heat2 = ax.imshow(
    spike_z,
    origin="lower",
    extent=[-5, 5, -5, 5],
    aspect="auto",
    alpha=0.75,
)
plt.colorbar(heat2, ax=ax, label="Spike score")
ax.scatter([-3], [3], s=220, marker="D", label="Sharp spike center")
draw_best_region_box(ax)
if recommended_center:
    ax.scatter([recommended_center["x"]], [recommended_center["y"]], s=350, marker="*", label="Recommended center: safe setting to use")
ax.set_title("FIGURE 2B — Sharp Spike: One Tiny Area Looks Good")
ax.set_xlabel("Parameter x (toy parameter 1)")
ax.set_ylabel("Parameter y (toy parameter 2)")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# FIGURE 3: Coarse-to-fine search rounds
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 9))

for round_index in sorted(set(p.round_index for p in all_points)):
    pts = [p for p in all_points if p.round_index == round_index]
    ax.scatter(
        [p.parameter_point["x"] for p in pts],
        [p.parameter_point["y"] for p in pts],
        s=55,
        alpha=0.55,
        label=f"Round {round_index + 1}: tested points only",
    )

draw_best_region_box(ax)

if recommended_center:
    ax.scatter(
        [recommended_center["x"]],
        [recommended_center["y"]],
        s=450,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label="Recommended center: safe setting to use",
        zorder=10,
    )

ax.text(
    0.02,
    0.02,
    "IMPORTANT FOR THIS FIGURE:\nColors mean SEARCH ROUND ONLY.\nThey do NOT mean good/bad.\nFinal decision = black box + star.",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", alpha=0.85),
)

ax.set_title("FIGURE 3 — SEARCH PROCESS ONLY: Coarse-to-Fine Tested Points")
ax.set_xlabel("Parameter x (toy parameter 1)")
ax.set_ylabel("Parameter y (toy parameter 2)")
ax.grid(True)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# FIGURE 4: Robust score view
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 9))

if all_points:
    robust_scores = [p.robust_score for p in all_points]
    scatter = ax.scatter(
        [p.parameter_point["x"] for p in all_points],
        [p.parameter_point["y"] for p in all_points],
        c=robust_scores,
        s=75,
        alpha=0.85,
    )
    plt.colorbar(scatter, ax=ax, label="Robust score: higher = more stable after penalties")

draw_best_region_box(ax)

if recommended_center:
    ax.scatter(
        [recommended_center["x"]],
        [recommended_center["y"]],
        s=450,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label="Recommended center: safe setting to use",
        zorder=10,
    )

ax.text(
    0.02,
    0.02,
    "Robust score includes:\nperformance + neighbor stability\n- noise - context instability\n- spike/fragility penalties",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", alpha=0.85),
)

ax.set_title("FIGURE 4 — ROBUSTNESS SCORE MAP: Higher Color = More Trustworthy")
ax.set_xlabel("Parameter x (toy parameter 1)")
ax.set_ylabel("Parameter y (toy parameter 2)")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
