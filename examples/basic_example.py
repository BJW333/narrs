"""
Basic example of Noise-Aware Robust Region Search.

Run:

    python examples/basic_example.py

This example optimizes a noisy 2D objective with a broad stable plateau.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narrs import (
    NoiseAwareRobustRegionSearch,
    NARRSConfig,
    ParameterSpec,
    MetricRule,
    ObjectiveResult,
    SearchContext,
)

def objective(params, context, random_seed):
    import random
    context_offsets = {
        "normal": 100,
        "hard_context": 200,
        "alternate": 300,
        "holdout_1": 400,
        "holdout_2": 500,
    }

    rng = random.Random(random_seed + context_offsets.get(context.name, 0))

    x = params["x"]
    y = params["y"]

    # Stable plateau centered around x=2, y=-1.
    distance = ((x - 2.0) ** 2 + 0.5 * (y + 1.0) ** 2)

    # Base score in 0-1-ish space.
    score = max(0.0, 1.0 - 0.12 * distance)

    # Context effect.
    if context.name == "hard_context":
        score -= 0.05

    # Noise.
    score += rng.gauss(0, 0.03)

    return ObjectiveResult(
        performance_metrics={"score": score},
        sample_count=1,
        validity_status="valid",
    )


if __name__ == "__main__":
    parameter_space = [
        ParameterSpec("x", -5, 5, weight=1.0),
        ParameterSpec("y", -5, 5, weight=1.5),
    ]

    search_contexts = [
        SearchContext("normal"),
        SearchContext("hard_context"),
        SearchContext("alternate"),
        SearchContext("late_regime"),
        SearchContext("stress"),
    ]

    holdout_contexts = [
        SearchContext("holdout_1"),
        SearchContext("holdout_2"),
    ]

    metric_rules = {
        "score": MetricRule(good_value=1.0, bad_value=0.0, higher_is_better=True)
    }

    metric_weights = {"score": 1.0}

    config = NARRSConfig(
        search_rounds=3,
        initial_grid_density=9,
        total_compute_budget=12000,
        minimum_region_size=3,
        minimum_region_width=0.03,
        minimum_sample_size=1,
        maximum_allowed_noise=0.50,
        maximum_confidence_interval_width=1.0,
        minimum_acceptable_worst_case_score=0.0,
        # Always set the bar: scores are normalised 0..1, so 0.5 is the
        # midpoint of your own metric range. Without it, "high" only means
        # STABLE, not GOOD (see "Stable is not the same as good" in the README).
        minimum_acceptable_holdout_score=0.5,
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

    print("\nRecommended center:")
    print(result["recommended_center"])

    print("\nBest region ranges:")
    print(result["confidence_report"].best_region_parameter_ranges)

    print("\nConfidence rating:")
    print(result["confidence_report"].final_confidence_rating)

    print("\np_survive (sharp fragility warning; the rating above carries the calibrated claim):")
    print(round(result["confidence_report"].p_survive, 2))

    print("\nWarnings:")
    print(result["confidence_report"].warning_signs)
