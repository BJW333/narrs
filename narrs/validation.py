from __future__ import annotations

import warnings
from typing import Dict, Sequence

from .types import MetricRule, ParameterSpec, SearchContext


class NARRSConfigurationError(ValueError):
    """
    Raised when a NARRS run is set up in a way that would silently produce
    misleading results (a typo'd metric weight, an inverted range, empty
    required inputs, and so on). The goal is to fail loudly at construction
    rather than return a confident-looking but wrong result.
    """


def validate_setup(
    parameter_space: Sequence[ParameterSpec],
    search_contexts: Sequence[SearchContext],
    holdout_contexts: Sequence[SearchContext],
    metric_rules: Dict[str, MetricRule],
    metric_weights: Dict[str, float],
) -> None:
    """
    Check a NARRS setup for configuration mistakes that would otherwise run
    silently. Raises NARRSConfigurationError on hard problems and issues a
    warning on softer ones. Called from NoiseAwareRobustRegionSearch.__init__;
    pass validate=False there to skip it.
    """
    errors = []

    # Required, non-empty inputs.
    if not parameter_space:
        errors.append("parameter_space is empty; at least one ParameterSpec is required.")
    if not search_contexts:
        errors.append("search_contexts is empty; at least one SearchContext is required.")
    if not metric_rules:
        errors.append("metric_rules is empty; at least one MetricRule is required.")
    if not metric_weights:
        errors.append("metric_weights is empty; at least one weighted metric is required.")

    # Parameter specs: duplicate names, inverted/degenerate ranges, bad weights.
    seen_names = set()
    for spec in parameter_space:
        if spec.name in seen_names:
            errors.append(f"Duplicate parameter name '{spec.name}'.")
        seen_names.add(spec.name)
        if spec.min_value > spec.max_value:
            errors.append(
                f"Parameter '{spec.name}' has min_value {spec.min_value} > max_value "
                f"{spec.max_value} (inverted range)."
            )
        elif spec.min_value == spec.max_value:
            errors.append(
                f"Parameter '{spec.name}' has min_value == max_value ({spec.min_value}); "
                f"the range is degenerate (nothing to search)."
            )
        if spec.weight < 0:
            errors.append(f"Parameter '{spec.name}' has negative weight {spec.weight}.")

    # Metric weights: every key must match a rule, or that term is silently dropped.
    for name in metric_weights:
        if name not in metric_rules:
            errors.append(
                f"metric_weights has key '{name}' with no matching MetricRule; that term "
                f"is silently dropped from the score. Known rules: {sorted(metric_rules)}."
            )
    for name, weight in metric_weights.items():
        if weight < 0:
            errors.append(f"metric_weights['{name}'] is negative ({weight}).")

    if errors:
        raise NARRSConfigurationError(
            "Invalid NARRS setup:\n  - " + "\n  - ".join(errors)
        )

    # Softer issues: warn but do not block.
    if not holdout_contexts:
        warnings.warn(
            "No holdout_contexts provided; confidence cannot be out-of-sample "
            "validated and will be capped accordingly.",
            RuntimeWarning,
            stacklevel=2,
        )
    dead = [
        name
        for name, rule in metric_rules.items()
        if name not in metric_weights and rule.hard_min is None and rule.hard_max is None
    ]
    if dead:
        warnings.warn(
            f"metric_rules define metrics with no weight and no hard rule: {sorted(dead)}. "
            f"They will not affect the score or reject anything.",
            RuntimeWarning,
            stacklevel=2,
        )