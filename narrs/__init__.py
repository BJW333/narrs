"""
NARRS: Noise-Aware Robust Region Search

A universal optimization/search package that finds stable high-performing
parameter regions instead of isolated best points.

Main import:

    from narrs import (
        NoiseAwareRobustRegionSearch,
        NARRSConfig,
        ParameterSpec,
        MetricRule,
        ObjectiveResult,
    )
"""

from .core import NoiseAwareRobustRegionSearch
from .types import (
    ParameterSpec,
    MetricRule,
    ObjectiveResult,
    NARRSConfig,
    SearchContext,
    Region,
    PointResult,
    ConfidenceReport,
)
from .samplers import (
    BaseSampler,
    GridSampler,
    RandomSampler,
    LatinHypercubeSampler,
    HybridSampler,
    CustomSampler,
)
from .validation import NARRSConfigurationError, validate_setup

__all__ = [
    "NoiseAwareRobustRegionSearch",
    "ParameterSpec",
    "MetricRule",
    "ObjectiveResult",
    "NARRSConfig",
    "SearchContext",
    "Region",
    "PointResult",
    "ConfidenceReport",
    "BaseSampler",
    "GridSampler",
    "RandomSampler",
    "LatinHypercubeSampler",
    "HybridSampler",
    "CustomSampler",
    "NARRSConfigurationError",
    "validate_setup",
]
