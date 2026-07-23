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
from .aggregators import (
    ContextAggregator,
    WorstCaseAggregator,
    CVaRAggregator,
    MeanStdAggregator,
    MeanAggregator,
    resolve_aggregator,
)
from .metrics import (
    pbo_cscv,
    deflated_sharpe,
    probabilistic_sharpe,
    oos_decay,
    sharpe,
    cvar,
)

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
    # context aggregation (pluggable robustness policy)
    "ContextAggregator",
    "WorstCaseAggregator",
    "CVaRAggregator",
    "MeanStdAggregator",
    "MeanAggregator",
    "resolve_aggregator",
    # overfitting metrics
    "pbo_cscv",
    "deflated_sharpe",
    "probabilistic_sharpe",
    "oos_decay",
    "sharpe",
    "cvar",
]

# narrs.benchmarks and narrs.adapters are deliberately NOT imported here.
# Benchmarks pull in optional extras (optuna, cma) and adapters are
# domain-specific; both stay opt-in so `import narrs` remains dependency-free.
