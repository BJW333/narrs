"""
Benchmark battery: synthetic landscapes with a known right answer, external
baselines, and the overfitting report card.

    from narrs.benchmarks import run_all
    run_all()

Optional extras: ``optuna`` and ``cma`` enable two of the three baselines.
Everything else is pure standard library.
"""

from .problems import (
    PlateauSpikeProblem,
    DecoyOverfittingProblem,
    normalized_score,
    candidate_pool,
    performance_matrix,
)
from .baselines import random_search, optuna_tpe, cma_es, ALL_BASELINES
from .battery import run_battery, run_all, print_report, MethodOutcome

__all__ = [
    "PlateauSpikeProblem",
    "DecoyOverfittingProblem",
    "normalized_score",
    "candidate_pool",
    "performance_matrix",
    "random_search",
    "optuna_tpe",
    "cma_es",
    "ALL_BASELINES",
    "run_battery",
    "run_all",
    "print_report",
    "MethodOutcome",
]
