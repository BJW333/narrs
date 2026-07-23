"""
Reference optimizers to compare NARRS against.

All three maximise the **in-sample mean across contexts**, because that is what
standard practice does: run the backtest over your data, keep the parameters
that scored highest. None of them is a strawman -- Optuna's TPE and CMA-ES are
strong, widely used optimizers. That is the point. They are not bad at finding
the maximum; they are just answering a different question than "what should I
deploy", and the benchmark landscapes are built so those two answers differ.

Budget is measured in **objective calls**, not iterations, so every method in
the battery gets the same number of evaluations regardless of how it spends
them. A comparison run on unequal budgets tells you nothing.

``optuna`` and ``cma`` are optional. If missing, those baselines return ``None``
and the battery quietly reports fewer methods rather than failing.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence

from ..types import ParamDict, ParameterSpec, SearchContext
from .problems import normalized_score


def _mean_in_sample(
    problem,
    params: ParamDict,
    contexts: Sequence[SearchContext],
    reps: int,
) -> float:
    total = 0.0
    n = 0
    for context in contexts:
        for rep in range(reps):
            total += normalized_score(problem, params, context, rep)
            n += 1
    return total / n if n else 0.0


def _points_affordable(budget_evals: int, contexts: Sequence[SearchContext], reps: int) -> int:
    per_point = max(1, len(contexts) * reps)
    return max(1, budget_evals // per_point)


def random_search(
    problem,
    contexts: Sequence[SearchContext],
    budget_evals: int,
    reps: int = 2,
    seed: int = 0,
) -> Dict:
    """Uniform random sampling, keep the best in-sample mean."""
    rng = random.Random(seed)
    specs: List[ParameterSpec] = problem.parameter_space
    n_points = _points_affordable(budget_evals, contexts, reps)

    best_point: Optional[ParamDict] = None
    best_value = float("-inf")
    for _ in range(n_points):
        p = {s.name: rng.uniform(s.min_value, s.max_value) for s in specs}
        v = _mean_in_sample(problem, p, contexts, reps)
        if v > best_value:
            best_value, best_point = v, p

    return {
        "name": "random (in-sample mean)",
        "point": best_point,
        "in_sample_value": best_value,
        "n_evals": n_points * len(contexts) * reps,
        "n_trials": n_points,
    }


def optuna_tpe(
    problem,
    contexts: Sequence[SearchContext],
    budget_evals: int,
    reps: int = 2,
    seed: int = 0,
) -> Optional[Dict]:
    """Tree-structured Parzen Estimator (Optuna). Returns None if optuna is absent."""
    try:
        import optuna
    except ImportError:
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    specs: List[ParameterSpec] = problem.parameter_space
    n_points = _points_affordable(budget_evals, contexts, reps)

    def trial_objective(trial) -> float:
        p = {
            s.name: trial.suggest_float(s.name, s.min_value, s.max_value)
            for s in specs
        }
        return _mean_in_sample(problem, p, contexts, reps)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(trial_objective, n_trials=n_points, show_progress_bar=False)

    return {
        "name": "Optuna TPE (in-sample mean)",
        "point": dict(study.best_params),
        "in_sample_value": study.best_value,
        "n_evals": n_points * len(contexts) * reps,
        "n_trials": n_points,
    }


def cma_es(
    problem,
    contexts: Sequence[SearchContext],
    budget_evals: int,
    reps: int = 2,
    seed: int = 0,
) -> Optional[Dict]:
    """CMA-ES. Returns None if the ``cma`` package is absent.

    Optimises in the normalised unit cube and maps back, so a single sigma is
    sensible across parameters with different ranges.
    """
    try:
        import cma
    except ImportError:
        return None

    specs: List[ParameterSpec] = problem.parameter_space
    n_points = _points_affordable(budget_evals, contexts, reps)

    def to_params(vector: Sequence[float]) -> ParamDict:
        out: ParamDict = {}
        for s, v in zip(specs, vector):
            v = min(max(v, 0.0), 1.0)
            out[s.name] = s.denormalize(v)
        return out

    es = cma.CMAEvolutionStrategy(
        [0.5] * len(specs),
        0.25,
        {"bounds": [0.0, 1.0], "seed": seed + 1, "verbose": -9, "verb_disp": 0},
    )

    best_point: Optional[ParamDict] = None
    best_value = float("-inf")
    used = 0
    while used < n_points and not es.stop():
        solutions = es.ask()
        # A CMA-ES generation has to be told about every solution it asked for --
        # truncating the population mid-generation corrupts the covariance update
        # and the library rejects it outright. So stop *before* a generation that
        # would not fit, and report the evaluations actually spent rather than
        # pretending the full budget was used.
        if used + len(solutions) > n_points:
            break
        values = []
        for vector in solutions:
            p = to_params(vector)
            v = _mean_in_sample(problem, p, contexts, reps)
            values.append(-v)  # cma minimises
            if v > best_value:
                best_value, best_point = v, p
        es.tell(solutions, values)
        used += len(solutions)

    if best_point is None:
        # Budget too small for even one generation: fall back to the start point
        # rather than returning None, which would silently drop the baseline.
        best_point = to_params([0.5] * len(specs))
        best_value = _mean_in_sample(problem, best_point, contexts, reps)
        used = 1

    return {
        "name": "CMA-ES (in-sample mean)",
        "point": best_point,
        "in_sample_value": best_value,
        "n_evals": used * len(contexts) * reps,
        "n_trials": used,
    }


ALL_BASELINES: List[Callable] = [random_search, optuna_tpe, cma_es]

__all__ = ["random_search", "optuna_tpe", "cma_es", "ALL_BASELINES"]
