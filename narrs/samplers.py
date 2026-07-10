from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

from .types import NARRSConfig, ParameterSpec, ParamDict, Region
from .grid import (
    generate_sparse_grid,
    generate_finer_grid_inside_regions,
    generate_boundary_points_around_regions,
)
from .utils import random_point, unique_points


class BaseSampler(ABC):
    """
    Base class for candidate-point samplers.

    A sampler decides WHICH parameter points to test.

    NARRS decides:
        - how to score them
        - how to reject unstable points
        - how to cluster robust regions
        - how to validate regions on holdout contexts
    """

    @abstractmethod
    def generate_initial(
        self,
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
    ) -> List[ParamDict]:
        pass

    @abstractmethod
    def generate_next(
        self,
        best_regions: Sequence[Region],
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
        current_candidate_count: int,
        round_index: int,
    ) -> List[ParamDict]:
        pass


class GridSampler(BaseSampler):
    """
    Structured sparse-grid sampler.

    This matches the original implementation.
    Useful for visualization and low-dimensional problems.
    """

    def generate_initial(
        self,
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
    ) -> List[ParamDict]:
        return generate_sparse_grid(
            parameter_specs,
            density=config.initial_grid_density,
        )

    def generate_next(
        self,
        best_regions: Sequence[Region],
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
        current_candidate_count: int,
        round_index: int,
    ) -> List[ParamDict]:
        density = max(2, config.initial_grid_density + 1)

        exploitation = generate_finer_grid_inside_regions(
            best_regions,
            parameter_specs,
            density=density,
            max_regions=config.max_regions_to_refine,
        )

        boundary = generate_boundary_points_around_regions(
            best_regions,
            parameter_specs,
            boundary_fraction=config.boundary_fraction,
            density=max(2, density - 1),
            max_regions=config.max_regions_to_refine,
        )

        exploration_count = max(
            1,
            int(current_candidate_count * config.exploration_fraction),
        )

        exploration = [
            random_point(parameter_specs, rng)
            for _ in range(exploration_count)
        ]

        return unique_points(exploitation + boundary + exploration)


class RandomSampler(BaseSampler):
    """
    Pure random sampler.

    Useful for higher-dimensional spaces where grids explode.
    """

    def generate_initial(
        self,
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
    ) -> List[ParamDict]:
        count = config.initial_candidate_count

        return [
            random_point(parameter_specs, rng)
            for _ in range(count)
        ]

    def generate_next(
        self,
        best_regions: Sequence[Region],
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
        current_candidate_count: int,
        round_index: int,
    ) -> List[ParamDict]:
        count = max(1, current_candidate_count)

        return [
            random_point(parameter_specs, rng)
            for _ in range(count)
        ]


class LatinHypercubeSampler(BaseSampler):
    """
    Latin hypercube sampler.

    More space-filling than plain random sampling.
    Better universal default than pure grid for many problems.
    """

    def generate_initial(
        self,
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
    ) -> List[ParamDict]:
        return generate_latin_hypercube_candidates(
            parameter_specs,
            count=config.initial_candidate_count,
            rng=rng,
        )

    def generate_next(
        self,
        best_regions: Sequence[Region],
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
        current_candidate_count: int,
        round_index: int,
    ) -> List[ParamDict]:
        count = max(1, current_candidate_count)

        return generate_latin_hypercube_candidates(
            parameter_specs,
            count=count,
            rng=rng,
        )


class HybridSampler(BaseSampler):
    """
    Recommended default sampler.

    It uses:
        - grid or Latin hypercube for the initial broad search
        - region refinement inside best regions
        - boundary testing around best regions
        - exploration outside best regions

    This keeps the NARRS idea:
        zoom into REGIONS, not points
    while avoiding being locked to pure grid search.
    """

    def __init__(
        self,
        initial_strategy: str = "latin_hypercube",
        exploration_strategy: str = "latin_hypercube",
    ) -> None:
        self.initial_strategy = initial_strategy
        self.exploration_strategy = exploration_strategy

    def generate_initial(
        self,
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
    ) -> List[ParamDict]:
        if self.initial_strategy == "grid":
            return generate_sparse_grid(
                parameter_specs,
                density=config.initial_grid_density,
            )

        if self.initial_strategy == "random":
            return [
                random_point(parameter_specs, rng)
                for _ in range(config.initial_candidate_count)
            ]

        if self.initial_strategy == "latin_hypercube":
            return generate_latin_hypercube_candidates(
                parameter_specs,
                count=config.initial_candidate_count,
                rng=rng,
            )

        raise ValueError(f"Unknown initial_strategy: {self.initial_strategy}")

    def generate_next(
        self,
        best_regions: Sequence[Region],
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
        current_candidate_count: int,
        round_index: int,
    ) -> List[ParamDict]:
        density = max(2, config.initial_grid_density + 1)

        # Exploit/refine inside robust regions.
        exploitation = generate_finer_grid_inside_regions(
            best_regions,
            parameter_specs,
            density=density,
            max_regions=config.max_regions_to_refine,
        )

        # Test boundaries so the algorithm knows if the region is truly wide.
        boundary = generate_boundary_points_around_regions(
            best_regions,
            parameter_specs,
            boundary_fraction=config.boundary_fraction,
            density=max(2, density - 1),
            max_regions=config.max_regions_to_refine,
        )

        # Keep exploration outside current best regions.
        exploration_count = max(
            1,
            int(current_candidate_count * config.exploration_fraction),
        )

        if self.exploration_strategy == "random":
            exploration = [
                random_point(parameter_specs, rng)
                for _ in range(exploration_count)
            ]

        elif self.exploration_strategy == "latin_hypercube":
            exploration = generate_latin_hypercube_candidates(
                parameter_specs,
                count=exploration_count,
                rng=rng,
            )

        elif self.exploration_strategy == "grid":
            # Grid exploration is allowed, but usually less ideal globally.
            exploration_density = max(2, int(exploration_count ** (1 / max(1, len(parameter_specs)))))
            exploration = generate_sparse_grid(
                parameter_specs,
                density=exploration_density,
                max_points=exploration_count,
                rng=rng,
            )

        else:
            raise ValueError(f"Unknown exploration_strategy: {self.exploration_strategy}")

        return unique_points(exploitation + boundary + exploration)


class CustomSampler(BaseSampler):
    """
    Wrapper for user-provided sampler functions.

    initial_fn:
        function(parameter_specs, config, rng) -> list[parameter points]

    next_fn:
        function(best_regions, parameter_specs, config, rng, current_candidate_count, round_index)
        -> list[parameter points]
    """

    def __init__(self, initial_fn, next_fn) -> None:
        self.initial_fn = initial_fn
        self.next_fn = next_fn

    def generate_initial(
        self,
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
    ) -> List[ParamDict]:
        return self.initial_fn(parameter_specs, config, rng)

    def generate_next(
        self,
        best_regions: Sequence[Region],
        parameter_specs: Dict[str, ParameterSpec],
        config: NARRSConfig,
        rng: random.Random,
        current_candidate_count: int,
        round_index: int,
    ) -> List[ParamDict]:
        return self.next_fn(
            best_regions,
            parameter_specs,
            config,
            rng,
            current_candidate_count,
            round_index,
        )


def generate_latin_hypercube_candidates(
    parameter_specs: Dict[str, ParameterSpec],
    count: int,
    rng: random.Random,
) -> List[ParamDict]:
    """
    Generate Latin hypercube candidates.

    This gives better coverage than pure random sampling because each
    parameter range is split into bins and sampled once per bin.
    """
    if count <= 0:
        return []

    names = list(parameter_specs.keys())

    # For each parameter, create shuffled normalized samples.
    normalized_values = {}

    for name in names:
        values = []

        for i in range(count):
            low = i / count
            high = (i + 1) / count
            values.append(rng.uniform(low, high))

        rng.shuffle(values)
        normalized_values[name] = values

    candidates = []

    for i in range(count):
        point = {}

        for name in names:
            spec = parameter_specs[name]
            norm_value = normalized_values[name][i]
            point[name] = spec.denormalize(norm_value)

        candidates.append(point)

    return candidates