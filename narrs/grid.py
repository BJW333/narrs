from __future__ import annotations

import itertools
import random
from typing import Dict, List, Sequence

from .types import ParameterSpec, ParamDict, Region
from .utils import (
    clamp_point,
    random_point,
    region_ranges,
    unique_points,
    weighted_geometric_center,
    weighted_normalized_distance,
)


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [(a + b) / 2.0]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def generate_sparse_grid(
    parameter_specs: Dict[str, ParameterSpec],
    density: int,
    max_points: int | None = None,
    rng: random.Random | None = None,
) -> List[ParamDict]:
    """
    Generates a Cartesian grid. Good for low/medium dimensional spaces.

    For high-dimensional problems, set max_points to randomly downsample.
    """
    names = list(parameter_specs.keys())
    values = [
        linspace(parameter_specs[name].min_value, parameter_specs[name].max_value, density)
        for name in names
    ]

    points = [dict(zip(names, combo)) for combo in itertools.product(*values)]

    if max_points is not None and len(points) > max_points:
        if rng is None:
            rng = random.Random(0)
        points = rng.sample(points, max_points)

    return points


def generate_finer_grid_inside_regions(
    regions: Sequence[Region],
    parameter_specs: Dict[str, ParameterSpec],
    density: int,
    max_regions: int,
) -> List[ParamDict]:
    selected = list(regions)[:max_regions]
    points: List[ParamDict] = []

    for region in selected:
        ranges = region_ranges(region, parameter_specs)
        names = list(parameter_specs.keys())
        values = [linspace(ranges[name][0], ranges[name][1], density) for name in names]
        points.extend(dict(zip(names, combo)) for combo in itertools.product(*values))

    return unique_points([clamp_point(p, parameter_specs) for p in points])


def generate_boundary_points_around_regions(
    regions: Sequence[Region],
    parameter_specs: Dict[str, ParameterSpec],
    boundary_fraction: float,
    density: int,
    max_regions: int,
) -> List[ParamDict]:
    selected = list(regions)[:max_regions]
    points: List[ParamDict] = []

    for region in selected:
        ranges = region_ranges(region, parameter_specs)
        expanded = {}

        for name, spec in parameter_specs.items():
            lo, hi = ranges[name]
            span = spec.max_value - spec.min_value
            pad = boundary_fraction * span
            expanded[name] = (max(spec.min_value, lo - pad), min(spec.max_value, hi + pad))

        names = list(parameter_specs.keys())
        values = [linspace(expanded[name][0], expanded[name][1], density) for name in names]
        points.extend(dict(zip(names, combo)) for combo in itertools.product(*values))

    return unique_points([clamp_point(p, parameter_specs) for p in points])


def generate_sparse_grid_outside_regions(
    parameter_specs: Dict[str, ParameterSpec],
    regions: Sequence[Region],
    exploration_fraction: float,
    current_grid_size: int,
    rng: random.Random,
) -> List[ParamDict]:
    n = max(1, int(current_grid_size * exploration_fraction))
    return [random_point(parameter_specs, rng) for _ in range(n)]


def choose_points_inside_region(
    region: Region,
    parameter_specs: Dict[str, ParameterSpec],
    count: int,
    rng: random.Random,
) -> List[ParamDict]:
    if not region.points:
        return []

    selected: List[ParamDict] = []

    center = weighted_geometric_center(region, parameter_specs)
    if center:
        selected.append(center)

    sorted_by_score = sorted(region.points, key=lambda p: p.robust_score, reverse=True)
    selected.extend([p.parameter_point for p in sorted_by_score[: max(1, count // 4)]])

    ranges = region_ranges(region, parameter_specs)

    # Edge-like points: farthest from center.
    center_point = center
    edge_points = sorted(
        region.points,
        key=lambda p: weighted_normalized_distance(
            p.parameter_point,
            center_point,
            parameter_specs,
        ),
        reverse=True,
    )
    selected.extend([p.parameter_point for p in edge_points[: max(1, count // 4)]])

    # Random interior samples from bounding box.
    names = list(parameter_specs.keys())
    for _ in range(max(1, count - len(selected))):
        p = {}
        for name in names:
            lo, hi = ranges[name]
            p[name] = rng.uniform(lo, hi)
        selected.append(p)

    return unique_points([clamp_point(p, parameter_specs) for p in selected])[:count]
