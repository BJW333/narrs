from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .types import ParameterSpec, PointResult, Region
from .utils import (
    _HAVE_SCIPY,
    _KDTREE_MIN_POINTS,
    cKDTree,
    weighted_coordinate_matrix,
    weighted_normalized_distance,
)


def find_nearby_points(
    point: PointResult,
    all_points: Sequence[PointResult],
    parameter_specs: Dict[str, ParameterSpec],
    neighbor_radius: float,
) -> List[PointResult]:
    neighbors: List[PointResult] = []

    for other in all_points:
        if other is point:
            continue

        d = weighted_normalized_distance(
            point.parameter_point,
            other.parameter_point,
            parameter_specs,
        )

        if d <= neighbor_radius:
            neighbors.append(other)

    return neighbors


def dbscan_style_cluster(
    surviving_points: Sequence[PointResult],
    parameter_specs: Dict[str, ParameterSpec],
    neighbor_radius: float,
    minimum_points_per_region: int,
) -> Tuple[List[Region], List[PointResult]]:
    """
    DBSCAN-style density clustering.

    A point becomes part of a robust region only if enough surviving
    neighbors are nearby under weighted normalized distance.
    """
    regions: List[Region] = []
    rejected_isolated: List[PointResult] = []
    points = list(surviving_points)
    n = len(points)
    if n == 0:
        return regions, rejected_isolated

    param_points = [p.parameter_point for p in points]

    # KD-tree gives O(n log n) radius queries when scipy is present; otherwise
    # the exact O(n^2) scan. Same inclusive radius, so results are identical.
    if _HAVE_SCIPY and n >= _KDTREE_MIN_POINTS:
        coords = weighted_coordinate_matrix(param_points, parameter_specs)
        tree = cKDTree(coords)

        def neighbor_indices(i: int) -> List[int]:
            return [j for j in tree.query_ball_point(coords[i], neighbor_radius) if j != i]
    else:
        def neighbor_indices(i: int) -> List[int]:
            src = param_points[i]
            found: List[int] = []
            for j in range(n):
                if j != i and weighted_normalized_distance(src, param_points[j], parameter_specs) <= neighbor_radius:
                    found.append(j)
            return found

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = neighbor_indices(i)

        # DBSCAN convention includes the core point in min_samples.
        if len(neighbors) + 1 < minimum_points_per_region:
            rejected_isolated.append(points[i])
            continue

        region_idx = {i, *neighbors}
        to_expand = list(neighbors)

        while to_expand:
            current = to_expand.pop()
            if visited[current]:
                continue
            visited[current] = True

            current_neighbors = neighbor_indices(current)
            if len(current_neighbors) + 1 >= minimum_points_per_region:
                for j in current_neighbors:
                    if j not in region_idx:
                        region_idx.add(j)
                        to_expand.append(j)

        regions.append(Region(points=[points[j] for j in region_idx]))

    return regions, rejected_isolated
