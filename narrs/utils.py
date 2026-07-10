from __future__ import annotations

import math
import random
from statistics import NormalDist, mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import MetricDict, MetricRule, ParameterSpec, ParamDict, PointResult, Region

try:  # optional; the package still imports without scipy
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except ImportError:
    cKDTree = None
    _HAVE_SCIPY = False

_KDTREE_MIN_POINTS = 64  # below this, brute force beats building a tree


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def safe_std(values: Sequence[float], default: float = 0.0) -> float:
    return pstdev(values) if len(values) > 1 else default


def percentile(values: Sequence[float], pct: float, default: float = 0.0) -> float:
    if not values:
        return default
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def confidence_interval(values: Sequence[float], z: float = 1.96) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    m = safe_mean(values)
    if len(values) < 2:
        return (m, m)
    sd = safe_std(values)
    half_width = z * sd / math.sqrt(len(values))
    return (m - half_width, m + half_width)


def confidence_interval_width(ci: Tuple[float, float]) -> float:
    return abs(ci[1] - ci[0])


def expected_max_under_null(n_trials: int) -> float:
    """
    Expected maximum of ``n_trials`` i.i.d. standard normals: the benchmark a
    best-of-N selection must beat under the null of no real edge. Standard
    approximation from the Deflated Sharpe Ratio literature (Bailey & Lopez de
    Prado). Returns 0.0 for n_trials <= 1 (no selection was made).
    """
    if n_trials <= 1:
        return 0.0
    nd = NormalDist()
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    a = nd.inv_cdf(1.0 - 1.0 / n_trials)
    b = nd.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return (1.0 - gamma) * a + gamma * b


def deflated_holdout_threshold(
    base_threshold: float,
    holdout_noise: float,
    n_holdout: int,
    n_trials: int,
) -> Tuple[float, float]:
    """
    Selection-adjusted ("deflated") holdout bar, DSR-style. Picking the best of
    ``n_trials`` regions biases the winner's holdout mean upward by chance.
    Require it to clear the base bar by more than a best-of-N fluke would
    produce, in units of the mean's own standard error (holdout_noise /
    sqrt(n_holdout)). A haircut on a composite holdout score, not a literal
    Sharpe. Returns (deflated_threshold, selection_penalty).
    """
    if n_holdout < 1 or holdout_noise <= 0 or n_trials <= 1:
        return base_threshold, 0.0
    standard_error = holdout_noise / math.sqrt(n_holdout)
    penalty = expected_max_under_null(n_trials) * standard_error
    return base_threshold + penalty, penalty


def normalize_metric(value: float, rule: MetricRule) -> float:
    return rule.normalize(value)


def normalize_metrics(metrics: MetricDict, rules: Dict[str, MetricRule]) -> MetricDict:
    normalized = {}
    for name, value in metrics.items():
        if name in rules:
            normalized[name] = normalize_metric(value, rules[name])
    return normalized


def weighted_metric_score(normalized_metrics: MetricDict, metric_weights: Dict[str, float]) -> float:
    total_weight = 0.0
    score = 0.0

    for name, value in normalized_metrics.items():
        weight = metric_weights.get(name, 0.0)
        score += weight * value
        total_weight += abs(weight)

    if total_weight == 0:
        return 0.0

    return score / total_weight


def violates_hard_metric_rules(metrics: MetricDict, rules: Dict[str, MetricRule]) -> bool:
    for name, value in metrics.items():
        rule = rules.get(name)
        if rule and rule.violates_hard_rule(value):
            return True
    return False


def normalize_point(point: ParamDict, parameter_specs: Dict[str, ParameterSpec]) -> Dict[str, float]:
    return {name: spec.normalize(point[name]) for name, spec in parameter_specs.items()}


def denormalize_point(point: Dict[str, float], parameter_specs: Dict[str, ParameterSpec]) -> ParamDict:
    return {name: parameter_specs[name].denormalize(value) for name, value in point.items()}


def weighted_normalized_distance(
    point_a: ParamDict,
    point_b: ParamDict,
    parameter_specs: Dict[str, ParameterSpec],
) -> float:
    total = 0.0
    for name, spec in parameter_specs.items():
        a = spec.normalize(point_a[name])
        b = spec.normalize(point_b[name])
        diff = a - b
        total += spec.weight * diff * diff
    return math.sqrt(total)


def weighted_coordinate_matrix(
    points: Sequence[ParamDict],
    parameter_specs: Dict[str, ParameterSpec],
) -> List[List[float]]:
    """L2 distance in this space equals weighted_normalized_distance:
    each axis is sqrt(weight) * normalized_value."""
    names = list(parameter_specs)
    scales = {name: math.sqrt(max(0.0, parameter_specs[name].weight)) for name in names}
    matrix: List[List[float]] = []
    for p in points:
        matrix.append([scales[name] * parameter_specs[name].normalize(p[name]) for name in names])
    return matrix


def estimate_weighted_grid_spacing(
    points: Sequence[ParamDict],
    parameter_specs: Dict[str, ParameterSpec],
) -> float:
    if len(points) < 2:
        return 1.0

    if _HAVE_SCIPY and len(points) >= _KDTREE_MIN_POINTS:
        coords = weighted_coordinate_matrix(points, parameter_specs)
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=2)  # k=2: nearest is the point itself
        nearest = [row[1] for row in distances if row[1] > 0]
        return safe_mean(nearest, default=1.0)

    nearest_distances = []
    for i, p in enumerate(points):
        best = None
        for j, q in enumerate(points):
            if i == j:
                continue
            d = weighted_normalized_distance(p, q, parameter_specs)
            if best is None or d < best:
                best = d
        if best is not None and best > 0:
            nearest_distances.append(best)

    return safe_mean(nearest_distances, default=1.0)


def region_ranges(region: Region, parameter_specs: Dict[str, ParameterSpec]) -> Dict[str, Tuple[float, float]]:
    ranges = {}
    for name in parameter_specs:
        vals = [p.parameter_point[name] for p in region.points]
        ranges[name] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    return ranges


def region_width(region: Region, parameter_specs: Dict[str, ParameterSpec]) -> float:
    if not region.points:
        return 0.0

    total = 0.0
    for name, spec in parameter_specs.items():
        vals = [spec.normalize(p.parameter_point[name]) for p in region.points]
        width = max(vals) - min(vals) if vals else 0.0
        total += spec.weight * width * width
    return math.sqrt(total)


def region_density(region: Region, parameter_specs: Dict[str, ParameterSpec]) -> float:
    width = region_width(region, parameter_specs)
    if width <= 0:
        return float(len(region.points))
    return len(region.points) / width


def average_metric_profile(metric_results: Sequence[MetricDict]) -> MetricDict:
    if not metric_results:
        return {}

    keys = set()
    for m in metric_results:
        keys.update(m.keys())

    profile = {}
    for k in keys:
        vals = [m[k] for m in metric_results if k in m]
        profile[k] = safe_mean(vals)
    return profile


def weighted_geometric_center(region: Region, parameter_specs: Dict[str, ParameterSpec]) -> ParamDict:
    if not region.points:
        return {}

    center_normalized = {}
    for name, spec in parameter_specs.items():
        vals = [spec.normalize(p.parameter_point[name]) for p in region.points]
        center_normalized[name] = safe_mean(vals)

    return denormalize_point(center_normalized, parameter_specs)


def size_bonus(width: float) -> float:
    # Saturating bonus. Wide helps, but it should not dominate quality.
    return width / (1.0 + width)


def point_to_key(point: ParamDict, precision: int = 12) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, round(v, precision)) for k, v in point.items()))


def unique_points(points: Sequence[ParamDict]) -> List[ParamDict]:
    seen = set()
    out = []
    for p in points:
        key = point_to_key(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def clamp_point(point: ParamDict, parameter_specs: Dict[str, ParameterSpec]) -> ParamDict:
    return {name: spec.clamp(point[name]) for name, spec in parameter_specs.items()}


def random_point(parameter_specs: Dict[str, ParameterSpec], rng: random.Random) -> ParamDict:
    return {name: rng.uniform(spec.min_value, spec.max_value) for name, spec in parameter_specs.items()}
