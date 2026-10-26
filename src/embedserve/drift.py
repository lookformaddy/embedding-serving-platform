"""Embedding drift monitoring: centroid shift + mean-norm shift.

Two cheap, interpretable signals instead of one opaque score. Centroid
shift catches topic drift (the average vector moved); mean-norm shift
catches representation collapse or scale changes (vectors got shorter
or longer on average). Both are Euclidean distances on mean vectors,
so thresholds are set in the same units as the embedding space.
"""

from __future__ import annotations

import math

from embedserve.models import DriftReport


def _centroid(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def _mean_norm(vectors: list[list[float]]) -> float:
    return sum(math.sqrt(sum(x * x for x in v)) for v in vectors) / len(vectors)


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def measure_drift(reference: list[list[float]], current: list[list[float]],
                  centroid_threshold: float = 0.15,
                  norm_threshold: float = 0.10) -> DriftReport:
    if not reference or not current:
        raise ValueError("reference and current sets must be non-empty")
    if len(reference[0]) != len(current[0]):
        raise ValueError("reference and current dimensions must match")
    centroid_shift = _distance(_centroid(reference), _centroid(current))
    mean_norm_shift = abs(_mean_norm(reference) - _mean_norm(current))
    return DriftReport(
        centroid_shift=centroid_shift, mean_norm_shift=mean_norm_shift,
        centroid_threshold=centroid_threshold, norm_threshold=norm_threshold,
        n_reference=len(reference), n_current=len(current),
    )
