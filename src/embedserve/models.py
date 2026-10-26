"""Value objects: model metadata, vectors, and search results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class ModelStage(str, Enum):
    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class Normalization(str, Enum):
    NONE = "none"
    L2 = "l2"


class Pooling(str, Enum):
    MEAN = "mean"
    CLS = "cls"
    MAX = "max"


@dataclass(frozen=True)
class ModelVersion:
    """One registered embedding model version.

    ``dimension`` is the vector width the model emits; ``normalized``
    records whether emitted vectors are L2-unit length, which decides
    whether cosine similarity equals a dot product at query time.
    """

    name: str
    version: str
    dimension: int
    normalization: Normalization = Normalization.L2
    pooling: Pooling = Pooling.MEAN
    stage: ModelStage = ModelStage.DRAFT
    description: str = ""

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if not self.name or not self.version:
            raise ValueError("name and version are required")

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"


@dataclass
class EmbeddingRecord:
    item_id: str
    vector: list[float]
    model_key: str
    text: str = ""


@dataclass
class SearchHit:
    item_id: str
    score: float
    rank: int


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [x / norm for x in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must share a dimension")
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    if denom == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must share a dimension")
    return sum(x * y for x, y in zip(a, b))


@dataclass
class DriftReport:
    centroid_shift: float
    mean_norm_shift: float
    centroid_threshold: float
    norm_threshold: float
    n_reference: int
    n_current: int

    @property
    def alert(self) -> bool:
        return (
            self.centroid_shift > self.centroid_threshold
            or self.mean_norm_shift > self.norm_threshold
        )
