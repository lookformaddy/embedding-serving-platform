"""Vector indexes: exact search plus a quantized, partitioned ANN index.

Exact search is the ground truth. The ANN index mirrors what production
systems actually do: partition the corpus into coarse clusters (by
first-coordinate bucketing -- simple, deterministic, no k-means
dependency), scan only the nearest partitions, and quantize stored
vectors to int8 so the memory footprint is ~4x smaller. Recall is
measured against exact search, never asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from embedserve.models import EmbeddingRecord, SearchHit, cosine_similarity


def _quantize(vector: list[float]) -> tuple[list[int], float]:
    scale = max((abs(x) for x in vector), default=0.0) or 1.0
    return [round(x / scale * 127) for x in vector], scale


def _dequantize(q: list[int], scale: float) -> list[float]:
    return [x / 127.0 * scale for x in q]


class ExactIndex:
    """Brute-force cosine search; the recall ground truth."""

    def __init__(self) -> None:
        self._records: dict[str, EmbeddingRecord] = {}

    def upsert(self, record: EmbeddingRecord) -> None:
        self._records[record.item_id] = record

    def delete(self, item_id: str) -> bool:
        return self._records.pop(item_id, None) is not None

    def __len__(self) -> int:
        return len(self._records)

    def search(self, query: list[float], k: int) -> list[SearchHit]:
        scored = sorted(
            ((cosine_similarity(query, r.vector), r.item_id) for r in self._records.values()),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [SearchHit(item_id=i, score=s, rank=n + 1)
                for n, (s, i) in enumerate(scored[:k])]


@dataclass
class _Partition:
    quantized: dict[str, tuple[list[int], float]] = field(default_factory=dict)


class QuantizedPartitionedIndex:
    """Coarse-partitioned int8-quantized ANN index.

    Partition assignment: bucket on the index of the largest-magnitude
    coordinate (``n_partitions`` buckets by coordinate index modulo).
    Search probes the partitions whose representative coordinate best
    matches the query's dominant coordinates, so recall trades against
    ``n_probe`` -- the knob eval measures.
    """

    def __init__(self, dimension: int, n_partitions: int = 8) -> None:
        self.dimension = dimension
        self.n_partitions = n_partitions
        self._partitions: list[_Partition] = [_Partition() for _ in range(n_partitions)]
        self._owners: dict[str, int] = {}

    def _partition_for(self, vector: list[float]) -> int:
        dominant = max(range(len(vector)), key=lambda i: abs(vector[i]))
        return dominant % self.n_partitions

    def upsert(self, record: EmbeddingRecord) -> None:
        self.delete(record.item_id)
        part = self._partition_for(record.vector)
        q, scale = _quantize(record.vector)
        self._partitions[part].quantized[record.item_id] = (q, scale)
        self._owners[record.item_id] = part

    def delete(self, item_id: str) -> bool:
        part = self._owners.pop(item_id, None)
        if part is None:
            return False
        self._partitions[part].quantized.pop(item_id, None)
        return True

    def __len__(self) -> int:
        return len(self._owners)

    def _probe_order(self, query: list[float]) -> list[int]:
        ranked = sorted(range(self.dimension), key=lambda i: -abs(query[i]))
        order: list[int] = []
        for coord in ranked:
            part = coord % self.n_partitions
            if part not in order:
                order.append(part)
        for part in range(self.n_partitions):
            if part not in order:
                order.append(part)
        return order

    def search(self, query: list[float], k: int, n_probe: int = 2) -> list[SearchHit]:
        scored: list[tuple[float, str]] = []
        for part in self._probe_order(query)[: max(1, n_probe)]:
            for item_id, (q, scale) in self._partitions[part].quantized.items():
                scored.append((cosine_similarity(query, _dequantize(q, scale)), item_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [SearchHit(item_id=i, score=s, rank=n + 1)
                for n, (s, i) in enumerate(scored[:k])]
