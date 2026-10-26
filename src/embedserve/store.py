"""Versioned embedding store.

Vectors are namespaced by model key (``name:version``) so a model
upgrade never silently mixes old and new vectors in one index. Writes
carry the model key they were embedded with; reads default to the
production model but can pin any version. ``cleanup_stale_versions``
drops every version except an explicit keep-list -- the re-index
finisher that reclaims storage after a migration.
"""

from __future__ import annotations

from embedserve.errors import StaleVersionError
from embedserve.index import ExactIndex, QuantizedPartitionedIndex
from embedserve.models import EmbeddingRecord


class EmbeddingStore:
    def __init__(self, dimension: int, n_partitions: int = 8) -> None:
        self.dimension = dimension
        self._exact: dict[str, ExactIndex] = {}
        self._ann: dict[str, QuantizedPartitionedIndex] = {}
        self._n_partitions = n_partitions

    def _indexes(self, model_key: str) -> tuple[ExactIndex, QuantizedPartitionedIndex]:
        if model_key not in self._exact:
            self._exact[model_key] = ExactIndex()
            self._ann[model_key] = QuantizedPartitionedIndex(self.dimension, self._n_partitions)
        return self._exact[model_key], self._ann[model_key]

    def upsert(self, record: EmbeddingRecord) -> None:
        if len(record.vector) != self.dimension:
            raise ValueError(f"vector dimension {len(record.vector)} != {self.dimension}")
        exact, ann = self._indexes(record.model_key)
        exact.upsert(record)
        ann.upsert(record)

    def upsert_many(self, records: list[EmbeddingRecord]) -> int:
        for record in records:
            self.upsert(record)
        return len(records)

    def delete(self, model_key: str, item_id: str) -> bool:
        if model_key not in self._exact:
            return False
        return self._exact[model_key].delete(item_id) | self._ann[model_key].delete(item_id)

    def search_exact(self, model_key: str, query: list[float], k: int):
        return self._indexes(model_key)[0].search(query, k)

    def search_ann(self, model_key: str, query: list[float], k: int, n_probe: int = 2):
        return self._indexes(model_key)[1].search(query, k, n_probe=n_probe)

    def count(self, model_key: str) -> int:
        return len(self._exact.get(model_key, ExactIndex()))

    def model_keys(self) -> list[str]:
        return sorted(self._exact)

    def cleanup_stale_versions(self, keep: list[str]) -> list[str]:
        """Drop every version not in ``keep``; return dropped keys."""
        keep_set = set(keep)
        dropped = [k for k in self._exact if k not in keep_set]
        for key in dropped:
            del self._exact[key]
            del self._ann[key]
        return sorted(dropped)

    def get(self, model_key: str, item_id: str) -> EmbeddingRecord:
        index = self._exact.get(model_key)
        if index is None or item_id not in index._records:
            raise StaleVersionError(f"no record {item_id} for model {model_key}")
        return index._records[item_id]
