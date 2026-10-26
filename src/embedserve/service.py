"""EmbedService: wires registry, embedder, store, cache, batcher, quotas.

The service is the unit operators actually run. ``embed_texts`` is the
hot path: quota check -> cache lookup -> batch embed -> cache fill.
``index_texts`` embeds and stores. ``search`` reads the production
model's index unless a version is pinned, so a model rollout that has
not re-indexed yet cannot silently serve mixed-version results -- the
caller must pass the new version explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from embedserve.batcher import BatchRequest, Batcher, TenantQuota
from embedserve.cache import TTLCache
from embedserve.embedders import HashingEmbedder, count_tokens
from embedserve.errors import QuotaExceeded
from embedserve.models import EmbeddingRecord, ModelVersion, SearchHit
from embedserve.registry import EmbeddingModelRegistry
from embedserve.store import EmbeddingStore


@dataclass
class ServiceConfig:
    dimension: int = 64
    max_batch: int = 32
    max_wait_seconds: float = 0.05
    cache_ttl_seconds: float = 3600.0
    default_token_budget: int = 100_000
    default_requests_per_minute: int = 600


class EmbedService:
    def __init__(self, config: ServiceConfig | None = None,
                 embedder: HashingEmbedder | None = None,
                 clock=time.monotonic) -> None:
        self.config = config or ServiceConfig()
        self.registry = EmbeddingModelRegistry()
        self.store = EmbeddingStore(self.config.dimension)
        self.cache = TTLCache(self.config.cache_ttl_seconds, clock=clock)
        self.batcher = Batcher(self.config.max_batch, self.config.max_wait_seconds, clock=clock)
        self.embedder = embedder or HashingEmbedder()
        self._quotas: dict[str, TenantQuota] = {}
        self._clock = clock

    def _quota(self, tenant: str) -> TenantQuota:
        if tenant not in self._quotas:
            self._quotas[tenant] = TenantQuota(
                token_budget=self.config.default_token_budget,
                max_requests_per_minute=self.config.default_requests_per_minute,
            )
        return self._quotas[tenant]

    def set_quota(self, tenant: str, quota: TenantQuota) -> None:
        self._quotas[tenant] = quota

    def embed_texts(self, tenant: str, model: ModelVersion,
                    texts: list[str]) -> list[list[float]]:
        tokens = count_tokens(texts)
        quota = self._quota(tenant)
        now = self._clock()
        quota.check(tokens, now)
        quota.record(tokens, now)
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []
        for i, text in enumerate(texts):
            cached = self.cache.get(model.key, text)
            if cached is not None:
                results[i] = cached
            else:
                misses.append(i)
        if misses:
            vectors = self.embedder.embed([texts[i] for i in misses], model)
            for idx, vector in zip(misses, vectors):
                self.cache.put(model.key, texts[idx], vector)
                results[idx] = vector
        return [v if v is not None else [] for v in results]

    def index_texts(self, tenant: str, model: ModelVersion,
                    items: list[tuple[str, str]]) -> int:
        vectors = self.embed_texts(tenant, model, [text for _, text in items])
        records = [EmbeddingRecord(item_id=item_id, vector=vec, model_key=model.key, text=text)
                   for (item_id, text), vec in zip(items, vectors)]
        return self.store.upsert_many(records)

    def search(self, model: ModelVersion, query_text: str, k: int = 5,
               n_probe: int = 2, use_ann: bool = True) -> list[SearchHit]:
        query = self.embedder.embed([query_text], model)[0]
        if use_ann:
            return self.store.search_ann(model.key, query, k, n_probe=n_probe)
        return self.store.search_exact(model.key, query, k)
