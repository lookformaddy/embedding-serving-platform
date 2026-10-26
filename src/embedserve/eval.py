"""Offline evaluation: ANN recall@k, latency percentiles, cache hit rate.

``evaluate`` runs a fixed query set against both the exact and the ANN
index, so recall@k is measured, not assumed. Latency is reported as
p50/p95 in milliseconds from wall-clock timings of the ANN path.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from embedserve.models import SearchHit
from embedserve.store import EmbeddingStore


@dataclass
class EvalReport:
    recall_at_k: float
    p50_ms: float
    p95_ms: float
    cache_hit_rate: float
    n_queries: int
    k: int


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(rank), min(int(rank) + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def recall_at_k(exact: list[SearchHit], ann: list[SearchHit], k: int) -> float:
    exact_ids = {h.item_id for h in exact[:k]}
    ann_ids = {h.item_id for h in ann[:k]}
    if not exact_ids:
        return 0.0
    return len(exact_ids & ann_ids) / len(exact_ids)


def evaluate(store: EmbeddingStore, model_key: str, queries: list[list[float]],
             k: int = 5, n_probe: int = 2, cache_hit_rate: float = 0.0) -> EvalReport:
    recalls: list[float] = []
    latencies: list[float] = []
    for query in queries:
        start = time.perf_counter()
        ann = store.search_ann(model_key, query, k, n_probe=n_probe)
        latencies.append((time.perf_counter() - start) * 1000.0)
        exact = store.search_exact(model_key, query, k)
        recalls.append(recall_at_k(exact, ann, k))
    return EvalReport(
        recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        cache_hit_rate=cache_hit_rate,
        n_queries=len(queries),
        k=k,
    )
