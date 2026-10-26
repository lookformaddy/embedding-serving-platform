"""End-to-end demo over a synthetic corpus (generic product/help topics)."""

from __future__ import annotations

from embedserve.drift import measure_drift
from embedserve.embedders import HashingEmbedder
from embedserve.eval import evaluate
from embedserve.models import ModelStage, ModelVersion, Normalization
from embedserve.service import EmbedService, ServiceConfig

TOPICS = {
    "billing": ["how do I update my payment method", "invoice shows the wrong amount",
                "refund status for my recent order", "change billing address"],
    "shipping": ["track my delivery", "package arrived damaged", "change delivery address",
                 "estimated arrival time for my shipment"],
    "account": ["reset my password", "update my email address", "close my account",
                "enable two factor authentication"],
    "returns": ["return an item for a refund", "exchange a size", "return label not working",
                "refund timeline after a return"],
}


def synthetic_corpus() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for topic, texts in TOPICS.items():
        for i, text in enumerate(texts):
            items.append((f"{topic}-{i}", text))
    return items


def run_demo() -> dict:
    service = EmbedService(ServiceConfig(dimension=64))
    model = ModelVersion(name="hash-small", version="1.0.0", dimension=64,
                         normalization=Normalization.L2, description="offline demo model")
    service.registry.register(model)
    service.registry.transition("hash-small", "1.0.0", ModelStage.STAGING)
    service.registry.transition("hash-small", "1.0.0", ModelStage.PRODUCTION)
    corpus = synthetic_corpus()
    service.index_texts("demo-tenant", model, corpus)
    hits = service.search(model, "how can I get my money back", k=3)
    embedder = HashingEmbedder()
    queries = embedder.embed(
        ["track my package", "reset password please", "refund for damaged item"], model)
    report = evaluate(service.store, model.key, queries, k=5, n_probe=4,
                      cache_hit_rate=service.cache.stats.hit_rate)
    reference = embedder.embed([t for _, t in corpus[:8]], model)
    current = embedder.embed([t for _, t in corpus[8:]], model)
    drift = measure_drift(reference, current)
    dropped = service.store.cleanup_stale_versions(keep=[model.key])
    return {
        "indexed": len(corpus),
        "top_hits": [(h.item_id, round(h.score, 3)) for h in hits],
        "recall_at_5": round(report.recall_at_k, 3),
        "p95_ms": round(report.p95_ms, 3),
        "drift_alert": drift.alert,
        "centroid_shift": round(drift.centroid_shift, 4),
        "stale_dropped": dropped,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_demo(), indent=2))
