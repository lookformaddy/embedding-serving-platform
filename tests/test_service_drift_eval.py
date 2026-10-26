import pytest

from embedserve.batcher import TenantQuota
from embedserve.demo import run_demo, synthetic_corpus
from embedserve.drift import measure_drift
from embedserve.embedders import HashingEmbedder
from embedserve.errors import QuotaExceeded
from embedserve.eval import evaluate, recall_at_k
from embedserve.models import SearchHit


def test_service_embed_uses_cache(service):
    svc, model = service
    first = svc.embed_texts("tenant-a", model, ["reset my password"])
    second = svc.embed_texts("tenant-a", model, ["reset my password"])
    assert first == second
    assert svc.cache.stats.hits == 1
    assert svc.cache.stats.hit_rate == pytest.approx(0.5)


def test_service_index_and_search(service):
    svc, model = service
    svc.index_texts("tenant-a", model, [("d0", "reset my password"), ("d1", "track my package")])
    hits = svc.search(model, "reset my password", k=1)
    assert hits[0].item_id == "d0"
    assert svc.store.count(model.key) == 2


def test_service_quota_enforced(service):
    svc, model = service
    svc.set_quota("tenant-b", TenantQuota(token_budget=1, max_requests_per_minute=100))
    with pytest.raises(QuotaExceeded):
        svc.embed_texts("tenant-b", model, ["this text has many tokens in it for sure"])


def test_drift_detects_shift_and_ignores_identical():
    ref = [[1.0, 0.0]] * 10
    same = [[1.0, 0.0]] * 10
    report = measure_drift(ref, same)
    assert report.alert is False and report.centroid_shift == pytest.approx(0.0)
    shifted = [[0.0, 1.0]] * 10
    report = measure_drift(ref, shifted)
    assert report.alert is True
    with pytest.raises(ValueError):
        measure_drift([], same)
    with pytest.raises(ValueError):
        measure_drift(ref, [[1.0]] * 5)


def test_recall_at_k_math():
    exact = [SearchHit("a", 1.0, 1), SearchHit("b", 0.9, 2), SearchHit("c", 0.8, 3)]
    ann = [SearchHit("a", 1.0, 1), SearchHit("x", 0.9, 2), SearchHit("c", 0.8, 3)]
    assert recall_at_k(exact, ann, 3) == pytest.approx(2 / 3)
    assert recall_at_k([], ann, 3) == 0.0


def test_eval_report_and_demo_end_to_end():
    result = run_demo()
    assert result["indexed"] == len(synthetic_corpus()) == 16
    assert 0.0 <= result["recall_at_5"] <= 1.0
    assert result["p95_ms"] >= 0.0
    assert len(result["top_hits"]) == 3
    assert result["stale_dropped"] == []
