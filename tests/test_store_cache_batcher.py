import pytest

from embedserve.batcher import BatchRequest, Batcher, TenantQuota
from embedserve.cache import TTLCache
from embedserve.errors import QuotaExceeded, StaleVersionError
from embedserve.models import EmbeddingRecord
from embedserve.store import EmbeddingStore


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_store_version_isolation_and_cleanup(model):
    store = EmbeddingStore(dimension=32)
    store.upsert(EmbeddingRecord(item_id="a", vector=[0.1] * 32, model_key="m:1.0"))
    store.upsert(EmbeddingRecord(item_id="a", vector=[0.2] * 32, model_key="m:2.0"))
    assert store.count("m:1.0") == 1 and store.count("m:2.0") == 1
    assert store.model_keys() == ["m:1.0", "m:2.0"]
    dropped = store.cleanup_stale_versions(keep=["m:2.0"])
    assert dropped == ["m:1.0"]
    with pytest.raises(StaleVersionError):
        store.get("m:1.0", "a")
    assert store.get("m:2.0", "a").vector == [0.2] * 32


def test_store_dimension_mismatch_rejected(model):
    store = EmbeddingStore(dimension=32)
    with pytest.raises(ValueError):
        store.upsert(EmbeddingRecord(item_id="a", vector=[0.1] * 16, model_key="m:1.0"))


def test_cache_hit_rate_and_ttl():
    clock = FakeClock()
    cache = TTLCache(ttl_seconds=10.0, clock=clock)
    assert cache.get("m:1.0", "hello") is None
    cache.put("m:1.0", "hello", [0.5] * 4)
    assert cache.get("m:1.0", "hello") == [0.5] * 4
    clock.now += 11.0
    assert cache.get("m:1.0", "hello") is None
    assert cache.stats.hits == 1 and cache.stats.misses == 2
    assert cache.stats.hit_rate == pytest.approx(1 / 3)


def test_batcher_flushes_on_max_batch_and_max_wait():
    clock = FakeClock()
    batcher = Batcher(max_batch=3, max_wait_seconds=5.0, clock=clock)
    assert batcher.submit(BatchRequest("t", "a", 1)) is None
    assert batcher.submit(BatchRequest("t", "b", 1)) is None
    flushed = batcher.submit(BatchRequest("t", "c", 1))
    assert flushed is not None and len(flushed) == 3
    batcher.submit(BatchRequest("t", "d", 1))
    assert batcher.flush_if_due() is None
    clock.now += 6.0
    flushed = batcher.flush_if_due()
    assert flushed is not None and [r.text for r in flushed] == ["d"]


def test_tenant_quota_token_and_rate_limits():
    quota = TenantQuota(token_budget=10, max_requests_per_minute=1)
    quota.check(5, now=1000.0)
    quota.record(5, now=1000.0)
    with pytest.raises(QuotaExceeded):
        quota.check(1, now=1001.0)  # rate limit hit first
    quota2 = TenantQuota(token_budget=10, max_requests_per_minute=100)
    quota2.check(10, now=1000.0)
    quota2.record(10, now=1000.0)
    with pytest.raises(QuotaExceeded):
        quota2.check(1, now=1001.0)  # token budget exhausted
