"""TTL result cache with hit-rate accounting.

Keys are (model key, text) pairs -- embeddings are pure functions of
(model, text), so caching them is safe and the hit rate is a real
production metric, not a benchmark artifact. A fakeable clock parameter
keeps TTL tests deterministic without sleeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class TTLCache:
    def __init__(self, ttl_seconds: float = 3600.0, clock=time.monotonic) -> None:
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[str, tuple[list[float], float]] = {}
        self.stats = CacheStats()

    def _key(self, model_key: str, text: str) -> str:
        return f"{model_key}\x00{text}"

    def get(self, model_key: str, text: str) -> list[float] | None:
        entry = self._entries.get(self._key(model_key, text))
        if entry is None:
            self.stats.misses += 1
            return None
        vector, expires_at = entry
        if self._clock() >= expires_at:
            del self._entries[self._key(model_key, text)]
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return list(vector)

    def put(self, model_key: str, text: str, vector: list[float]) -> None:
        self._entries[self._key(model_key, text)] = (list(vector), self._clock() + self.ttl_seconds)

    def __len__(self) -> int:
        return len(self._entries)
