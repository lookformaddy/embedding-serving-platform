"""Batching embed service: max-batch / max-wait semantics, quotas.

``Batcher`` is the synchronous core: ``submit()`` accumulates requests
and flushes when either ``max_batch`` items or ``max_wait_seconds``
have elapsed since the oldest pending request. Per-tenant accounting
tracks a token budget (hard cap) and a request rate (sliding window);
both raise :class:`QuotaExceeded` before any embedding work happens,
so a noisy tenant cannot starve the provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from embedserve.errors import QuotaExceeded


@dataclass
class TenantQuota:
    token_budget: int
    max_requests_per_minute: int
    tokens_used: int = 0
    request_timestamps: list[float] = field(default_factory=list)

    def check(self, tokens: int, now: float) -> None:
        window_start = now - 60.0
        self.request_timestamps = [t for t in self.request_timestamps if t > window_start]
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            raise QuotaExceeded("per-minute request limit exceeded")
        if self.tokens_used + tokens > self.token_budget:
            raise QuotaExceeded("token budget exceeded")

    def record(self, tokens: int, now: float) -> None:
        self.tokens_used += tokens
        self.request_timestamps.append(now)


@dataclass
class BatchRequest:
    tenant: str
    text: str
    tokens: int


class Batcher:
    def __init__(self, max_batch: int = 32, max_wait_seconds: float = 0.05,
                 clock=time.monotonic) -> None:
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        self.max_batch = max_batch
        self.max_wait_seconds = max_wait_seconds
        self._clock = clock
        self._pending: list[BatchRequest] = []
        self._oldest_at: float | None = None

    def submit(self, request: BatchRequest) -> list[BatchRequest] | None:
        """Return the flushed batch when a flush triggers, else None."""
        if not self._pending:
            self._oldest_at = self._clock()
        self._pending.append(request)
        if len(self._pending) >= self.max_batch:
            return self.flush()
        return None

    def flush_if_due(self) -> list[BatchRequest] | None:
        if self._pending and self._oldest_at is not None:
            if self._clock() - self._oldest_at >= self.max_wait_seconds:
                return self.flush()
        return None

    def flush(self) -> list[BatchRequest]:
        batch, self._pending = self._pending, []
        self._oldest_at = None
        return batch

    @property
    def pending(self) -> int:
        return len(self._pending)
