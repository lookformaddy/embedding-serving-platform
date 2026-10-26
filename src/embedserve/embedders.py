"""Embedders: a deterministic hashing stand-in plus a provider interface.

The hashing embedder exists so the whole platform (index, store, cache,
drift, eval) runs offline with zero downloads and zero API keys, while
remaining a drop-in stand-in for a real model: swap in any object that
implements :class:`EmbeddingProvider` and nothing else changes.

Determinism contract: the same text + the same model dimension always
produces the same vector, on any machine, because hashing uses SHA-256
(token hashes), never Python's salted ``hash()``.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

from embedserve.models import ModelVersion, Normalization, l2_normalize

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str], model: ModelVersion) -> list[list[float]]:
        """Return one vector per input text, in input order."""


def _token_vector(token: str, dimension: int) -> list[float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    # Re-hash with a counter so any dimension is reachable from 32 bytes.
    out: list[float] = []
    counter = 0
    while len(out) < dimension:
        block = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
        out.extend((b / 255.0) * 2.0 - 1.0 for b in block)
        counter += 1
    return out[:dimension]


class HashingEmbedder:
    """Deterministic, offline embedder: token hashing + mean pooling."""

    def embed(self, texts: Sequence[str], model: ModelVersion) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            tokens = _TOKEN_RE.findall(text.lower())
            if not tokens:
                vec = [0.0] * model.dimension
            else:
                acc = [0.0] * model.dimension
                for token in tokens:
                    tv = _token_vector(token, model.dimension)
                    for i, v in enumerate(tv):
                        acc[i] += v
                vec = [x / len(tokens) for x in acc]
            if model.normalization == Normalization.L2:
                vec = l2_normalize(vec)
            results.append(vec)
        return results

    def token_count(self, text: str) -> int:
        """Cheap budget-accounting proxy: whitespace token count."""
        return max(1, len(text.split()))


def count_tokens(texts: Sequence[str]) -> int:
    return sum(max(1, len(t.split())) for t in texts)
