# EmbedServe — Versioned Embedding Serving Platform

## The enterprise problem

Search, recommendations, and retrieval-augmented generation all depend on
embeddings — but most teams treat embedding serving as "call a model and
store the vector." That breaks in production in predictable ways:

- A model upgrade silently invalidates every stored vector, and search
  quality collapses with no alert.
- Old and new vectors mix in one index, so results are unrankable.
- A noisy tenant burns the embedding budget for everyone else.
- ANN recall is assumed, never measured, so a bad index ships unnoticed.
- Embedding drift (a topic shift in incoming content) is discovered by
  users, not by monitoring.

EmbedServe is a small, complete platform that addresses all five: a
versioned model registry with a strict stage machine, a version-namespaced
embedding store, exact + quantized-partitioned ANN search with measured
recall, per-tenant quotas, a TTL result cache, drift monitoring, and an
offline evaluation harness — all runnable offline with zero API keys.

## Architecture overview

```
texts ──► EmbedService ──► quota check ──► TTL cache ──► embedder
              │                                     │
              ▼                                     ▼
         ModelRegistry                     EmbeddingStore (per model-key)
         (draft→staging→                    ├── ExactIndex (ground truth)
          production→deprecated)            └── QuantizedPartitionedIndex (ANN)
              │
              ▼
         Drift monitor + offline eval (recall@k, p50/p95, hit rate)
```

See `docs/architecture.md` for the full design, including the stage
machine, the ANN trade-off, and the re-index migration playbook.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # core needs no dependencies at all
python examples/run_demo.py            # end-to-end demo, synthetic corpus
python -m pytest -q                    # full test suite
```

With FastAPI installed (`pip install -r requirements.txt` includes it as
an optional extra via `pip install ".[api]"`), the HTTP surface is:

```bash
uvicorn embedserve.api:app --factory ...   # see docs/architecture.md
```

In practice, construct the app in code:

```python
from embedserve.api import create_app
from embedserve.service import EmbedService
app = create_app(EmbedService())
```

## Key design decisions and trade-offs

1. **Strict stage machine, not free-form tags.** Only
   `draft → staging → production → deprecated` (plus staging↔draft rework
   and production→staging rollback) are legal. Deprecation is terminal.
   This is what makes a model rollout a reviewable event instead of a
   silent config edit.
2. **Vectors namespaced by model key.** The store keeps one index pair
   per `name:version`. A dimension change is flagged by the registry, and
   `cleanup_stale_versions()` reclaims storage only after re-indexing.
   Cost: duplicate storage during migration. Benefit: no mixed-version
   results, ever.
3. **Deterministic hashing embedder as the default provider.** The
   platform runs offline with no downloads and no keys, and the provider
   interface means a real model swaps in without touching the index,
   store, cache, or eval code. The hashing embedder is a stand-in, not a
   quality claim — eval measures the platform, not the model.
4. **Quantized, partitioned ANN with measured recall.** int8 quantization
   cuts memory ~4×; coarse partitioning cuts the scan to `n_probe`
   partitions. Recall@k is measured against exact search on every eval
   run, so the speed/recall trade-off is a number, not a hope.
5. **Quotas checked before work, cache keyed on (model, text).**
   Embeddings are pure functions of (model, text), so caching is safe;
   quotas fail fast so one tenant cannot starve the provider.
6. **Two drift signals, not one opaque score.** Centroid shift catches
   topic drift; mean-norm shift catches representation collapse. Each
   has its own threshold and its own alert.

## What it demonstrates

Versioned model governance, ANN index design with recall measurement,
multi-tenant serving with quotas and caching, embedding drift
monitoring, offline evaluation methodology, and a clean provider
abstraction — the full embedding-serving slice of a production ML
platform, in dependency-free Python.

## Clean-room note

All code is original. All data is synthetic (generic help-center-style
topics). No employer code, data, internal architecture, project names,
or proprietary designs are used anywhere in this repository.
