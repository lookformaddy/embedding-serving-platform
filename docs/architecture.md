# Architecture

## Components

| Component | Responsibility |
|---|---|
| `EmbeddingModelRegistry` | Model versions + stage machine; dimension-change flagging |
| `HashingEmbedder` / `EmbeddingProvider` | Offline deterministic embedder; pluggable real-model interface |
| `ExactIndex` | Brute-force cosine search; recall ground truth |
| `QuantizedPartitionedIndex` | int8-quantized, coarse-partitioned ANN search |
| `EmbeddingStore` | Per-model-key index pairs; upsert/delete; stale-version cleanup |
| `TTLCache` | (model, text) result cache with hit-rate stats |
| `Batcher` + `TenantQuota` | max-batch / max-wait flushing; token budget + rate limits |
| `EmbedService` | The unit operators run: quota → cache → embed → store |
| `drift.measure_drift` | Centroid shift + mean-norm shift vs a reference set |
| `eval.evaluate` | Recall@k (ANN vs exact), p50/p95 latency, cache hit rate |
| `api.create_app` | Optional FastAPI adapter over `EmbedService` (only if installed) |

## Diagram (description)

```
Client ──POST /embed──► EmbedService ──► TenantQuota (fail fast)
                                      ──► TTLCache (hit? return)
                                      ──► EmbeddingProvider
                                      ──► TTLCache (fill)
Client ──POST /search─► EmbedService ──► EmbeddingStore[model-key]
                                      ├── ExactIndex  (eval / debugging)
                                      └── ANN Index   (serving path)
Background ──► drift.measure_drift(reference, current) ──► alert?
Offline   ──► eval.evaluate(store, queries) ──► recall@k, p50/p95
```

## Stage machine

```
draft ──► staging ──► production ──► deprecated
          ▲    │          │
          └───┘          └──► staging (rollback)
```

Deprecation is terminal. Only one version per model name should be in
production at a time; the registry does not enforce uniqueness (to keep
blue/green rollouts possible) but `production_model()` returns the first
production version, so operators keep exactly one.

## ANN design trade-off

Partitioning by dominant-coordinate bucket is deliberately simple and
deterministic: no k-means training step, no cluster drift to manage.
The cost is lower recall than a learned index at small `n_probe`; the
mitigation is that recall is measured on every eval run and `n_probe`
is a tuned knob, not a constant. int8 quantization costs a small
similarity error for a ~4× memory saving — acceptable for a serving
index, unacceptable for the eval ground truth, which is why exact
search stays unquantized.

## Model migration playbook (re-index without downtime)

1. Register the new version; promote it to staging.
2. Embed the corpus with the new version into its own model-key namespace.
3. Run `eval.evaluate` against the new namespace; require recall@k and
   drift checks to pass.
4. Promote to production; point the serving path at the new key.
5. Call `cleanup_stale_versions(keep=[new_key])` to drop old vectors.

## Deployment

Stateless core: the service holds indexes in memory. For production,
persist the store behind a vector database and the registry/cache behind
a durable store; the interfaces (`EmbeddingStore`, `TTLCache`) are the
seams. The Dockerfile runs the demo; wire `uvicorn` with `create_app()`
for the HTTP service. CI is defined in `docs/github-actions-ci.yml` —
activate it by moving that file to `.github/workflows/ci.yml` (the
push integration cannot create `.github/workflows/` directly).
