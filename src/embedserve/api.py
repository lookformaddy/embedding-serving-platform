"""Optional FastAPI surface. Import only when FastAPI is installed.

The core stays pure-stdlib; this module is a thin adapter over
:class:`EmbedService`, so the HTTP layer can never drift from the
service semantics (same quotas, same cache, same version pinning).
"""

from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - exercised only when installed
    FastAPI = None  # type: ignore

from embedserve.errors import EmbedServeError
from embedserve.models import ModelVersion
from embedserve.service import EmbedService


class EmbedRequest(BaseModel):
    tenant: str
    model_name: str
    model_version: str
    texts: list[str]


class SearchRequest(BaseModel):
    tenant: str
    model_name: str
    model_version: str
    query: str
    k: int = 5
    use_ann: bool = True


def create_app(service: EmbedService) -> "FastAPI":
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed")
    app = FastAPI(title="EmbedServe")

    @app.post("/embed")
    def embed(req: EmbedRequest):
        try:
            model = service.registry.get(req.model_name, req.model_version)
            vectors = service.embed_texts(req.tenant, model, req.texts)
        except EmbedServeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"model": model.key, "vectors": vectors,
                "cache_hit_rate": service.cache.stats.hit_rate}

    @app.post("/search")
    def search(req: SearchRequest):
        try:
            model = service.registry.get(req.model_name, req.model_version)
            hits = service.search(model, req.query, k=req.k, use_ann=req.use_ann)
        except EmbedServeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"hits": [{"item_id": h.item_id, "score": h.score, "rank": h.rank}
                         for h in hits]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
