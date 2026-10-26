import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from embedserve.api import create_app
from embedserve.models import ModelStage
from embedserve.service import EmbedService, ServiceConfig


@pytest.fixture
def client(model):
    svc = EmbedService(ServiceConfig(dimension=32))
    svc.registry.register(model)
    svc.registry.transition(model.name, model.version, ModelStage.STAGING)
    svc.registry.transition(model.name, model.version, ModelStage.PRODUCTION)
    return TestClient(create_app(svc)), model


def test_health_and_embed_search(client):
    app, model = client
    assert app.get("/health").json() == {"status": "ok"}
    payload = {"tenant": "t", "model_name": model.name, "model_version": model.version,
               "texts": ["reset my password"]}
    resp = app.post("/embed", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["vectors"][0]) == 32
    search = {"tenant": "t", "model_name": model.name, "model_version": model.version,
              "query": "reset my password", "k": 1, "use_ann": False}
    assert app.post("/search", json=search).status_code == 200


def test_unknown_model_returns_400(client):
    app, _ = client
    payload = {"tenant": "t", "model_name": "nope", "model_version": "9.9", "texts": ["hi"]}
    assert app.post("/embed", json=payload).status_code == 400
