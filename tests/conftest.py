import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from embedserve.models import ModelVersion, Normalization
from embedserve.service import EmbedService, ServiceConfig


@pytest.fixture
def model():
    return ModelVersion(name="hash-small", version="1.0.0", dimension=32,
                        normalization=Normalization.L2)


@pytest.fixture
def service(model):
    svc = EmbedService(ServiceConfig(dimension=32))
    svc.registry.register(model)
    return svc, model
