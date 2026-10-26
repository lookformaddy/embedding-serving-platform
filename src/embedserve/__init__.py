"""EmbedServe: a versioned embedding serving platform.

Pure-stdlib core (no network, no API keys). Optional FastAPI surface lives
in :mod:`embedserve.api` and is only imported when FastAPI is installed.
"""

from embedserve.models import ModelStage, ModelVersion
from embedserve.registry import EmbeddingModelRegistry

__all__ = ["ModelStage", "ModelVersion", "EmbeddingModelRegistry"]
__version__ = "0.1.0"
