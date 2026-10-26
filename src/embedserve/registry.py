"""Versioned embedding-model registry with a strict stage machine.

Allowed transitions (nothing else is legal, which is the whole point --
an embedding change silently shipped to production invalidates every
stored vector built with the old model):

    draft -> staging -> production -> deprecated
    staging -> draft   (send a bad candidate back for rework)
    production -> staging (rollback path, deliberately allowed)

Deprecation is terminal. Registering a model whose dimension differs
from an already-production version of the same name is allowed but
flagged by :meth:`dimension_changed`, so callers can re-index first.
"""

from __future__ import annotations

from embedserve.errors import InvalidStageTransition, RegistryError
from embedserve.models import ModelStage, ModelVersion

_ALLOWED: dict[ModelStage, set[ModelStage]] = {
    ModelStage.DRAFT: {ModelStage.STAGING},
    ModelStage.STAGING: {ModelStage.PRODUCTION, ModelStage.DRAFT},
    ModelStage.PRODUCTION: {ModelStage.DEPRECATED, ModelStage.STAGING},
    ModelStage.DEPRECATED: set(),
}


class EmbeddingModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelVersion] = {}

    def register(self, model: ModelVersion) -> ModelVersion:
        if model.key in self._models:
            raise RegistryError(f"model already registered: {model.key}")
        self._models[model.key] = model
        return model

    def get(self, name: str, version: str) -> ModelVersion:
        key = f"{name}:{version}"
        try:
            return self._models[key]
        except KeyError as exc:
            raise RegistryError(f"unknown model: {key}") from exc

    def transition(self, name: str, version: str, to: ModelStage) -> ModelVersion:
        model = self.get(name, version)
        if to not in _ALLOWED[model.stage]:
            raise InvalidStageTransition(f"{model.stage.value} -> {to.value} is not allowed")
        updated = ModelVersion(
            name=model.name, version=model.version, dimension=model.dimension,
            normalization=model.normalization, pooling=model.pooling,
            stage=to, description=model.description,
        )
        self._models[model.key] = updated
        return updated

    def production_model(self, name: str) -> ModelVersion:
        for model in self._models.values():
            if model.name == name and model.stage == ModelStage.PRODUCTION:
                return model
        raise RegistryError(f"no production model for {name}")

    def dimension_changed(self, name: str, version: str) -> bool:
        """True if this version's dimension differs from the production one."""
        try:
            prod = self.production_model(name)
        except RegistryError:
            return False
        return self.get(name, version).dimension != prod.dimension

    def list(self, name: str | None = None) -> list[ModelVersion]:
        models = sorted(self._models.values(), key=lambda m: m.key)
        return [m for m in models if name is None or m.name == name]
