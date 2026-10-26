import pytest

from embedserve.errors import InvalidStageTransition, RegistryError
from embedserve.models import ModelStage, ModelVersion
from embedserve.registry import EmbeddingModelRegistry


def _reg():
    reg = EmbeddingModelRegistry()
    reg.register(ModelVersion(name="m", version="1.0", dimension=32))
    return reg


def test_full_stage_lifecycle():
    reg = _reg()
    assert reg.transition("m", "1.0", ModelStage.STAGING).stage == ModelStage.STAGING
    assert reg.transition("m", "1.0", ModelStage.PRODUCTION).stage == ModelStage.PRODUCTION
    assert reg.transition("m", "1.0", ModelStage.DEPRECATED).stage == ModelStage.DEPRECATED


def test_illegal_transition_rejected():
    reg = _reg()
    with pytest.raises(InvalidStageTransition):
        reg.transition("m", "1.0", ModelStage.PRODUCTION)
    with pytest.raises(InvalidStageTransition):
        reg.transition("m", "1.0", ModelStage.DEPRECATED)


def test_deprecated_is_terminal():
    reg = _reg()
    for stage in (ModelStage.STAGING, ModelStage.PRODUCTION):
        reg.transition("m", "1.0", stage)
    reg.transition("m", "1.0", ModelStage.DEPRECATED)
    with pytest.raises(InvalidStageTransition):
        reg.transition("m", "1.0", ModelStage.STAGING)


def test_production_rollback_to_staging_allowed():
    reg = _reg()
    reg.transition("m", "1.0", ModelStage.STAGING)
    reg.transition("m", "1.0", ModelStage.PRODUCTION)
    assert reg.transition("m", "1.0", ModelStage.STAGING).stage == ModelStage.STAGING


def test_duplicate_and_unknown_rejected():
    reg = _reg()
    with pytest.raises(RegistryError):
        reg.register(ModelVersion(name="m", version="1.0", dimension=32))
    with pytest.raises(RegistryError):
        reg.get("m", "9.9")


def test_dimension_change_flag():
    reg = _reg()
    reg.transition("m", "1.0", ModelStage.STAGING)
    reg.transition("m", "1.0", ModelStage.PRODUCTION)
    reg.register(ModelVersion(name="m", version="2.0", dimension=64))
    assert reg.dimension_changed("m", "2.0") is True
    assert reg.dimension_changed("m", "1.0") is False
    assert reg.production_model("m").version == "1.0"
