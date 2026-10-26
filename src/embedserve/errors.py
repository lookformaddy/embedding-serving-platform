"""Domain errors for EmbedServe."""


class EmbedServeError(Exception):
    """Base error."""


class RegistryError(EmbedServeError):
    """Invalid registry operation (duplicate, unknown model, bad transition)."""


class InvalidStageTransition(RegistryError):
    """Raised when a stage move is not in the allowed state machine."""


class QuotaExceeded(EmbedServeError):
    """Per-tenant rate limit or token budget exhausted."""


class UnknownModel(EmbedServeError):
    """No model/version matches the requested identifier."""


class StaleVersionError(EmbedServeError):
    """A write targeted a version that has been cleaned up or superseded."""
