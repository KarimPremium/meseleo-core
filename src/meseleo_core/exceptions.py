"""Typed failures raised before a trustworthy decision can be returned."""


class DecisionEngineError(ValueError):
    """Base class for exact-engine input and contract failures."""


class DuplicatePredictionError(DecisionEngineError):
    """Raised when more than one prediction exists for a radio mode."""


class UnsupportedConstraintError(DecisionEngineError):
    """Raised when the engine cannot evaluate a declared constraint."""


class FallbackUnavailableError(DecisionEngineError):
    """Raised when the configured safety fallback is not currently available."""
