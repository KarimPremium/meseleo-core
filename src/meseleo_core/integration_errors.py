"""Structured failures at academic-V1 integration boundaries."""


class MeseleoIntegrationError(RuntimeError):
    """Base error for a failed external/scientific adapter boundary."""


class DatasetNotFoundError(MeseleoIntegrationError):
    """The configured V1 dataset does not exist."""


class InvalidDatasetSchemaError(MeseleoIntegrationError):
    """The configured dataset cannot satisfy the V1 state/mode contract."""


class UnsupportedRadioModeError(MeseleoIntegrationError):
    """A V1 mode label cannot be mapped to a core radio mode."""


class MissingTLELinesError(MeseleoIntegrationError):
    """A TLE request lacks valid line 1 or line 2 data."""


class InvalidGroundStationError(MeseleoIntegrationError):
    """A ground-station input cannot be used by the geometry adapter."""


class GeometryComputationError(MeseleoIntegrationError):
    """Skyfield/SGP4 could not produce a trusted geometry result."""


class PredictionEngineUnavailableError(MeseleoIntegrationError):
    """A configured prediction engine or optional dependency is unavailable."""


class RagProviderUnavailableError(PredictionEngineUnavailableError):
    """The configured RAG LLM or embedding provider is unavailable."""


class MalformedLLMResponseError(MeseleoIntegrationError):
    """An LLM response violates the strict eight-mode PER schema."""


class MissingVectorStoreError(PredictionEngineUnavailableError):
    """The configured persistent vector store is absent or empty."""


class UnsupportedPromptTechniqueError(MeseleoIntegrationError):
    """The requested V1 RAG prompt technique is unsupported."""


class IncompletePredictionSetError(MeseleoIntegrationError):
    """A pipeline predictor did not return exactly one result per requested mode."""


class CommandMappingUnavailableError(MeseleoIntegrationError):
    """No validated modem command exists for the selected mode."""


class TransmissionIneligibleError(MeseleoIntegrationError):
    """Command generation was requested without validated transmission eligibility."""


class PredictionTimeoutError(PredictionEngineUnavailableError):
    """A prediction provider exceeded its configured timeout."""
