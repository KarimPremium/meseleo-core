"""Explicit immutable configuration for V1 integration adapters."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, model_validator

from .models import FrozenModel, ModelVersion, RadioMode

V1_GIT_COMMIT = "233e28634a70e1e0898ced684772f2b0e57502cb"


class V1GeometryConfiguration(FrozenModel):
    adapter_version: ModelVersion = ModelVersion(name="v1-skyfield-geometry", version="2.0.0")
    speed_of_light_m_s: Annotated[float, Field(gt=0.0)] = 299_792_458.0


class V1DatasetConfiguration(FrozenModel):
    path: Path
    version: Annotated[str, Field(min_length=1)]
    sha256_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    source_git_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] = V1_GIT_COMMIT


class V1KnnConfiguration(FrozenModel):
    dataset: V1DatasetConfiguration
    predictor_version: ModelVersion = ModelVersion(name="v1-state-space-knn", version="2.0.0")
    state_feature_names: tuple[str, str, str] = (
        "elevation_deg",
        "v_rel_kmps",
        "n_nodes",
    )
    k: Annotated[int, Field(ge=1)] = 3
    inverse_distance_epsilon: Annotated[float, Field(gt=0.0)] = 1e-9
    exact_match_tolerance: Annotated[float, Field(ge=0.0)] = 1e-12
    allowed_radio_modes: frozenset[RadioMode] = frozenset(RadioMode)
    ood_standardized_distance_threshold: Annotated[float, Field(gt=0.0)] | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        expected = ("elevation_deg", "v_rel_kmps", "n_nodes")
        if self.state_feature_names != expected:
            raise ValueError(f"V1 k-NN features must remain {expected!r}")
        if not self.allowed_radio_modes:
            raise ValueError("allowed_radio_modes cannot be empty")
        return self


class PromptTechnique(StrEnum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    COT = "cot"


class V1RagConfiguration(FrozenModel):
    provider: Annotated[str, Field(min_length=1)] = "ollama"
    ollama_base_url: Annotated[str, Field(pattern=r"^https?://[^\s]+$")] = "http://localhost:11434"
    model_name: Annotated[str, Field(min_length=1)] = "llama3.2:3b"
    embedding_model: Annotated[str, Field(min_length=1)] = "nomic-embed-text"
    vector_store_path: Path
    state_dataset: V1DatasetConfiguration | None = None
    collection_name: Annotated[str, Field(min_length=1)] = "leo_modulation_rag"
    retrieval_k: Annotated[int, Field(ge=1)] = 3
    prompt_technique: PromptTechnique = PromptTechnique.ZERO_SHOT
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0
    timeout_seconds: Annotated[float, Field(gt=0.0)] = 30.0
    dataset_version: Annotated[str, Field(min_length=1)]
    predictor_version: ModelVersion = ModelVersion(name="v1-rag-per", version="2.0.0")
    source_git_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] = V1_GIT_COMMIT
    receiver_noise_figure_db: float = 6.0
    lora_noise_bandwidth_hz: Annotated[float, Field(gt=0.0)] = 125_000.0
    lr_fhss_noise_bandwidth_hz: Annotated[float, Field(gt=0.0)] = 488.0


class CommandMapping(FrozenModel):
    radio_mode: RadioMode
    command: Annotated[str, Field(min_length=1)]


V1_COMMAND_MAPPINGS: tuple[CommandMapping, ...] = (
    CommandMapping(radio_mode=RadioMode.LORA_SF7, command="AT+MOD=LORA-SF7"),
    CommandMapping(radio_mode=RadioMode.LORA_SF8, command="AT+MOD=LORA-SF8"),
    CommandMapping(radio_mode=RadioMode.LORA_SF9, command="AT+MOD=LORA-SF9"),
    CommandMapping(radio_mode=RadioMode.LORA_SF10, command="AT+MOD=LORA-SF10"),
    CommandMapping(radio_mode=RadioMode.LORA_SF11, command="AT+MOD=LORA-SF11"),
    CommandMapping(radio_mode=RadioMode.LORA_SF12, command="AT+MOD=LORA-SF12"),
    CommandMapping(radio_mode=RadioMode.LR_FHSS_DR8, command="AT+MOD=LR-FHSS-DR8"),
    CommandMapping(radio_mode=RadioMode.LR_FHSS_DR9, command="AT+MOD=LR-FHSS-DR9"),
)


class V1CommandConfiguration(FrozenModel):
    adapter_version: ModelVersion = ModelVersion(name="v1-at-command", version="2.0.0")
    modem_profile: Annotated[str, Field(min_length=1)] = "v1-generic-at"
    mappings: tuple[CommandMapping, ...] = V1_COMMAND_MAPPINGS

    @model_validator(mode="after")
    def validate_mappings(self) -> Self:
        modes = tuple(item.radio_mode for item in self.mappings)
        if len(modes) != len(set(modes)):
            raise ValueError("command mappings cannot repeat a radio mode")
        return self
