"""Academic-V1 dataset, k-NN, RAG, and predictor-registry adapters."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .configuration import (
    PromptTechnique,
    V1DatasetConfiguration,
    V1KnnConfiguration,
    V1RagConfiguration,
)
from .integration_errors import (
    DatasetNotFoundError,
    InvalidDatasetSchemaError,
    MalformedLLMResponseError,
    MissingVectorStoreError,
    PredictionEngineUnavailableError,
    PredictionTimeoutError,
    RagProviderUnavailableError,
    UnsupportedPromptTechniqueError,
    UnsupportedRadioModeError,
)
from .models import (
    RADIO_MODE_ORDER,
    ConfidenceStatus,
    DistributionStatus,
    FrozenModel,
    LinkState,
    MetadataItem,
    OutOfDistributionResult,
    RadioMode,
    RadioPrediction,
    RadioScenario,
)

V1_REQUIRED_COLUMNS: tuple[str, ...] = (
    "elevation_deg",
    "v_rel_kmps",
    "n_nodes",
    "modulation",
    "PER_pct",
    "distance_km",
    "kappa",
    "doppler_hz",
    "doppler_rate_hz",
    "visibility_window_s",
    "RSSI_dBm",
    "SNR_dB",
    "toa_s",
    "max_packets_in_window",
)

_V1_TO_MODE: dict[str, RadioMode] = {
    "SF7": RadioMode.LORA_SF7,
    "SF8": RadioMode.LORA_SF8,
    "SF9": RadioMode.LORA_SF9,
    "SF10": RadioMode.LORA_SF10,
    "SF11": RadioMode.LORA_SF11,
    "SF12": RadioMode.LORA_SF12,
    "DR8": RadioMode.LR_FHSS_DR8,
    "DR9": RadioMode.LR_FHSS_DR9,
}
_MODE_TO_V1: dict[RadioMode, str] = {value: key for key, value in _V1_TO_MODE.items()}


@dataclass(frozen=True)
class V1ModeRecord:
    radio_mode: RadioMode
    per_percent: float
    airtime_seconds: float
    max_packets_per_window: float
    rssi_dbm: float
    snr_db: float


@dataclass(frozen=True)
class V1StateRecord:
    elevation_deg: float
    v_rel_km_s: float
    active_nodes: int
    distance_km: float
    kappa: float
    doppler_hz: float
    doppler_rate_hz_s: float
    visibility_window_seconds: float
    reference_rssi_dbm: float
    modes: tuple[V1ModeRecord, ...]

    @property
    def feature_vector(self) -> tuple[float, float, float]:
        return (self.elevation_deg, self.v_rel_km_s, float(self.active_nodes))

    def mode(self, radio_mode: RadioMode) -> V1ModeRecord:
        return next(item for item in self.modes if item.radio_mode is radio_mode)


class V1Dataset:
    """Validated immutable in-memory view of the V1 pooled-label CSV."""

    def __init__(self, configuration: V1DatasetConfiguration) -> None:
        self.configuration = configuration
        self.digest_sha256 = self._digest(configuration.path)
        if (
            configuration.sha256_digest is not None
            and configuration.sha256_digest != self.digest_sha256
        ):
            raise InvalidDatasetSchemaError(
                "V1 dataset SHA-256 does not match the configured provenance digest"
            )
        self.records = self._load(configuration.path)
        self.features = np.asarray([record.feature_vector for record in self.records], dtype=float)
        self.feature_mean = self.features.mean(axis=0)
        if len(self.records) > 1:
            standard_deviation = self.features.std(axis=0, ddof=1)
        else:
            standard_deviation = np.ones(3, dtype=float)
        self.feature_standard_deviation = np.where(
            standard_deviation == 0.0, 1.0, standard_deviation
        )
        self.normalized_features = (
            self.features - self.feature_mean
        ) / self.feature_standard_deviation
        # Data-driven OOD distance detector: the 99th percentile of each state's nearest-OTHER
        # standardized distance, scaled, flags queries far from any pooled state. None (report
        # UNKNOWN, never a false in-distribution) when there are too few states to estimate it.
        self.ood_distance_threshold: float | None = self._compute_ood_threshold()

    def _compute_ood_threshold(self) -> float | None:
        count = len(self.records)
        if count < 30:
            return None
        features = self.normalized_features
        differences = features[:, None, :] - features[None, :, :]
        distances = np.sqrt(np.sum(differences * differences, axis=2))
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        return float(np.percentile(nearest, 99.0) * 3.0)

    @staticmethod
    def _digest(path: Path) -> str:
        if not path.is_file():
            raise DatasetNotFoundError(f"V1 dataset not found: {path}")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as error:
            raise DatasetNotFoundError(f"cannot read V1 dataset: {path}") from error
        return digest.hexdigest()

    @staticmethod
    def _number(row: dict[str, str], column: str, line: int) -> float:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidDatasetSchemaError(
                f"invalid numeric value for {column!r} on CSV line {line}"
            ) from error
        if not math.isfinite(value):
            raise InvalidDatasetSchemaError(f"non-finite value for {column!r} on CSV line {line}")
        return value

    def _load(self, path: Path) -> tuple[V1StateRecord, ...]:
        if not path.is_file():
            raise DatasetNotFoundError(f"V1 dataset not found: {path}")
        try:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                columns = tuple(reader.fieldnames or ())
                missing = [column for column in V1_REQUIRED_COLUMNS if column not in columns]
                if missing:
                    raise InvalidDatasetSchemaError(
                        f"V1 dataset is missing required columns: {missing}"
                    )
                grouped: dict[tuple[float, float, int], list[tuple[int, dict[str, str]]]] = {}
                for line, row in enumerate(reader, start=2):
                    elevation = self._number(row, "elevation_deg", line)
                    radial_speed = abs(self._number(row, "v_rel_kmps", line))
                    nodes_number = self._number(row, "n_nodes", line)
                    if not nodes_number.is_integer() or nodes_number < 0:
                        raise InvalidDatasetSchemaError(
                            f"n_nodes must be a non-negative integer on CSV line {line}"
                        )
                    key = (elevation, radial_speed, int(nodes_number))
                    grouped.setdefault(key, []).append((line, row))
        except InvalidDatasetSchemaError:
            raise
        except OSError as error:
            raise DatasetNotFoundError(f"cannot read V1 dataset: {path}") from error

        records: list[V1StateRecord] = []
        expected_modes = frozenset(RADIO_MODE_ORDER)
        for key in sorted(grouped):
            rows = grouped[key]
            modes: list[V1ModeRecord] = []
            common: dict[str, float] = {}
            for line, row in rows:
                label = row["modulation"].strip().upper()
                try:
                    mode = _V1_TO_MODE[label]
                except KeyError as error:
                    raise UnsupportedRadioModeError(
                        f"unsupported V1 mode {label!r} on CSV line {line}"
                    ) from error
                per_percent = self._number(row, "PER_pct", line)
                airtime = self._number(row, "toa_s", line)
                packets = self._number(row, "max_packets_in_window", line)
                if not 0.0 <= per_percent <= 100.0:
                    raise InvalidDatasetSchemaError(
                        f"PER_pct must be in [0, 100] on CSV line {line}"
                    )
                if airtime <= 0.0 or packets < 0.0:
                    raise InvalidDatasetSchemaError(
                        f"airtime must be positive and packets non-negative on CSV line {line}"
                    )
                modes.append(
                    V1ModeRecord(
                        radio_mode=mode,
                        per_percent=per_percent,
                        airtime_seconds=airtime,
                        max_packets_per_window=packets,
                        rssi_dbm=self._number(row, "RSSI_dBm", line),
                        snr_db=self._number(row, "SNR_dB", line),
                    )
                )
                if not common:
                    common = {
                        "distance_km": self._number(row, "distance_km", line),
                        "kappa": self._number(row, "kappa", line),
                        "doppler_hz": self._number(row, "doppler_hz", line),
                        "doppler_rate_hz": self._number(row, "doppler_rate_hz", line),
                        "visibility_window_s": self._number(row, "visibility_window_s", line),
                        "reference_rssi_dbm": self._number(row, "RSSI_dBm", line),
                    }
            mode_set = frozenset(item.radio_mode for item in modes)
            if len(modes) != len(mode_set) or mode_set != expected_modes:
                raise InvalidDatasetSchemaError(
                    f"state {key!r} must contain each of the eight V1 modes exactly once"
                )
            ordered_modes = tuple(
                next(item for item in modes if item.radio_mode is mode) for mode in RADIO_MODE_ORDER
            )
            records.append(
                V1StateRecord(
                    elevation_deg=key[0],
                    v_rel_km_s=key[1],
                    active_nodes=key[2],
                    distance_km=common["distance_km"],
                    kappa=common["kappa"],
                    doppler_hz=common["doppler_hz"],
                    doppler_rate_hz_s=common["doppler_rate_hz"],
                    visibility_window_seconds=common["visibility_window_s"],
                    reference_rssi_dbm=common["reference_rssi_dbm"],
                    modes=ordered_modes,
                )
            )
        if not records:
            raise InvalidDatasetSchemaError("V1 dataset contains no complete states")
        return tuple(records)


def _derived_metrics(
    scenario: RadioScenario,
    predicted_per: float,
    airtime_seconds: float | None,
) -> tuple[float | None, float | None]:
    if airtime_seconds is None or airtime_seconds <= 0.0:
        return None, None
    goodput_bps = scenario.payload_bytes * 8.0 * (1.0 - predicted_per) / airtime_seconds
    transmit_power_mw = 10.0 ** (scenario.transmit_power_dbm / 10.0)
    transmit_energy_millijoules = transmit_power_mw * airtime_seconds
    return goodput_bps, transmit_energy_millijoules


class V1KnnPredictionEngine:
    """V1 paper baseline: z-scored three-axis inverse-distance k-NN."""

    def __init__(self, configuration: V1KnnConfiguration) -> None:
        self.configuration = configuration
        self.dataset = V1Dataset(configuration.dataset)

    def predict(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        radio_modes: tuple[RadioMode, ...],
    ) -> tuple[RadioPrediction, ...]:
        if len(radio_modes) != len(set(radio_modes)):
            raise InvalidDatasetSchemaError("requested radio modes cannot contain duplicates")
        forbidden = set(radio_modes) - set(self.configuration.allowed_radio_modes)
        if forbidden:
            names = ", ".join(sorted(mode.value for mode in forbidden))
            raise UnsupportedRadioModeError(f"V1 k-NN is not configured for: {names}")

        query = np.asarray(
            [
                link_state.elevation_deg,
                abs(link_state.radial_speed_km_s),
                float(link_state.active_nodes),
            ],
            dtype=float,
        )
        normalized_query = (
            query - self.dataset.feature_mean
        ) / self.dataset.feature_standard_deviation
        distances = np.linalg.norm(self.dataset.normalized_features - normalized_query, axis=1)
        ordered_indices = sorted(range(len(distances)), key=lambda index: (distances[index], index))
        nearest_index = ordered_indices[0]
        nearest_distance = float(distances[nearest_index])
        exact_match = nearest_distance <= self.configuration.exact_match_tolerance
        neighbor_indices: tuple[int, ...]
        if exact_match:
            neighbor_indices = (nearest_index,)
            weights = np.asarray([1.0], dtype=float)
        else:
            neighbor_indices = tuple(
                ordered_indices[: min(self.configuration.k, len(ordered_indices))]
            )
            selected_distances = distances[list(neighbor_indices)]
            weights = 1.0 / (selected_distances + self.configuration.inverse_distance_epsilon)
            weights = weights / weights.sum()

        ood = self._ood_result(query, nearest_distance)
        nearest_record = self.dataset.records[nearest_index]
        predictions: list[RadioPrediction] = []
        for mode in radio_modes:
            per_values = np.asarray(
                [
                    self.dataset.records[index].mode(mode).per_percent / 100.0
                    for index in neighbor_indices
                ],
                dtype=float,
            )
            # Clamp against floating-point overshoot: a weighted mean of ratios in [0,1] is
            # mathematically in [0,1], but e.g. averaging several 1.0 values can round to
            # 1.0000000000000002. This is numerical hygiene, not a change to the science.
            weighted_mean = float(np.dot(weights, per_values))
            predicted_per = min(max(weighted_mean, 0.0), 1.0)
            # P5-01: dispersion-based agreement score. The weighted standard deviation (std) of
            # neighbour PER ratios is at most 0.5 for values in [0, 1], so 1 - 2*std maps perfect
            # neighbour agreement (std = 0, including an exact match) to 1.0 and maximal
            # disagreement to 0.0. It is a heuristic, not a calibrated probability, hence
            # UNCALIBRATED: the decision engine keeps ignoring it for policy gates.
            weighted_std = math.sqrt(
                max(float(np.dot(weights, (per_values - weighted_mean) ** 2)), 0.0)
            )
            confidence = min(max(1.0 - 2.0 * weighted_std, 0.0), 1.0)
            ancillary = nearest_record.mode(mode)
            goodput, energy = _derived_metrics(
                scenario,
                predicted_per,
                ancillary.airtime_seconds,
            )
            predictions.append(
                RadioPrediction(
                    radio_mode=mode,
                    predicted_per=predicted_per,
                    predicted_goodput_bps=goodput,
                    predicted_packets_per_window=ancillary.max_packets_per_window,
                    airtime_ms=ancillary.airtime_seconds * 1000.0,
                    energy_per_attempt_millijoules=energy,
                    confidence=confidence,
                    confidence_status=ConfidenceStatus.UNCALIBRATED,
                    predictor_version=self.configuration.predictor_version,
                    ood=ood,
                    metadata=(
                        MetadataItem(
                            key="dataset_version",
                            value=self.configuration.dataset.version,
                        ),
                        MetadataItem(
                            key="dataset_sha256",
                            value=self.dataset.digest_sha256,
                        ),
                        MetadataItem(
                            key="source_git_commit",
                            value=self.configuration.dataset.source_git_commit,
                        ),
                        MetadataItem(key="knn_k", value=self.configuration.k),
                        MetadataItem(
                            key="state_feature_names",
                            value=",".join(self.configuration.state_feature_names),
                        ),
                        MetadataItem(key="neighbor_count", value=len(neighbor_indices)),
                        MetadataItem(key="exact_match", value=exact_match),
                        MetadataItem(key="nearest_distance_z", value=nearest_distance),
                        MetadataItem(
                            key="confidence_method",
                            value="1-2*weighted_std(neighbor_per)",
                        ),
                        MetadataItem(key="neighbor_per_weighted_std", value=weighted_std),
                        MetadataItem(
                            key="neighbor_state_indices",
                            value=",".join(str(index) for index in neighbor_indices),
                        ),
                        MetadataItem(
                            key="ancillary_metric_source",
                            value="nearest_v1_state",
                        ),
                        MetadataItem(
                            key="goodput_formula",
                            value="payload_bits*(1-PER)/airtime_s",
                        ),
                        MetadataItem(
                            key="energy_formula",
                            value="RF_power_mW*airtime_s",
                        ),
                    ),
                )
            )
        return tuple(predictions)

    def _ood_result(
        self, query: np.ndarray[Any, Any], nearest_distance: float
    ) -> OutOfDistributionResult:
        below = query < self.dataset.features.min(axis=0)
        above = query > self.dataset.features.max(axis=0)
        outside_bounds = bool(np.any(below | above))
        if outside_bounds:
            return OutOfDistributionResult(
                status=DistributionStatus.OUT_OF_DISTRIBUTION,
                score=1.0,
                reasons=("query is outside one or more V1 feature ranges",),
            )
        # Prefer an explicitly configured threshold; otherwise the data-driven one.
        threshold = self.configuration.ood_standardized_distance_threshold
        if threshold is None:
            threshold = self.dataset.ood_distance_threshold
        if threshold is None:
            # No distance-based detector is available: never claim in-distribution.
            return OutOfDistributionResult(
                status=DistributionStatus.INSUFFICIENT_DATA,
                reasons=("no distance-based OOD detector is available for this dataset",),
            )
        score = min(nearest_distance / threshold, 1.0)  # 1.0 means at/over the OOD boundary
        if nearest_distance > threshold:
            return OutOfDistributionResult(
                status=DistributionStatus.OUT_OF_DISTRIBUTION,
                score=1.0,
                reasons=("nearest standardized distance exceeds the OOD threshold",),
            )
        return OutOfDistributionResult(status=DistributionStatus.IN_DISTRIBUTION, score=score)


class RetrievedModeContext(FrozenModel):
    radio_mode: RadioMode
    per_percent: float | None = None
    airtime_seconds: float | None = None
    max_packets_per_window: float | None = None


class RetrievedContext(FrozenModel):
    context_id: str
    llm_text: str
    elevation_deg: float
    v_rel_km_s: float
    active_nodes: int
    distance: float | None = None
    modes: tuple[RetrievedModeContext, ...]

    def mode(self, radio_mode: RadioMode) -> RetrievedModeContext | None:
        return next((item for item in self.modes if item.radio_mode is radio_mode), None)


class RetrievalProvider(Protocol):
    def retrieve(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        k: int,
    ) -> tuple[RetrievedContext, ...]: ...


class LLMProvider(Protocol):
    def complete(self, prompt: str, timeout_seconds: float) -> str: ...


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbeddingProvider:
    """Lazy optional wrapper preserving the V1 Ollama embedding provider."""

    def __init__(self, model_name: str, base_url: str | None = None) -> None:
        self.model_name = model_name
        self.base_url = base_url

    def embed_query(self, text: str) -> list[float]:
        try:
            module = importlib.import_module("langchain_ollama")
            options = {"model": self.model_name}
            if self.base_url is not None:
                options["base_url"] = self.base_url
            return list(module.OllamaEmbeddings(**options).embed_query(text))
        except ImportError as error:
            raise RagProviderUnavailableError(
                "install the 'rag' optional dependencies for Ollama embeddings"
            ) from error
        except Exception as error:
            raise RagProviderUnavailableError(
                f"Ollama embedding model unavailable: {self.model_name}"
            ) from error


class OllamaLLMProvider:
    """Lazy optional wrapper preserving the V1 JSON ChatOllama provider."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        base_url: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url

    def complete(self, prompt: str, timeout_seconds: float) -> str:
        try:
            module = importlib.import_module("langchain_ollama")
            options: dict[str, Any] = {
                "model": self.model_name,
                "temperature": self.temperature,
                "format": "json",
                "client_kwargs": {"timeout": timeout_seconds},
            }
            if self.base_url is not None:
                options["base_url"] = self.base_url
            model = module.ChatOllama(
                **options,
            )
            response = model.invoke(prompt)
            content = response.content
            if not isinstance(content, str):
                raise TypeError("Ollama response content is not text")
            return content
        except ImportError as error:
            raise RagProviderUnavailableError(
                "install the 'rag' optional dependencies for ChatOllama"
            ) from error
        except TimeoutError as error:
            raise PredictionTimeoutError("Ollama prediction timed out") from error
        except (PredictionTimeoutError, RagProviderUnavailableError):
            raise
        except Exception as error:
            raise RagProviderUnavailableError(
                f"Ollama model unavailable: {self.model_name}"
            ) from error


class ChromaV1Retriever:
    """Lazy ChromaDB retrieval adapter for an existing V1 vector store."""

    def __init__(
        self,
        configuration: V1RagConfiguration,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.configuration = configuration
        self.embedding_provider = embedding_provider or OllamaEmbeddingProvider(
            configuration.embedding_model,
            configuration.ollama_base_url,
        )
        self.state_dataset = (
            V1Dataset(configuration.state_dataset)
            if configuration.state_dataset is not None
            else None
        )

    def retrieve(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        k: int,
    ) -> tuple[RetrievedContext, ...]:
        path = self.configuration.vector_store_path
        if not path.is_dir() or not any(path.iterdir()):
            raise MissingVectorStoreError(f"V1 Chroma vector store is missing or empty: {path}")
        try:
            chromadb = importlib.import_module("chromadb")
        except ImportError as error:
            raise RagProviderUnavailableError(
                "install the 'rag' optional dependencies for ChromaDB"
            ) from error
        query = self.build_query_text(link_state, scenario)
        vector = self.embedding_provider.embed_query(query)
        try:
            client = chromadb.PersistentClient(path=str(path))
            collection = client.get_collection(self.configuration.collection_name)
            if collection.count() == 0:
                raise MissingVectorStoreError(
                    f"V1 Chroma collection is empty: {self.configuration.collection_name}"
                )
            raw: Any = collection.query(query_embeddings=[vector], n_results=k)
        except MissingVectorStoreError:
            raise
        except Exception as error:
            raise MissingVectorStoreError(
                f"cannot query V1 Chroma collection {self.configuration.collection_name!r}"
            ) from error
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw.get("distances", [[None] * len(ids)])[0]
        contexts: list[RetrievedContext] = []
        for index, context_id in enumerate(ids):
            metadata = metadatas[index]
            modes = tuple(
                RetrievedModeContext(
                    radio_mode=mode,
                    per_percent=_optional_float(metadata.get(f"per_{_MODE_TO_V1[mode]}")),
                    airtime_seconds=_optional_float(metadata.get(f"toa_{_MODE_TO_V1[mode]}")),
                    max_packets_per_window=_optional_float(
                        metadata.get(f"max_packets_{_MODE_TO_V1[mode]}")
                    ),
                )
                for mode in RADIO_MODE_ORDER
            )
            contexts.append(
                RetrievedContext(
                    context_id=str(context_id),
                    llm_text=str(documents[index]),
                    elevation_deg=float(metadata["elevation_deg"]),
                    v_rel_km_s=float(metadata["v_rel_kmps"]),
                    active_nodes=int(metadata["n_nodes"]),
                    distance=_optional_float(distances[index]),
                    modes=modes,
                )
            )
        return tuple(contexts)

    def build_query_text(self, link_state: LinkState, scenario: RadioScenario) -> str:
        """Reproduce V1 ``agent_text_for_query`` using explicit dataset enrichment."""

        del scenario  # The historical V1 embedding text is carrier-independent.
        if self.state_dataset is None:
            raise PredictionEngineUnavailableError(
                "the real V1 Chroma retriever requires configuration.state_dataset "
                "to reproduce build_state enrichment"
            )
        same_density = tuple(
            record
            for record in self.state_dataset.records
            if record.active_nodes == link_state.active_nodes
        )
        candidates = same_density or self.state_dataset.records
        nearest_elevation = min(
            candidates,
            key=lambda record: (
                abs(record.elevation_deg - link_state.elevation_deg),
                record.elevation_deg,
            ),
        ).elevation_deg
        elevation_candidates = tuple(
            record for record in candidates if record.elevation_deg == nearest_elevation
        )
        record = min(
            elevation_candidates,
            key=lambda item: (
                abs(item.v_rel_km_s - abs(link_state.radial_speed_km_s)),
                item.v_rel_km_s,
            ),
        )
        elevation_bin = _classify_v1_bin(
            link_state.elevation_deg,
            (
                (0.0, 15.0, "extreme_low"),
                (15.0, 30.0, "low"),
                (30.0, 60.0, "mid"),
                (60.0, 91.0, "high"),
            ),
        )
        doppler_bin = _classify_v1_bin(
            abs(link_state.radial_speed_km_s),
            ((0.0, 2.0, "low"), (2.0, 5.0, "mid"), (5.0, 7.6, "extreme")),
        )
        density_bin = _classify_v1_bin(
            float(link_state.active_nodes),
            (
                (0.0, 50_000.0, "not_busy"),
                (50_000.0, 100_000.0, "busy"),
                (100_000.0, 1_000_000.0, "very_busy"),
            ),
        )
        rssi = record.reference_rssi_dbm
        snr_lora = rssi - _v1_noise_floor_dbm(
            self.configuration.lora_noise_bandwidth_hz,
            self.configuration.receiver_noise_figure_db,
        )
        snr_lr_fhss = rssi - _v1_noise_floor_dbm(
            self.configuration.lr_fhss_noise_bandwidth_hz,
            self.configuration.receiver_noise_figure_db,
        )
        return (
            f"LEO uplink at elevation {link_state.elevation_deg:g}° ({elevation_bin}), "
            f"relative speed {abs(link_state.radial_speed_km_s)} km/s "
            f"({doppler_bin} Doppler), network of {link_state.active_nodes} nodes "
            f"({density_bin}). Link: distance {record.distance_km:.0f} km, "
            f"kappa {record.kappa:.1f}, RSSI {rssi:.0f} dBm, "
            f"SNR LoRa {snr_lora:.0f} dB / LR-FHSS {snr_lr_fhss:.0f} dB."
        )


def _classify_v1_bin(
    value: float,
    bins: tuple[tuple[float, float, str], ...],
) -> str:
    for low, high, label in bins:
        if low <= value < high:
            return label
    return bins[-1][2]


def _v1_noise_floor_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    return -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MalformedLLMResponseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_v1_per_response(raw_text: str) -> dict[RadioMode, float]:
    """Parse exactly eight V1 PER percentages and return domain ratios."""

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start < 0 or end < start:
        raise MalformedLLMResponseError("LLM response contains no JSON object")
    try:
        parsed = json.loads(
            raw_text[start : end + 1],
            object_pairs_hook=_pairs_without_duplicates,
        )
    except MalformedLLMResponseError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise MalformedLLMResponseError("LLM response contains malformed JSON") from error
    if not isinstance(parsed, dict):
        raise MalformedLLMResponseError("LLM response JSON must be an object")
    expected = {f"per_{label.lower()}" for label in _V1_TO_MODE}
    present_per_keys = {str(key) for key in parsed if str(key).startswith("per_")}
    if present_per_keys != expected:
        missing = sorted(expected - present_per_keys)
        extra = sorted(present_per_keys - expected)
        raise MalformedLLMResponseError(
            f"LLM PER keys must contain all eight modes; missing={missing}, extra={extra}"
        )
    result: dict[RadioMode, float] = {}
    for label, mode in _V1_TO_MODE.items():
        value = parsed[f"per_{label.lower()}"]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MalformedLLMResponseError(f"PER for {label} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
            raise MalformedLLMResponseError(f"PER for {label} must be in [0, 100]")
        result[mode] = numeric / 100.0
    return result


def build_v1_rag_prompt(
    technique: PromptTechnique,
    link_state: LinkState,
    scenario: RadioScenario,
    contexts: tuple[RetrievedContext, ...],
) -> str:
    if technique not in set(PromptTechnique):
        raise UnsupportedPromptTechniqueError(f"unsupported V1 prompt technique: {technique}")
    context_lines: list[str] = []
    for index, context in enumerate(contexts, start=1):
        per_values = ", ".join(
            f"{_MODE_TO_V1[item.radio_mode]}_PER:{item.per_percent:.2f}%"
            for item in context.modes
            if item.per_percent is not None
        )
        context_lines.append(
            f"SIM {index} (EL={context.elevation_deg}deg, "
            f"v={context.v_rel_km_s}km/s, N={context.active_nodes}): "
            f"{context.llm_text} | {per_values}"
        )
    technique_instruction = {
        PromptTechnique.ZERO_SHOT: "Interpolate only from the retrieved simulations.",
        PromptTechnique.FEW_SHOT: (
            "Use the retrieved simulations as examples; output follows the same eight-key schema."
        ),
        PromptTechnique.COT: (
            "Reason internally over the retrieved simulations; an optional interpolation_logic "
            "string may precede the eight PER keys."
        ),
    }[technique]
    return (
        "You are the V1 MESELEO LEO performance predictor. "
        "Estimate Packet Error Rate percentages for all eight modes.\n"
        f"Technique: {technique.value}. {technique_instruction}\n"
        f"Current state: elevation={link_state.elevation_deg} deg, "
        f"radial_speed={abs(link_state.radial_speed_km_s)} km/s, "
        f"active_nodes={link_state.active_nodes}, RSSI={link_state.rssi_dbm}, "
        f"SNR={link_state.snr_db}.\n"
        f"Scenario: carrier={scenario.carrier_frequency_hz} Hz, "
        f"payload={scenario.payload_bytes} bytes, "
        f"transmit_power={scenario.transmit_power_dbm} dBm.\n"
        "Retrieved V1 simulations:\n"
        + "\n".join(context_lines)
        + "\nReturn only one JSON object with numeric keys per_sf7, per_sf8, per_sf9, "
        "per_sf10, per_sf11, per_sf12, per_dr8, per_dr9, each in [0,100]."
    )


class V1RagPredictionEngine:
    """V1 RAG PER adapter with injected retrieval and LLM providers."""

    def __init__(
        self,
        configuration: V1RagConfiguration,
        retrieval_provider: RetrievalProvider | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.configuration = configuration
        self.retrieval_provider = retrieval_provider or ChromaV1Retriever(configuration)
        self.llm_provider = llm_provider or OllamaLLMProvider(
            configuration.model_name,
            configuration.temperature,
            configuration.ollama_base_url,
        )
        self._last_retrieved_context: tuple[RetrievedContext, ...] = ()

    @property
    def last_retrieved_context(self) -> tuple[RetrievedContext, ...]:
        return self._last_retrieved_context

    def predict(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        radio_modes: tuple[RadioMode, ...],
    ) -> tuple[RadioPrediction, ...]:
        if tuple(radio_modes) != tuple(
            mode for mode in RADIO_MODE_ORDER if mode in set(radio_modes)
        ):
            raise UnsupportedRadioModeError(
                "V1 RAG requested modes must be unique and in deterministic radio-mode order"
            )
        try:
            contexts = self.retrieval_provider.retrieve(
                link_state,
                scenario,
                self.configuration.retrieval_k,
            )
        except (MissingVectorStoreError, RagProviderUnavailableError):
            raise
        except Exception as error:
            raise PredictionEngineUnavailableError("V1 RAG retrieval failed") from error
        if not contexts:
            raise MissingVectorStoreError("V1 RAG retrieval returned no context")
        self._last_retrieved_context = contexts
        prompt = build_v1_rag_prompt(
            self.configuration.prompt_technique,
            link_state,
            scenario,
            contexts,
        )
        try:
            raw = self.llm_provider.complete(prompt, self.configuration.timeout_seconds)
        except TimeoutError as error:
            raise PredictionTimeoutError("V1 RAG provider timed out") from error
        except (PredictionTimeoutError, RagProviderUnavailableError):
            raise
        except Exception as error:
            raise RagProviderUnavailableError("V1 RAG provider unavailable") from error
        per_by_mode = parse_v1_per_response(raw)
        nearest = contexts[0]
        predictions: list[RadioPrediction] = []
        for mode in radio_modes:
            ancillary = nearest.mode(mode)
            airtime = ancillary.airtime_seconds if ancillary is not None else None
            packets = ancillary.max_packets_per_window if ancillary is not None else None
            goodput, energy = _derived_metrics(scenario, per_by_mode[mode], airtime)
            predictions.append(
                RadioPrediction(
                    radio_mode=mode,
                    predicted_per=per_by_mode[mode],
                    predicted_goodput_bps=goodput,
                    predicted_packets_per_window=packets,
                    airtime_ms=airtime * 1000.0 if airtime is not None else None,
                    energy_per_attempt_millijoules=energy,
                    confidence=None,
                    confidence_status=ConfidenceStatus.UNAVAILABLE,
                    predictor_version=self.configuration.predictor_version,
                    ood=OutOfDistributionResult(
                        status=DistributionStatus.UNCERTAIN,
                        reasons=("V1 RAG has no calibrated OOD detector",),
                    ),
                    metadata=(
                        MetadataItem(
                            key="dataset_version",
                            value=self.configuration.dataset_version,
                        ),
                        MetadataItem(
                            key="state_dataset_sha256",
                            value=(
                                self.configuration.state_dataset.sha256_digest
                                if self.configuration.state_dataset is not None
                                else None
                            ),
                        ),
                        MetadataItem(
                            key="source_git_commit",
                            value=self.configuration.source_git_commit,
                        ),
                        MetadataItem(
                            key="prompt_technique",
                            value=self.configuration.prompt_technique.value,
                        ),
                        MetadataItem(key="model_provider", value=self.configuration.provider),
                        MetadataItem(key="model_name", value=self.configuration.model_name),
                        MetadataItem(
                            key="embedding_model",
                            value=self.configuration.embedding_model,
                        ),
                        MetadataItem(key="retrieval_k", value=self.configuration.retrieval_k),
                        MetadataItem(
                            key="retrieved_context_ids",
                            value=",".join(context.context_id for context in contexts),
                        ),
                        MetadataItem(
                            key="ancillary_metric_source",
                            value="nearest_retrieved_v1_context",
                        ),
                    ),
                )
            )
        return tuple(predictions)


class PredictorRegistry:
    """Small framework-independent registry for configured prediction engines."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Any]] = {}

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        if not name or name in self._factories:
            raise ValueError(f"predictor name must be non-empty and unique: {name!r}")
        self._factories[name] = factory

    def create(self, name: str) -> Any:
        try:
            return self._factories[name]()
        except KeyError as error:
            raise PredictionEngineUnavailableError(
                f"prediction engine is not registered: {name}"
            ) from error

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def build_v1_predictor_registry(
    knn: V1KnnPredictionEngine,
    rag_zero_shot: V1RagPredictionEngine,
    rag_few_shot: V1RagPredictionEngine,
    rag_cot: V1RagPredictionEngine,
) -> PredictorRegistry:
    registry = PredictorRegistry()
    registry.register("knn", lambda: knn)
    registry.register("rag-zero-shot", lambda: rag_zero_shot)
    registry.register("rag-few-shot", lambda: rag_few_shot)
    registry.register("rag-cot", lambda: rag_cot)
    return registry
