from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from meseleo_core import (
    ConfidenceStatus,
    DistributionStatus,
    GroundStation,
    LinkState,
    MessagePriority,
    MissionPolicy,
    ModelVersion,
    OptimizationObjective,
    OutOfDistributionResult,
    RadioMode,
    RadioPrediction,
    RadioScenario,
    TLEInput,
    UncertaintyAction,
    V1DatasetConfiguration,
    V1KnnConfiguration,
    V1KnnPredictionEngine,
)

PREDICTOR_VERSION = ModelVersion(name="fixture-predictor", version="1.0.0")
POLICY_VERSION = ModelVersion(name="fixture-policy", version="1.0.0")


@pytest.fixture(scope="session")
def scientific_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "v1_scientific_fixture.csv"


@pytest.fixture(scope="session")
def v1_knn(scientific_fixture_path: Path) -> V1KnnPredictionEngine:
    dataset = V1DatasetConfiguration(
        path=scientific_fixture_path,
        version="v1-scientific-fixture-1",
        sha256_digest="474f20043c479852ae96a0e72f2fbc1b93323dad8e306c3e36fa36ac8e3779e6",
    )
    # The fixture has too few states to auto-estimate an OOD detector, so configure a threshold
    # explicitly: its states are genuine in-grid pooled states and must read in-distribution for
    # the decision-logic reference scenarios (the production dataset auto-derives its own detector).
    return V1KnnPredictionEngine(
        V1KnnConfiguration(dataset=dataset, ood_standardized_distance_threshold=3.0)
    )


@pytest.fixture(scope="session")
def radio_scenario() -> RadioScenario:
    return RadioScenario(
        carrier_frequency_hz=868_000_000.0,
        payload_bytes=12,
        transmit_power_dbm=14.0,
        bandwidth_hz=125_000.0,
        channel_plan="EU868",
    )


@pytest.fixture(scope="session")
def fixed_tle() -> TLEInput:
    return TLEInput(
        satellite_id="ISS-V1-FIXTURE",
        line1="1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9000",
        line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49114000563537",
        source="academic-v1-dashboard-fixture",
        epoch_provenance="TLE epoch 2024-001.5 UTC",
    )


@pytest.fixture(scope="session")
def paris_station() -> GroundStation:
    return GroundStation(
        ground_station_id="paris-v1-fixture",
        latitude_deg=48.8566,
        longitude_deg=2.3522,
        altitude_m=35.0,
    )


@pytest.fixture
def link_state() -> LinkState:
    return LinkState(
        elevation_deg=45.0,
        radial_speed_km_s=4.0,
        active_nodes=50_000,
        rssi_dbm=-121.0,
        snr_db=-8.0,
        doppler_hz=11_573.0,
    )


@pytest.fixture
def prediction_factory() -> Callable[..., RadioPrediction]:
    def build(
        mode: RadioMode,
        *,
        per: float = 0.01,
        goodput_bps: float | None = 1_000.0,
        packets_per_window: float | None = 10.0,
        expected_delivery_latency_ms: float | None = 50.0,
        energy_per_attempt_millijoules: float | None = 10.0,
        confidence: float | None = 0.95,
        confidence_status: ConfidenceStatus | None = None,
        distribution_status: DistributionStatus = DistributionStatus.IN_DISTRIBUTION,
        predictor_version: ModelVersion = PREDICTOR_VERSION,
    ) -> RadioPrediction:
        resolved_confidence_status = confidence_status or (
            ConfidenceStatus.UNAVAILABLE if confidence is None else ConfidenceStatus.CALIBRATED
        )
        return RadioPrediction(
            radio_mode=mode,
            predicted_per=per,
            predicted_goodput_bps=goodput_bps,
            predicted_packets_per_window=packets_per_window,
            expected_delivery_latency_ms=expected_delivery_latency_ms,
            energy_per_attempt_millijoules=energy_per_attempt_millijoules,
            confidence=confidence,
            confidence_status=resolved_confidence_status,
            predictor_version=predictor_version,
            ood=OutOfDistributionResult(status=distribution_status),
        )

    return build


@pytest.fixture
def predictions(
    prediction_factory: Callable[..., RadioPrediction],
) -> tuple[RadioPrediction, ...]:
    values = (
        (RadioMode.LORA_SF7, 0.20, 5_000.0, 8.0),
        (RadioMode.LORA_SF8, 0.05, 3_000.0, 10.0),
        (RadioMode.LORA_SF9, 0.02, 1_800.0, 12.0),
        (RadioMode.LORA_SF10, 0.01, 900.0, 14.0),
        (RadioMode.LORA_SF11, 0.008, 450.0, 18.0),
        (RadioMode.LORA_SF12, 0.005, 200.0, 24.0),
        (RadioMode.LR_FHSS_DR8, 0.002, 120.0, 20.0),
        (RadioMode.LR_FHSS_DR9, 0.01, 300.0, 15.0),
    )
    return tuple(
        prediction_factory(
            mode,
            per=per,
            goodput_bps=throughput,
            energy_per_attempt_millijoules=energy,
        )
        for mode, per, throughput, energy in values
    )


@pytest.fixture
def policy_factory() -> Callable[..., MissionPolicy]:
    def build(**overrides: object) -> MissionPolicy:
        values: dict[str, object] = {
            "policy_name": "fixture-policy",
            "version": POLICY_VERSION,
            "message_priority": MessagePriority.ROUTINE,
            "maximum_admissible_per": 0.10,
            "optimization_objective": OptimizationObjective.MAXIMIZE_GOODPUT_BPS,
            "allowed_radio_modes": frozenset(RadioMode),
            "fallback_mode": RadioMode.LR_FHSS_DR8,
            "uncertainty_action": UncertaintyAction.REJECT,
        }
        values.update(overrides)
        return MissionPolicy(**values)  # type: ignore[arg-type]

    return build
