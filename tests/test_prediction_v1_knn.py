from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meseleo_core import (
    RADIO_MODE_ORDER,
    ConfidenceStatus,
    DatasetNotFoundError,
    DistributionStatus,
    InvalidDatasetSchemaError,
    LinkState,
    RadioMode,
    RadioScenario,
    V1Dataset,
    V1DatasetConfiguration,
    V1KnnConfiguration,
    V1KnnPredictionEngine,
)

FIXTURE_SHA256 = "474f20043c479852ae96a0e72f2fbc1b93323dad8e306c3e36fa36ac8e3779e6"


def test_actual_v1_fixture_dataset_loads(scientific_fixture_path: Path) -> None:
    dataset = V1Dataset(
        V1DatasetConfiguration(
            path=scientific_fixture_path,
            version="v1-scientific-fixture-1",
            sha256_digest=FIXTURE_SHA256,
        )
    )

    assert len(dataset.records) == 5
    assert dataset.digest_sha256 == FIXTURE_SHA256
    assert all(len(record.modes) == 8 for record in dataset.records)


def test_exact_match_predicts_all_eight_pooled_labels(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    predictions = v1_knn.predict(
        LinkState(elevation_deg=27.0, radial_speed_km_s=-2.5, active_nodes=10_000),
        radio_scenario,
        RADIO_MODE_ORDER,
    )
    by_mode = {item.radio_mode: item for item in predictions}

    assert tuple(item.radio_mode for item in predictions) == RADIO_MODE_ORDER
    assert by_mode[RadioMode.LORA_SF9].predicted_per == pytest.approx(0.1728395061728395)
    assert by_mode[RadioMode.LR_FHSS_DR9].predicted_per == 0.0
    assert by_mode[RadioMode.LORA_SF9].predicted_packets_per_window == 3_552.0
    assert by_mode[RadioMode.LORA_SF9].airtime_ms == pytest.approx(144.384)


def test_exact_match_uses_absolute_radial_speed(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    positive = v1_knn.predict(
        LinkState(elevation_deg=14.0, radial_speed_km_s=5.5, active_nodes=50_000),
        radio_scenario,
        RADIO_MODE_ORDER,
    )
    negative = v1_knn.predict(
        LinkState(elevation_deg=14.0, radial_speed_km_s=-5.5, active_nodes=50_000),
        radio_scenario,
        RADIO_MODE_ORDER,
    )

    assert positive == negative


def test_weighted_neighbor_interpolation_matches_v1_equation(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    state = LinkState(elevation_deg=35.0, radial_speed_km_s=3.2, active_nodes=30_000)
    predictions = v1_knn.predict(state, radio_scenario, RADIO_MODE_ORDER)

    query = np.asarray([35.0, 3.2, 30_000.0])
    normalized = (query - v1_knn.dataset.feature_mean) / v1_knn.dataset.feature_standard_deviation
    distances = np.linalg.norm(v1_knn.dataset.normalized_features - normalized, axis=1)
    indices = sorted(range(len(distances)), key=lambda index: (distances[index], index))[:3]
    weights = 1.0 / (distances[indices] + v1_knn.configuration.inverse_distance_epsilon)
    weights /= weights.sum()
    labels = np.asarray(
        [
            v1_knn.dataset.records[index].mode(RadioMode.LORA_SF8).per_percent / 100.0
            for index in indices
        ]
    )
    expected = float(np.dot(weights, labels))

    sf8 = next(item for item in predictions if item.radio_mode is RadioMode.LORA_SF8)
    assert sf8.predicted_per == pytest.approx(expected)


def test_v1_knn_populates_only_genuinely_derived_metrics(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    prediction = v1_knn.predict(
        LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000),
        radio_scenario,
        (RadioMode.LORA_SF7,),
    )[0]

    expected_goodput = 12 * 8 * (1.0 - prediction.predicted_per) / 0.041216
    expected_rf_energy = (10 ** (14.0 / 10.0)) * 0.041216
    assert prediction.predicted_goodput_bps == pytest.approx(expected_goodput)
    assert prediction.energy_per_attempt_millijoules == pytest.approx(expected_rf_energy)
    assert prediction.expected_delivery_latency_ms is None


def test_v1_knn_provenance_and_exact_match_confidence(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    prediction = v1_knn.predict(
        LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000),
        radio_scenario,
        (RadioMode.LR_FHSS_DR8,),
    )[0]
    metadata = {item.key: item.value for item in prediction.metadata}

    # An exact dataset match has one neighbour, so dispersion is zero: confidence 1.0.
    assert prediction.confidence == 1.0
    assert prediction.confidence_status is ConfidenceStatus.UNCALIBRATED
    assert metadata["confidence_method"] == "1-2*weighted_std(neighbor_per)"
    assert metadata["neighbor_per_weighted_std"] == 0.0
    assert metadata["dataset_version"] == "v1-scientific-fixture-1"
    assert metadata["dataset_sha256"] == FIXTURE_SHA256
    assert metadata["source_git_commit"] == "233e28634a70e1e0898ced684772f2b0e57502cb"
    assert metadata["state_feature_names"] == "elevation_deg,v_rel_kmps,n_nodes"
    assert metadata["neighbor_count"] == 1


def test_interpolated_confidence_reflects_neighbor_dispersion(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    state = LinkState(elevation_deg=35.0, radial_speed_km_s=3.2, active_nodes=30_000)
    predictions = v1_knn.predict(state, radio_scenario, RADIO_MODE_ORDER)

    query = np.asarray([35.0, 3.2, 30_000.0])
    normalized = (query - v1_knn.dataset.feature_mean) / v1_knn.dataset.feature_standard_deviation
    distances = np.linalg.norm(v1_knn.dataset.normalized_features - normalized, axis=1)
    indices = sorted(range(len(distances)), key=lambda index: (distances[index], index))[:3]
    weights = 1.0 / (distances[indices] + v1_knn.configuration.inverse_distance_epsilon)
    weights /= weights.sum()

    for prediction in predictions:
        labels = np.asarray(
            [
                v1_knn.dataset.records[index].mode(prediction.radio_mode).per_percent / 100.0
                for index in indices
            ]
        )
        mean = float(np.dot(weights, labels))
        std = float(np.sqrt(np.dot(weights, (labels - mean) ** 2)))
        expected = min(max(1.0 - 2.0 * std, 0.0), 1.0)

        assert prediction.confidence_status is ConfidenceStatus.UNCALIBRATED
        assert prediction.confidence == pytest.approx(expected)
        assert prediction.confidence is not None
        assert 0.0 <= prediction.confidence <= 1.0


def test_ood_status_reports_outside_fixture_domain(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    prediction = v1_knn.predict(
        LinkState(elevation_deg=90.0, radial_speed_km_s=8.0, active_nodes=200_000),
        radio_scenario,
        (RadioMode.LR_FHSS_DR8,),
    )[0]

    assert prediction.ood.status is DistributionStatus.OUT_OF_DISTRIBUTION
    assert prediction.ood.reasons


def test_missing_dataset_raises_structured_error(tmp_path: Path) -> None:
    configuration = V1KnnConfiguration(
        dataset=V1DatasetConfiguration(path=tmp_path / "missing.csv", version="missing")
    )

    with pytest.raises(DatasetNotFoundError):
        V1KnnPredictionEngine(configuration)


def test_malformed_dataset_raises_structured_error(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("elevation_deg,modulation\n10,SF7\n", encoding="utf-8")
    configuration = V1KnnConfiguration(
        dataset=V1DatasetConfiguration(path=malformed, version="malformed")
    )

    with pytest.raises(InvalidDatasetSchemaError, match="missing required columns"):
        V1KnnPredictionEngine(configuration)


def test_dataset_digest_mismatch_is_rejected(scientific_fixture_path: Path) -> None:
    configuration = V1KnnConfiguration(
        dataset=V1DatasetConfiguration(
            path=scientific_fixture_path,
            version="tampered",
            sha256_digest="0" * 64,
        )
    )

    with pytest.raises(InvalidDatasetSchemaError, match="SHA-256"):
        V1KnnPredictionEngine(configuration)
