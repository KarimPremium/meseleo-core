from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from meseleo_core import (
    BalancedWeights,
    ConfidenceStatus,
    ConstraintKind,
    ConstraintMetric,
    ConstraintOperator,
    ConstraintSatisfactionStatus,
    DecisionConstraint,
    DecisionResult,
    DeterministicDecisionEngine,
    DistributionStatus,
    EnergyMode,
    LinkState,
    LowElevationSafetyRule,
    MessagePriority,
    MissionPolicy,
    ModelVersion,
    ModeRejection,
    OperationalRequest,
    OptimizationObjective,
    OutOfDistributionResult,
    RadioMode,
    RadioPrediction,
    RadioScenario,
    ReasonCode,
    VisibilityWindow,
)


def _constraint(metric: ConstraintMetric, threshold: float, unit: str) -> DecisionConstraint:
    return DecisionConstraint(
        constraint_id=f"boundary-{metric.value}-{threshold}",
        kind=ConstraintKind.HARD,
        metric=metric,
        operator=ConstraintOperator.LTE,
        threshold=threshold,
        unit=unit,
    )


def test_invalid_physical_input_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LinkState(elevation_deg=90.1, radial_speed_km_s=0.0, active_nodes=1)
    with pytest.raises(ValidationError):
        LinkState(elevation_deg=10.0, radial_speed_km_s=0.0, active_nodes=-1)
    with pytest.raises(ValidationError):
        LinkState(elevation_deg=10.0, radial_speed_km_s=float("inf"), active_nodes=1)


def test_invalid_per_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RadioPrediction(
            radio_mode=RadioMode.LORA_SF7,
            predicted_per=1.01,
            predicted_goodput_bps=1.0,
            predictor_version=ModelVersion(name="predictor", version="1"),
        )


@pytest.mark.parametrize(
    ("metric", "unit", "threshold"),
    [
        (ConstraintMetric.PREDICTED_PER, "ratio", 0.0),
        (ConstraintMetric.PREDICTED_PER, "ratio", 1.0),
        (ConstraintMetric.CONFIDENCE, "ratio", 0.0),
        (ConstraintMetric.CONFIDENCE, "ratio", 1.0),
        (ConstraintMetric.PREDICTED_GOODPUT_BPS, "bit/s", 0.0),
        (ConstraintMetric.PREDICTED_PACKETS_PER_WINDOW, "packets/window", 0.0),
        (ConstraintMetric.EXPECTED_DELIVERY_LATENCY_MS, "ms", 0.0),
        (ConstraintMetric.ENERGY_PER_ATTEMPT_MILLIJOULES, "mJ/attempt", 0.0),
        (ConstraintMetric.ELEVATION_DEG, "deg", -90.0),
        (ConstraintMetric.ELEVATION_DEG, "deg", 90.0),
    ],
)
def test_decision_constraint_accepts_every_boundary_value(
    metric: ConstraintMetric,
    unit: str,
    threshold: float,
) -> None:
    assert _constraint(metric, threshold, unit).threshold == threshold


@pytest.mark.parametrize(
    ("metric", "unit", "threshold"),
    [
        (ConstraintMetric.PREDICTED_PER, "ratio", -0.0001),
        (ConstraintMetric.PREDICTED_PER, "ratio", 1.0001),
        (ConstraintMetric.CONFIDENCE, "ratio", -0.0001),
        (ConstraintMetric.CONFIDENCE, "ratio", 1.0001),
        (ConstraintMetric.PREDICTED_GOODPUT_BPS, "bit/s", -0.0001),
        (ConstraintMetric.PREDICTED_PACKETS_PER_WINDOW, "packets/window", -0.0001),
        (ConstraintMetric.EXPECTED_DELIVERY_LATENCY_MS, "ms", -0.0001),
        (ConstraintMetric.ENERGY_PER_ATTEMPT_MILLIJOULES, "mJ/attempt", -0.0001),
        (ConstraintMetric.ELEVATION_DEG, "deg", -90.0001),
        (ConstraintMetric.ELEVATION_DEG, "deg", 90.0001),
    ],
)
def test_decision_constraint_rejects_every_out_of_range_threshold(
    metric: ConstraintMetric,
    unit: str,
    threshold: float,
) -> None:
    with pytest.raises(ValidationError):
        _constraint(metric, threshold, unit)


def test_constraint_units_are_explicit_and_validated() -> None:
    with pytest.raises(ValidationError):
        _constraint(ConstraintMetric.EXPECTED_DELIVERY_LATENCY_MS, 10.0, "seconds")


def test_unsupported_optimization_objective_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MissionPolicy.model_validate(
            {
                "policy_name": "invalid",
                "version": {"name": "policy", "version": "1"},
                "message_priority": "routine",
                "maximum_admissible_per": 0.1,
                "optimization_objective": "maximize_throughput",
                "allowed_radio_modes": ["lr_fhss_dr8"],
                "fallback_mode": "lr_fhss_dr8",
            },
            strict=False,
        )


def test_allowed_radio_modes_serialize_in_deterministic_domain_order(
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    reversed_modes = frozenset(reversed(tuple(RadioMode)))
    expected = [mode.value for mode in RadioMode]
    request = OperationalRequest(
        message_priority=MessagePriority.ROUTINE,
        allowed_radio_modes=reversed_modes,
        fallback_mode=RadioMode.LR_FHSS_DR8,
    )
    policy = policy_factory(
        allowed_radio_modes=reversed_modes,
        source_request=request,
    )

    serialized = policy.model_dump(mode="json")
    assert serialized["allowed_radio_modes"] == expected
    assert serialized["source_request"]["allowed_radio_modes"] == expected
    restored = MissionPolicy.model_validate(serialized, strict=False)
    assert restored.model_dump(mode="json") == serialized


def test_balanced_objective_requires_explicit_weights() -> None:
    with pytest.raises(ValidationError):
        MissionPolicy(
            policy_name="balanced",
            version=ModelVersion(name="policy", version="1"),
            message_priority=MessagePriority.IMPORTANT,
            maximum_admissible_per=0.1,
            optimization_objective=OptimizationObjective.BALANCED_WEIGHTED_SCORE,
            allowed_radio_modes=frozenset({RadioMode.LR_FHSS_DR8}),
            fallback_mode=RadioMode.LR_FHSS_DR8,
        )


def test_balanced_weights_require_a_positive_policy_choice() -> None:
    with pytest.raises(ValidationError):
        BalancedWeights(
            goodput_bps=0.0,
            packets_per_window=0.0,
            reliability=0.0,
            expected_delivery_latency=0.0,
            energy_per_attempt=0.0,
        )


def test_visibility_window_and_operational_deadline_require_timezone_awareness() -> None:
    start = datetime(2026, 7, 13, tzinfo=UTC)
    window = VisibilityWindow(start=start, end=start + timedelta(minutes=5))
    state = LinkState(
        elevation_deg=5.0,
        radial_speed_km_s=-6.5,
        active_nodes=0,
        timestamp=start,
        visibility_window=window,
    )
    request = OperationalRequest(
        message_priority=MessagePriority.URGENT,
        deadline=start + timedelta(minutes=2),
        delivery_reliability_target=0.999,
        energy_mode=EnergyMode.CONSERVE_ENERGY,
    )
    assert state.visibility_window == window
    assert request.deadline is not None

    with pytest.raises(ValidationError):
        VisibilityWindow(
            start=datetime(2026, 7, 13),
            end=datetime(2026, 7, 13, 0, 5),
        )
    with pytest.raises(ValidationError):
        OperationalRequest(
            message_priority=MessagePriority.ROUTINE,
            deadline=datetime(2026, 7, 13),
        )


def test_prediction_context_is_explicit_and_immutable() -> None:
    scenario = RadioScenario(
        carrier_frequency_hz=868_100_000.0,
        payload_bytes=32,
        transmit_power_dbm=14.0,
        bandwidth_hz=125_000.0,
        channel_plan="EU868",
        device_profile_id="sensor-v2",
    )
    assert scenario.payload_bytes == 32
    with pytest.raises(ValidationError):
        scenario.payload_bytes = 16
    with pytest.raises(ValidationError):
        RadioScenario(
            carrier_frequency_hz=0.0,
            payload_bytes=1,
            transmit_power_dbm=14.0,
        )


def test_confidence_status_never_invents_or_mislabels_a_value() -> None:
    version = ModelVersion(name="predictor", version="1")
    unavailable = RadioPrediction(
        radio_mode=RadioMode.LR_FHSS_DR8,
        predicted_per=0.01,
        predictor_version=version,
    )
    assert unavailable.confidence is None
    assert unavailable.confidence_status is ConfidenceStatus.UNAVAILABLE

    with pytest.raises(ValidationError):
        RadioPrediction(
            radio_mode=RadioMode.LR_FHSS_DR8,
            predicted_per=0.01,
            confidence=0.8,
            confidence_status=ConfidenceStatus.UNAVAILABLE,
            predictor_version=version,
        )
    with pytest.raises(ValidationError):
        RadioPrediction(
            radio_mode=RadioMode.LR_FHSS_DR8,
            predicted_per=0.01,
            confidence=None,
            confidence_status=ConfidenceStatus.CALIBRATED,
            predictor_version=version,
        )


def test_ood_status_is_the_only_serialized_source_of_truth() -> None:
    prediction = RadioPrediction(
        radio_mode=RadioMode.LR_FHSS_DR8,
        predicted_per=0.01,
        predictor_version=ModelVersion(name="predictor", version="1"),
        ood=OutOfDistributionResult(status=DistributionStatus.OUT_OF_DISTRIBUTION),
    )
    serialized = prediction.model_dump()
    assert prediction.is_out_of_distribution is True
    assert "out_of_distribution" not in serialized
    assert serialized["ood"]["status"] is DistributionStatus.OUT_OF_DISTRIBUTION


def test_legacy_metric_input_aliases_serialize_only_canonical_names() -> None:
    prediction = RadioPrediction.model_validate(
        {
            "radio_mode": RadioMode.LORA_SF7,
            "predicted_per": 0.1,
            "predicted_throughput_bps": 100.0,
            "latency_ms": 20.0,
            "energy_millijoules": 5.0,
            "predictor_version": ModelVersion(name="legacy", version="1"),
        }
    )
    serialized = prediction.model_dump()
    assert serialized["predicted_goodput_bps"] == 100.0
    assert serialized["expected_delivery_latency_ms"] == 20.0
    assert serialized["energy_per_attempt_millijoules"] == 5.0
    assert "predicted_throughput_bps" not in serialized
    assert "latency_ms" not in serialized
    assert "energy_millijoules" not in serialized


def test_models_are_immutable(link_state: LinkState) -> None:
    with pytest.raises(ValidationError):
        link_state.elevation_deg = 10.0


def test_low_elevation_rule_must_force_an_allowed_mode() -> None:
    with pytest.raises(ValidationError):
        MissionPolicy(
            policy_name="invalid-low-rule",
            version=ModelVersion(name="policy", version="1"),
            message_priority=MessagePriority.CRITICAL,
            maximum_admissible_per=0.1,
            optimization_objective=OptimizationObjective.MAXIMIZE_GOODPUT_BPS,
            allowed_radio_modes=frozenset({RadioMode.LR_FHSS_DR9}),
            fallback_mode=RadioMode.LR_FHSS_DR9,
            low_elevation_safety=LowElevationSafetyRule(
                below_elevation_deg=15.0,
                forced_mode=RadioMode.LR_FHSS_DR8,
            ),
        )


def test_mode_rejection_rejects_duplicate_reasons_and_constraint_ids() -> None:
    with pytest.raises(ValidationError):
        ModeRejection(
            radio_mode=RadioMode.LORA_SF7,
            reason_codes=(ReasonCode.MISSING_CONSTRAINT_DATA,) * 2,
        )
    with pytest.raises(ValidationError):
        ModeRejection(
            radio_mode=RadioMode.LORA_SF7,
            reason_codes=(ReasonCode.MISSING_CONSTRAINT_DATA,),
            constraint_ids=("latency", "latency"),
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"reason_codes": (ReasonCode.MODE_SELECTED, ReasonCode.MODE_SELECTED)},
        {"fallback_applied": True},
        {"constraint_satisfaction_status": ConstraintSatisfactionStatus.FALLBACK_UNVERIFIED},
        {"selected_prediction": None},
        {"admissible_modes": ()},
    ],
)
def test_decision_result_rejects_inconsistent_state(
    updates: dict[str, object],
    link_state: LinkState,
    predictions: tuple[RadioPrediction, ...],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    result = DeterministicDecisionEngine().decide(link_state, policy_factory(), predictions)
    values = result.model_dump()
    values.update(updates)
    with pytest.raises(ValidationError):
        DecisionResult.model_validate(values)
