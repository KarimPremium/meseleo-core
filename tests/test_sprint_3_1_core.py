from __future__ import annotations

from datetime import UTC, datetime

import pytest
from tests.test_pipeline_v2 import build_pipeline

from meseleo_core import (
    ConstraintEvaluationStatus,
    ConstraintKind,
    ConstraintMetric,
    ConstraintOperator,
    DecisionConstraint,
    DefaultPolicyEngine,
    DeterministicDecisionEngine,
    EnergyMode,
    GroundStation,
    LinkAvailabilityStatus,
    LinkState,
    MessagePriority,
    MissionPolicy,
    ModelVersion,
    OperationalRequest,
    OptimizationObjective,
    ProfileSelectionMode,
    RadioMode,
    RadioScenario,
    TLEInput,
    TLEStateRequest,
    TransmissionEligibilityReason,
    V1KnnPredictionEngine,
)


@pytest.mark.parametrize(
    ("priority", "energy_mode", "expected_profile"),
    (
        (MessagePriority.ROUTINE, None, "routine_telemetry"),
        (MessagePriority.IMPORTANT, None, "reliable_telemetry"),
        (MessagePriority.URGENT, None, "reliable_telemetry"),
        (MessagePriority.ROUTINE, EnergyMode.CONSERVE_ENERGY, "battery_saving"),
        (MessagePriority.CRITICAL, EnergyMode.CONSERVE_ENERGY, "critical_alert"),
    ),
)
def test_automatic_policy_resolution_is_deterministic(
    priority: MessagePriority,
    energy_mode: EnergyMode | None,
    expected_profile: str,
) -> None:
    policy = DefaultPolicyEngine().compile_policy(
        OperationalRequest(message_priority=priority, energy_mode=energy_mode),
        LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=10),
    )

    assert policy.policy_name == expected_profile
    assert policy.policy_resolution is not None
    assert policy.policy_resolution.requested_mode is ProfileSelectionMode.AUTOMATIC
    assert policy.policy_resolution.resolved_profile == expected_profile


def test_manual_policy_remains_authoritative_and_warns_on_contradiction() -> None:
    policy = DefaultPolicyEngine().compile_policy(
        OperationalRequest(
            message_priority=MessagePriority.CRITICAL,
            profile_selection_mode=ProfileSelectionMode.MANUAL,
            policy_profile="routine_telemetry",
        ),
        LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=10),
    )

    assert policy.policy_name == "routine_telemetry"
    assert policy.policy_resolution is not None
    assert policy.policy_resolution.warnings
    assert "critical_alert" in policy.policy_resolution.warnings[0]


def test_profile_per_default_and_stricter_override() -> None:
    engine = DefaultPolicyEngine()
    state = LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=10)
    inherited = engine.compile_policy(
        OperationalRequest(message_priority=MessagePriority.CRITICAL),
        state,
    )
    stricter = engine.compile_policy(
        OperationalRequest(
            message_priority=MessagePriority.CRITICAL,
            maximum_admissible_per=0.005,
        ),
        state,
    )

    assert inherited.maximum_admissible_per == 0.01
    assert stricter.maximum_admissible_per == 0.005
    assert inherited.policy_resolution is not None
    assert inherited.policy_resolution.requested_maximum_per_override is None


def test_weaker_per_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot weaken"):
        DefaultPolicyEngine().compile_policy(
            OperationalRequest(
                message_priority=MessagePriority.CRITICAL,
                maximum_admissible_per=0.1,
            ),
            LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=10),
        )


@pytest.mark.parametrize(
    ("timestamp", "expected_status"),
    (
        (datetime(2024, 1, 1, 12, 0, tzinfo=UTC), LinkAvailabilityStatus.BELOW_HORIZON),
        (
            datetime(2024, 1, 1, 19, 50, tzinfo=UTC),
            LinkAvailabilityStatus.BELOW_ELEVATION_MASK,
        ),
    ),
)
def test_non_visible_tle_blocks_prediction_and_command_without_research_override(
    timestamp: datetime,
    expected_status: LinkAvailabilityStatus,
    v1_knn: V1KnnPredictionEngine,
    fixed_tle: TLEInput,
    paris_station: GroundStation,
    radio_scenario: RadioScenario,
) -> None:
    result = build_pipeline(v1_knn).run_tle_state(
        TLEStateRequest(
            tle=fixed_tle,
            ground_station=paris_station,
            timestamp=timestamp,
            active_nodes=50_000,
            minimum_elevation_deg=5.0,
        ),
        radio_scenario,
        OperationalRequest(message_priority=MessagePriority.ROUTINE),
    )

    assert result.geometry.link_availability.status is expected_status
    assert not result.production_ready
    assert result.predictions == ()
    assert result.decision is None
    assert result.command is None
    assert any("Blocked production" in event.message for event in result.audit_trace)


def test_research_override_evaluates_non_visible_state_without_command(
    v1_knn: V1KnnPredictionEngine,
    fixed_tle: TLEInput,
    paris_station: GroundStation,
    radio_scenario: RadioScenario,
) -> None:
    result = build_pipeline(v1_knn).run_tle_state(
        TLEStateRequest(
            tle=fixed_tle,
            ground_station=paris_station,
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            active_nodes=50_000,
            minimum_elevation_deg=5.0,
        ),
        radio_scenario,
        OperationalRequest(message_priority=MessagePriority.ROUTINE),
        allow_non_visible_state_for_research=True,
    )

    assert len(result.predictions) == 8
    assert result.decision is not None
    assert result.command is None
    assert result.transmission_eligibility.research_only
    assert (
        TransmissionEligibilityReason.RESEARCH_ONLY_NON_VISIBLE_STATE
        in result.transmission_eligibility.reason_codes
    )


def test_constraint_evaluations_are_tri_state_and_preserve_admissibility(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    state = LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000)
    predictions = tuple(
        item.model_copy(
            update={
                "energy_per_attempt_millijoules": None,
                "expected_delivery_latency_ms": None,
            }
        )
        for item in v1_knn.predict(state, radio_scenario, tuple(RadioMode))
    )
    policy = MissionPolicy(
        policy_name="tri-state",
        version=ModelVersion(name="tri-state", version="1.0.0"),
        message_priority=MessagePriority.ROUTINE,
        maximum_admissible_per=1.0,
        optimization_objective=OptimizationObjective.MAXIMIZE_PACKETS_PER_WINDOW,
        allowed_radio_modes=frozenset(RadioMode),
        fallback_mode=RadioMode.LR_FHSS_DR8,
        constraints=(
            DecisionConstraint(
                constraint_id="soft.energy",
                kind=ConstraintKind.SOFT,
                metric=ConstraintMetric.ENERGY_PER_ATTEMPT_MILLIJOULES,
                operator=ConstraintOperator.LTE,
                threshold=10.0,
                unit="mJ/attempt",
            ),
            DecisionConstraint(
                constraint_id="hard.latency",
                kind=ConstraintKind.HARD,
                metric=ConstraintMetric.EXPECTED_DELIVERY_LATENCY_MS,
                operator=ConstraintOperator.LTE,
                threshold=1000.0,
                unit="ms",
                modes=frozenset({RadioMode.LORA_SF7}),
            ),
            DecisionConstraint(
                constraint_id="hard.elevation",
                kind=ConstraintKind.HARD,
                metric=ConstraintMetric.ELEVATION_DEG,
                operator=ConstraintOperator.LTE,
                threshold=-90.0,
                unit="deg",
                modes=frozenset({RadioMode.LORA_SF8}),
            ),
        ),
    )

    decision = DeterministicDecisionEngine().decide(state, policy, predictions)
    selected = [
        item
        for item in decision.constraint_evaluations
        if item.radio_mode is decision.selected_mode
    ]
    assert any(item.status is ConstraintEvaluationStatus.PASSED for item in selected)
    assert any(item.status is ConstraintEvaluationStatus.NOT_EVALUATED for item in selected)
    assert any(
        item.status is ConstraintEvaluationStatus.NOT_EVALUATED
        and item.kind is ConstraintKind.SOFT
        for item in selected
    )
    sf7 = next(item for item in decision.rejected_modes if item.radio_mode is RadioMode.LORA_SF7)
    assert "hard.latency" in sf7.constraint_ids
    assert any(
        item.radio_mode is RadioMode.LORA_SF7
        and item.constraint_id == "hard.latency"
        and item.status is ConstraintEvaluationStatus.NOT_EVALUATED
        for item in decision.constraint_evaluations
    )
    assert any(
        item.radio_mode is RadioMode.LORA_SF8
        and item.constraint_id == "hard.elevation"
        and item.status is ConstraintEvaluationStatus.FAILED
        for item in decision.constraint_evaluations
    )
