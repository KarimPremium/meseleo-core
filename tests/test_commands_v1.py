from __future__ import annotations

import pytest

from meseleo_core import (
    RADIO_MODE_ORDER,
    CommandMappingUnavailableError,
    DeterministicDecisionEngine,
    LinkAvailability,
    LinkAvailabilityStatus,
    LinkState,
    MessagePriority,
    MissionPolicy,
    ModelVersion,
    OptimizationObjective,
    RadioMode,
    RadioScenario,
    TransmissionEligibility,
    TransmissionEligibilityReason,
    TransmissionEligibilityStatus,
    TransmissionIneligibleError,
    V1ATCommandAdapter,
    V1CommandConfiguration,
    V1KnnPredictionEngine,
)

EXPECTED_COMMANDS = {
    RadioMode.LORA_SF7: "AT+MOD=LORA-SF7",
    RadioMode.LORA_SF8: "AT+MOD=LORA-SF8",
    RadioMode.LORA_SF9: "AT+MOD=LORA-SF9",
    RadioMode.LORA_SF10: "AT+MOD=LORA-SF10",
    RadioMode.LORA_SF11: "AT+MOD=LORA-SF11",
    RadioMode.LORA_SF12: "AT+MOD=LORA-SF12",
    RadioMode.LR_FHSS_DR8: "AT+MOD=LR-FHSS-DR8",
    RadioMode.LR_FHSS_DR9: "AT+MOD=LR-FHSS-DR9",
}


def eligible_transmission() -> TransmissionEligibility:
    return TransmissionEligibility(
        status=TransmissionEligibilityStatus.READY,
        eligible=True,
        link_availability=LinkAvailability(
            status=LinkAvailabilityStatus.VISIBLE,
            elevation_deg=45.0,
            minimum_elevation_deg=5.0,
            source="test",
        ),
        reason_codes=(TransmissionEligibilityReason.LINK_WINDOW_VALID,),
    )


@pytest.mark.parametrize("mode", RADIO_MODE_ORDER)
def test_all_v1_modes_have_explicit_at_mapping(
    mode: RadioMode,
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    state = LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000)
    prediction = v1_knn.predict(state, radio_scenario, (mode,))
    policy = MissionPolicy(
        policy_name="single-mode-command-test",
        version=ModelVersion(name="command-test", version="1.0.0"),
        message_priority=MessagePriority.ROUTINE,
        maximum_admissible_per=1.0,
        optimization_objective=OptimizationObjective.MINIMIZE_PER,
        allowed_radio_modes=frozenset({mode}),
        fallback_mode=mode,
    )
    decision = DeterministicDecisionEngine().decide(state, policy, prediction)

    command = V1ATCommandAdapter().translate(decision, eligible_transmission())

    assert command.command == EXPECTED_COMMANDS[mode]
    assert command.radio_mode is mode
    assert command.modem_profile == "v1-generic-at"
    assert command.validation_status.value == "validated"


def test_missing_command_mapping_never_falls_back_silently(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    mode = RadioMode.LR_FHSS_DR8
    state = LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000)
    decision = DeterministicDecisionEngine().decide(
        state,
        MissionPolicy(
            policy_name="missing-mapping",
            version=ModelVersion(name="command-test", version="1.0.0"),
            message_priority=MessagePriority.ROUTINE,
            maximum_admissible_per=1.0,
            optimization_objective=OptimizationObjective.MINIMIZE_PER,
            allowed_radio_modes=frozenset({mode}),
            fallback_mode=mode,
        ),
        v1_knn.predict(state, radio_scenario, (mode,)),
    )

    with pytest.raises(CommandMappingUnavailableError):
        V1ATCommandAdapter(V1CommandConfiguration(mappings=())).translate(
            decision,
            eligible_transmission(),
        )


def test_command_adapter_rejects_ineligible_transmission(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    state = LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000)
    mode = RadioMode.LR_FHSS_DR8
    decision = DeterministicDecisionEngine().decide(
        state,
        MissionPolicy(
            policy_name="ineligible-command",
            version=ModelVersion(name="command-test", version="1.0.0"),
            message_priority=MessagePriority.ROUTINE,
            maximum_admissible_per=1.0,
            optimization_objective=OptimizationObjective.MINIMIZE_PER,
            allowed_radio_modes=frozenset({mode}),
            fallback_mode=mode,
        ),
        v1_knn.predict(state, radio_scenario, (mode,)),
    )
    ineligible = TransmissionEligibility(
        status=TransmissionEligibilityStatus.BLOCKED,
        eligible=False,
        link_availability=LinkAvailability(
            status=LinkAvailabilityStatus.BELOW_HORIZON,
            elevation_deg=-5.0,
            minimum_elevation_deg=5.0,
            source="test",
        ),
        reason_codes=(TransmissionEligibilityReason.LINK_NOT_VISIBLE,),
    )

    with pytest.raises(TransmissionIneligibleError):
        V1ATCommandAdapter().translate(decision, ineligible)
