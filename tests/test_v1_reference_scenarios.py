from __future__ import annotations

import pytest

from meseleo_core import (
    RADIO_MODE_ORDER,
    DeterministicDecisionEngine,
    LinkState,
    LowElevationSafetyRule,
    MessagePriority,
    MissionPolicy,
    ModelVersion,
    OptimizationObjective,
    RadioMode,
    RadioScenario,
    ReasonCode,
    V1KnnPredictionEngine,
)

SCENARIO_POLICY_VERSION = ModelVersion(name="v1-reference-scenario", version="1.0.0")


def policy(
    maximum_per: float,
    objective: OptimizationObjective,
    low_elevation: bool = False,
) -> MissionPolicy:
    return MissionPolicy(
        policy_name="v1-reference-scenario",
        version=SCENARIO_POLICY_VERSION,
        message_priority=MessagePriority.ROUTINE,
        maximum_admissible_per=maximum_per,
        optimization_objective=objective,
        allowed_radio_modes=frozenset(RADIO_MODE_ORDER),
        fallback_mode=RadioMode.LR_FHSS_DR8,
        low_elevation_safety=(
            LowElevationSafetyRule(
                below_elevation_deg=15.0,
                forced_mode=RadioMode.LR_FHSS_DR8,
            )
            if low_elevation
            else None
        ),
    )


@pytest.mark.parametrize(
    ("state", "maximum_per", "expected_mode", "expected_per_range", "low_elevation"),
    (
        (
            LinkState(elevation_deg=5.0, radial_speed_km_s=0.0, active_nodes=10_000),
            0.10,
            RadioMode.LR_FHSS_DR8,
            (0.0034, 0.0036),
            True,
        ),
        (
            LinkState(elevation_deg=85.0, radial_speed_km_s=0.5, active_nodes=100_000),
            0.10,
            RadioMode.LR_FHSS_DR9,
            (0.0202, 0.0205),
            False,
        ),
        (
            LinkState(elevation_deg=27.0, radial_speed_km_s=2.5, active_nodes=10_000),
            0.40,
            RadioMode.LORA_SF8,
            (0.324, 0.325),
            False,
        ),
        (
            LinkState(elevation_deg=14.0, radial_speed_km_s=5.5, active_nodes=50_000),
            0.01,
            RadioMode.LR_FHSS_DR8,
            (0.0058, 0.0059),
            False,
        ),
    ),
)
def test_versioned_v1_reference_scenarios(
    state: LinkState,
    maximum_per: float,
    expected_mode: RadioMode,
    expected_per_range: tuple[float, float],
    low_elevation: bool,
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    predictions = v1_knn.predict(state, radio_scenario, RADIO_MODE_ORDER)
    decision = DeterministicDecisionEngine().decide(
        state,
        policy(
            maximum_per,
            OptimizationObjective.MAXIMIZE_PACKETS_PER_WINDOW,
            low_elevation=low_elevation,
        ),
        predictions,
    )

    assert decision.selected_mode is expected_mode
    assert decision.selected_prediction is not None
    assert (
        expected_per_range[0]
        <= decision.selected_prediction.predicted_per
        <= (expected_per_range[1])
    )


def test_low_elevation_dr8_rule_is_policy_configuration(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    state = LinkState(elevation_deg=5.0, radial_speed_km_s=0.0, active_nodes=10_000)
    predictions = v1_knn.predict(state, radio_scenario, RADIO_MODE_ORDER)
    decision = DeterministicDecisionEngine().decide(
        state,
        policy(0.10, OptimizationObjective.MAXIMIZE_PACKETS_PER_WINDOW, low_elevation=True),
        predictions,
    )

    assert decision.selected_mode is RadioMode.LR_FHSS_DR8
    assert ReasonCode.LOW_ELEVATION_SAFETY_FALLBACK in decision.reason_codes
