from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from meseleo_core import (
    DefaultPolicyEngine,
    EnergyMode,
    LinkState,
    MessagePriority,
    OperationalRequest,
    OptimizationObjective,
    RadioMode,
    UncertaintyAction,
)


@pytest.mark.parametrize(
    ("priority", "expected_profile"),
    (
        (MessagePriority.ROUTINE, "routine_telemetry"),
        (MessagePriority.IMPORTANT, "reliable_telemetry"),
        (MessagePriority.URGENT, "reliable_telemetry"),
        (MessagePriority.CRITICAL, "critical_alert"),
    ),
)
def test_priority_compiles_to_generic_profile(
    priority: MessagePriority,
    expected_profile: str,
    link_state: LinkState,
) -> None:
    request = OperationalRequest(message_priority=priority)

    policy = DefaultPolicyEngine().compile_policy(request, link_state)

    assert policy.policy_name == expected_profile
    assert policy.source_request == request
    assert policy.assumptions
    assert {item.key for item in policy.metadata} >= {
        "policy_compiler",
        "profile_name",
        "profile_version",
        "generated_maximum_per",
    }


def test_critical_alert_is_strict_and_uses_safe_fallback(link_state: LinkState) -> None:
    policy = DefaultPolicyEngine().compile_policy(
        OperationalRequest(message_priority=MessagePriority.CRITICAL),
        link_state,
    )

    assert policy.maximum_admissible_per == 0.01
    assert policy.optimization_objective is OptimizationObjective.MINIMIZE_PER
    assert policy.uncertainty_action is UncertaintyAction.SAFE_FALLBACK
    assert policy.fallback_mode is RadioMode.LR_FHSS_DR8


def test_energy_mode_selects_battery_profile(link_state: LinkState) -> None:
    policy = DefaultPolicyEngine().compile_policy(
        OperationalRequest(
            message_priority=MessagePriority.ROUTINE,
            energy_mode=EnergyMode.CONSERVE_ENERGY,
        ),
        link_state,
    )

    assert policy.policy_name == "battery_saving"
    assert policy.optimization_objective is OptimizationObjective.MINIMIZE_ENERGY_PER_ATTEMPT


def test_maximum_throughput_profile_is_explicit(link_state: LinkState) -> None:
    policy = DefaultPolicyEngine().compile_policy(
        OperationalRequest(
            message_priority=MessagePriority.ROUTINE,
            policy_profile="maximum_throughput",
        ),
        link_state,
    )

    assert policy.optimization_objective is OptimizationObjective.MAXIMIZE_GOODPUT_BPS
    assert policy.maximum_admissible_per == 0.20


def test_explicit_constraints_may_only_tighten_profile(link_state: LinkState) -> None:
    engine = DefaultPolicyEngine()
    tightened = engine.compile_policy(
        OperationalRequest(
            message_priority=MessagePriority.ROUTINE,
            maximum_admissible_per=0.02,
            delivery_reliability_target=0.995,
            allowed_radio_modes=frozenset({RadioMode.LR_FHSS_DR8, RadioMode.LR_FHSS_DR9}),
            fallback_mode=RadioMode.LR_FHSS_DR8,
        ),
        link_state,
    )

    assert tightened.maximum_admissible_per == pytest.approx(0.005)
    assert tightened.allowed_radio_modes == frozenset(
        {RadioMode.LR_FHSS_DR8, RadioMode.LR_FHSS_DR9}
    )

    with pytest.raises(ValueError, match="cannot weaken"):
        engine.compile_policy(
            OperationalRequest(
                message_priority=MessagePriority.ROUTINE,
                maximum_admissible_per=0.20,
            ),
            link_state,
        )


def test_deadline_compiles_to_explicit_latency_unit() -> None:
    timestamp = datetime(2026, 7, 13, tzinfo=UTC)
    state = LinkState(
        elevation_deg=40.0,
        radial_speed_km_s=3.0,
        active_nodes=10_000,
        timestamp=timestamp,
    )

    policy = DefaultPolicyEngine().compile_policy(
        OperationalRequest(
            message_priority=MessagePriority.IMPORTANT,
            deadline=timestamp + timedelta(seconds=2.5),
        ),
        state,
    )

    assert policy.maximum_expected_delivery_latency_ms == 2_500.0


def test_low_elevation_rule_is_visible_and_configurable(link_state: LinkState) -> None:
    policy = DefaultPolicyEngine().compile_policy(
        OperationalRequest(
            message_priority=MessagePriority.ROUTINE,
            low_elevation_threshold_deg=8.0,
            fallback_mode=RadioMode.LR_FHSS_DR9,
        ),
        link_state,
    )

    assert policy.low_elevation_safety is not None
    assert policy.low_elevation_safety.below_elevation_deg == 8.0
    assert policy.low_elevation_safety.forced_mode is RadioMode.LR_FHSS_DR9
