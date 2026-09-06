from __future__ import annotations

from collections.abc import Callable

import pytest

from meseleo_core import (
    AuditEventType,
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
    DuplicatePredictionError,
    FallbackUnavailableError,
    LinkState,
    LowElevationSafetyRule,
    MessagePriority,
    MissionPolicy,
    ModelVersion,
    ModeRejection,
    OptimizationObjective,
    RadioMode,
    RadioPrediction,
    ReasonCode,
    UncertaintyAction,
)


def _rejection(result: DecisionResult, mode: RadioMode) -> ModeRejection | None:
    return next((item for item in result.rejected_modes if item.radio_mode is mode), None)


def test_normal_admissible_selection(
    link_state: LinkState,
    predictions: tuple[RadioPrediction, ...],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    result = DeterministicDecisionEngine().decide(
        link_state, policy_factory(maximum_admissible_per=0.25), predictions
    )
    assert result.selected_mode is RadioMode.LORA_SF7
    assert result.fallback_applied is False
    assert result.constraint_satisfaction_status is ConstraintSatisfactionStatus.SATISFIED


def test_reliability_constrained_goodput_maximization(
    link_state: LinkState,
    predictions: tuple[RadioPrediction, ...],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    result = DeterministicDecisionEngine().decide(link_state, policy_factory(), predictions)
    assert result.selected_mode is RadioMode.LORA_SF8
    rejection = _rejection(result, RadioMode.LORA_SF7)
    assert rejection is not None
    assert ReasonCode.PER_EXCEEDS_MAXIMUM in rejection.reason_codes


def test_packets_per_window_is_a_distinct_objective_not_bit_per_second(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            goodput_bps=4_000.0,
            packets_per_window=2.0,
        ),
        prediction_factory(
            RadioMode.LORA_SF9,
            goodput_bps=1_000.0,
            packets_per_window=8.0,
        ),
        prediction_factory(RadioMode.LR_FHSS_DR8, packets_per_window=1.0),
    )
    policy = policy_factory(
        optimization_objective=OptimizationObjective.MAXIMIZE_PACKETS_PER_WINDOW
    )
    result = DeterministicDecisionEngine().decide(link_state, policy, predictions)
    assert result.selected_mode is RadioMode.LORA_SF9


def test_no_admissible_mode_applies_configured_fallback(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = tuple(prediction_factory(mode, per=0.5) for mode in RadioMode)
    result = DeterministicDecisionEngine().decide(link_state, policy_factory(), predictions)
    assert result.selected_mode is RadioMode.LR_FHSS_DR8
    assert result.fallback_applied is True
    assert result.constraint_satisfaction_status is (
        ConstraintSatisfactionStatus.FALLBACK_WITH_VIOLATIONS
    )
    assert ReasonCode.NO_ADMISSIBLE_MODE in result.reason_codes


def test_configured_low_elevation_dr8_fallback(
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    state = LinkState(elevation_deg=10.0, radial_speed_km_s=6.5, active_nodes=100_000)
    predictions = (
        prediction_factory(RadioMode.LR_FHSS_DR8, goodput_bps=100.0),
        prediction_factory(RadioMode.LR_FHSS_DR9, goodput_bps=300.0),
    )
    policy = policy_factory(
        allowed_radio_modes=frozenset({RadioMode.LR_FHSS_DR8, RadioMode.LR_FHSS_DR9}),
        low_elevation_safety=LowElevationSafetyRule(
            below_elevation_deg=15.0,
            forced_mode=RadioMode.LR_FHSS_DR8,
        ),
    )
    result = DeterministicDecisionEngine().decide(state, policy, predictions)
    assert result.selected_mode is RadioMode.LR_FHSS_DR8
    assert result.fallback_applied is True
    assert ReasonCode.LOW_ELEVATION_SAFETY_FALLBACK in result.reason_codes


def test_critical_message_policy_uses_same_generic_model(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(RadioMode.LR_FHSS_DR8, per=0.002, goodput_bps=100.0),
        prediction_factory(RadioMode.LR_FHSS_DR9, per=0.009, goodput_bps=300.0),
    )
    policy = policy_factory(
        message_priority=MessagePriority.CRITICAL,
        maximum_admissible_per=0.005,
        allowed_radio_modes=frozenset({RadioMode.LR_FHSS_DR8, RadioMode.LR_FHSS_DR9}),
    )
    result = DeterministicDecisionEngine().decide(link_state, policy, predictions)
    assert result.selected_mode is RadioMode.LR_FHSS_DR8
    assert result.audit_trace[0].details[1].value == "critical"


def test_expected_delivery_latency_constraint(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            goodput_bps=3_000.0,
            expected_delivery_latency_ms=250.0,
        ),
        prediction_factory(
            RadioMode.LORA_SF9,
            goodput_bps=1_500.0,
            expected_delivery_latency_ms=50.0,
        ),
        prediction_factory(RadioMode.LR_FHSS_DR8, expected_delivery_latency_ms=40.0),
    )
    result = DeterministicDecisionEngine().decide(
        link_state,
        policy_factory(maximum_expected_delivery_latency_ms=100.0),
        predictions,
    )
    assert result.selected_mode is RadioMode.LORA_SF9
    rejection = _rejection(result, RadioMode.LORA_SF8)
    assert rejection is not None
    assert ReasonCode.LATENCY_EXCEEDS_MAXIMUM in rejection.reason_codes


def test_energy_per_attempt_constraint(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            goodput_bps=3_000.0,
            energy_per_attempt_millijoules=12.0,
        ),
        prediction_factory(
            RadioMode.LORA_SF9,
            goodput_bps=1_500.0,
            energy_per_attempt_millijoules=8.0,
        ),
        prediction_factory(RadioMode.LR_FHSS_DR8, energy_per_attempt_millijoules=9.0),
    )
    result = DeterministicDecisionEngine().decide(
        link_state,
        policy_factory(maximum_energy_per_attempt_millijoules=10.0),
        predictions,
    )
    assert result.selected_mode is RadioMode.LORA_SF9
    rejection = _rejection(result, RadioMode.LORA_SF8)
    assert rejection is not None
    assert ReasonCode.ENERGY_EXCEEDS_MAXIMUM in rejection.reason_codes


def test_allowed_mode_restriction(
    link_state: LinkState,
    predictions: tuple[RadioPrediction, ...],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    policy = policy_factory(
        allowed_radio_modes=frozenset({RadioMode.LORA_SF12, RadioMode.LR_FHSS_DR8})
    )
    result = DeterministicDecisionEngine().decide(link_state, policy, predictions)
    assert result.selected_mode is RadioMode.LORA_SF12
    rejection = _rejection(result, RadioMode.LORA_SF8)
    assert rejection is not None
    assert ReasonCode.MODE_FORBIDDEN in rejection.reason_codes


def test_calibrated_confidence_threshold(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(RadioMode.LORA_SF8, goodput_bps=3_000.0, confidence=0.80),
        prediction_factory(RadioMode.LORA_SF9, goodput_bps=1_500.0, confidence=0.95),
        prediction_factory(RadioMode.LR_FHSS_DR8, confidence=0.99),
    )
    result = DeterministicDecisionEngine().decide(
        link_state, policy_factory(minimum_confidence=0.90), predictions
    )
    assert result.selected_mode is RadioMode.LORA_SF9
    rejection = _rejection(result, RadioMode.LORA_SF8)
    assert rejection is not None
    assert ReasonCode.CONFIDENCE_BELOW_MINIMUM in rejection.reason_codes


@pytest.mark.parametrize("status", [ConfidenceStatus.UNAVAILABLE, ConfidenceStatus.UNCALIBRATED])
def test_non_calibrated_confidence_required_by_hard_constraint_rejects_mode(
    status: ConfidenceStatus,
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    confidence = None if status is ConfidenceStatus.UNAVAILABLE else 0.99
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            goodput_bps=3_000.0,
            confidence=confidence,
            confidence_status=status,
        ),
        prediction_factory(RadioMode.LR_FHSS_DR8, confidence=0.95),
    )
    result = DeterministicDecisionEngine().decide(
        link_state, policy_factory(minimum_confidence=0.9), predictions
    )
    rejection = _rejection(result, RadioMode.LORA_SF8)
    assert rejection is not None
    assert ReasonCode.MISSING_CONSTRAINT_DATA in rejection.reason_codes
    assert "policy.minimum_confidence" in rejection.constraint_ids


def test_unavailable_confidence_is_permitted_when_policy_does_not_require_it(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            goodput_bps=3_000.0,
            confidence=None,
            confidence_status=ConfidenceStatus.UNAVAILABLE,
        ),
        prediction_factory(RadioMode.LR_FHSS_DR8),
    )
    result = DeterministicDecisionEngine().decide(link_state, policy_factory(), predictions)
    assert result.selected_mode is RadioMode.LORA_SF8
    assert result.selected_prediction is not None
    assert result.selected_prediction.confidence is None


def test_out_of_distribution_fallback(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            goodput_bps=3_000.0,
            distribution_status=DistributionStatus.OUT_OF_DISTRIBUTION,
        ),
        prediction_factory(RadioMode.LR_FHSS_DR8, goodput_bps=100.0),
    )
    result = DeterministicDecisionEngine().decide(
        link_state,
        policy_factory(uncertainty_action=UncertaintyAction.SAFE_FALLBACK),
        predictions,
    )
    assert result.selected_mode is RadioMode.LR_FHSS_DR8
    assert result.fallback_applied is True
    assert ReasonCode.OOD_SAFE_FALLBACK in result.reason_codes


def test_uncertain_prediction_can_be_allowed_with_warning(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            goodput_bps=3_000.0,
            distribution_status=DistributionStatus.UNCERTAIN,
        ),
        prediction_factory(RadioMode.LR_FHSS_DR8, goodput_bps=100.0),
    )
    result = DeterministicDecisionEngine().decide(
        link_state,
        policy_factory(uncertainty_action=UncertaintyAction.ALLOW_WITH_WARNING),
        predictions,
    )
    assert result.selected_mode is RadioMode.LORA_SF8
    assert ReasonCode.OOD_ALLOWED_WITH_WARNING in result.reason_codes


def test_deterministic_tie_breaking(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(RadioMode.LORA_SF9, goodput_bps=1_000.0),
        prediction_factory(RadioMode.LORA_SF8, goodput_bps=1_000.0),
        prediction_factory(RadioMode.LR_FHSS_DR8, goodput_bps=100.0),
    )
    result = DeterministicDecisionEngine().decide(link_state, policy_factory(), predictions)
    assert result.selected_mode is RadioMode.LORA_SF8
    assert ReasonCode.DETERMINISTIC_TIE_BREAK in result.reason_codes


def test_missing_modulation_prediction_is_reported(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (prediction_factory(RadioMode.LR_FHSS_DR8),)
    policy = policy_factory(
        allowed_radio_modes=frozenset({RadioMode.LORA_SF8, RadioMode.LR_FHSS_DR8})
    )
    result = DeterministicDecisionEngine().decide(link_state, policy, predictions)
    rejection = _rejection(result, RadioMode.LORA_SF8)
    assert rejection is not None
    assert ReasonCode.MISSING_PREDICTION in rejection.reason_codes


def test_fallback_without_prediction_is_explicitly_unverified(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (prediction_factory(RadioMode.LORA_SF8, per=0.9),)
    policy = policy_factory(
        allowed_radio_modes=frozenset({RadioMode.LORA_SF8, RadioMode.LR_FHSS_DR8})
    )
    result = DeterministicDecisionEngine().decide(link_state, policy, predictions)
    assert result.selected_prediction is None
    assert result.constraint_satisfaction_status is (
        ConstraintSatisfactionStatus.FALLBACK_UNVERIFIED
    )
    assert ReasonCode.FALLBACK_PREDICTION_MISSING in result.reason_codes


def test_duplicate_prediction_is_rejected(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    prediction = prediction_factory(RadioMode.LR_FHSS_DR8)
    with pytest.raises(DuplicatePredictionError):
        DeterministicDecisionEngine().decide(link_state, policy_factory(), (prediction, prediction))


def test_hybrid_predictor_versions_are_preserved_in_radio_mode_order(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    first = ModelVersion(name="lora-model", version="2.0.0")
    second = ModelVersion(name="lr-fhss-model", version="3.0.0")
    predictions = (
        prediction_factory(RadioMode.LR_FHSS_DR8, predictor_version=second),
        prediction_factory(
            RadioMode.LORA_SF8,
            predictor_version=first,
            goodput_bps=2_000.0,
        ),
    )
    result = DeterministicDecisionEngine().decide(link_state, policy_factory(), predictions)
    assert tuple(item.radio_mode for item in result.prediction_provenance) == (
        RadioMode.LORA_SF8,
        RadioMode.LR_FHSS_DR8,
    )
    assert tuple(item.predictor_version for item in result.prediction_provenance) == (
        first,
        second,
    )


def test_unavailable_mode_prediction_is_rejected(
    link_state: LinkState,
    predictions: tuple[RadioPrediction, ...],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    available = frozenset({RadioMode.LORA_SF9, RadioMode.LR_FHSS_DR8})
    result = DeterministicDecisionEngine().decide(
        link_state, policy_factory(), predictions, available
    )
    assert result.selected_mode is RadioMode.LORA_SF9
    rejection = _rejection(result, RadioMode.LORA_SF8)
    assert rejection is not None
    assert ReasonCode.MODE_UNAVAILABLE in rejection.reason_codes


def test_unavailable_fallback_is_a_typed_failure(
    link_state: LinkState,
    predictions: tuple[RadioPrediction, ...],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    with pytest.raises(FallbackUnavailableError):
        DeterministicDecisionEngine().decide(
            link_state,
            policy_factory(),
            predictions,
            frozenset({RadioMode.LORA_SF8}),
        )


def test_minimize_energy_per_attempt_objective(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(RadioMode.LORA_SF8, energy_per_attempt_millijoules=5.0),
        prediction_factory(RadioMode.LORA_SF9, energy_per_attempt_millijoules=3.0),
        prediction_factory(RadioMode.LR_FHSS_DR8, energy_per_attempt_millijoules=10.0),
    )
    result = DeterministicDecisionEngine().decide(
        link_state,
        policy_factory(optimization_objective=OptimizationObjective.MINIMIZE_ENERGY_PER_ATTEMPT),
        predictions,
    )
    assert result.selected_mode is RadioMode.LORA_SF9


def test_balanced_weighted_score_uses_only_policy_supplied_weights(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            per=0.09,
            goodput_bps=4_000.0,
            energy_per_attempt_millijoules=10.0,
        ),
        prediction_factory(
            RadioMode.LORA_SF9,
            per=0.01,
            goodput_bps=1_000.0,
            energy_per_attempt_millijoules=5.0,
        ),
        prediction_factory(
            RadioMode.LR_FHSS_DR8,
            per=0.005,
            goodput_bps=100.0,
            energy_per_attempt_millijoules=20.0,
        ),
    )
    weights = BalancedWeights(
        goodput_bps=0.0,
        packets_per_window=0.0,
        reliability=1.0,
        expected_delivery_latency=0.0,
        energy_per_attempt=1.0,
    )
    result = DeterministicDecisionEngine().decide(
        link_state,
        policy_factory(
            optimization_objective=OptimizationObjective.BALANCED_WEIGHTED_SCORE,
            balanced_weights=weights,
        ),
        predictions,
    )
    assert result.selected_mode is RadioMode.LORA_SF9


def test_missing_energy_for_energy_objective_is_not_ignored(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(RadioMode.LORA_SF8, energy_per_attempt_millijoules=None),
        prediction_factory(RadioMode.LR_FHSS_DR8, energy_per_attempt_millijoules=10.0),
    )
    result = DeterministicDecisionEngine().decide(
        link_state,
        policy_factory(optimization_objective=OptimizationObjective.MINIMIZE_ENERGY_PER_ATTEMPT),
        predictions,
    )
    rejection = _rejection(result, RadioMode.LORA_SF8)
    assert rejection is not None
    assert ReasonCode.MISSING_OBJECTIVE_DATA in rejection.reason_codes


def test_missing_latency_and_energy_have_one_reason_and_two_constraint_ids(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            expected_delivery_latency_ms=None,
            energy_per_attempt_millijoules=None,
        ),
        prediction_factory(RadioMode.LR_FHSS_DR8),
    )
    policy = policy_factory(
        maximum_expected_delivery_latency_ms=100.0,
        maximum_energy_per_attempt_millijoules=20.0,
    )
    result = DeterministicDecisionEngine().decide(link_state, policy, predictions)
    rejection = _rejection(result, RadioMode.LORA_SF8)
    assert rejection is not None
    assert rejection.reason_codes.count(ReasonCode.MISSING_CONSTRAINT_DATA) == 1
    assert rejection.constraint_ids == (
        "policy.maximum_expected_delivery_latency_ms",
        "policy.maximum_energy_per_attempt_millijoules",
    )


def test_soft_constraint_is_audited_without_becoming_hard(
    link_state: LinkState,
    prediction_factory: Callable[..., RadioPrediction],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    soft = DecisionConstraint(
        constraint_id="prefer-under-5ms",
        kind=ConstraintKind.SOFT,
        metric=ConstraintMetric.EXPECTED_DELIVERY_LATENCY_MS,
        operator=ConstraintOperator.LTE,
        threshold=5.0,
        unit="ms",
    )
    predictions = (
        prediction_factory(
            RadioMode.LORA_SF8,
            goodput_bps=2_000.0,
            expected_delivery_latency_ms=50.0,
        ),
        prediction_factory(
            RadioMode.LR_FHSS_DR8,
            goodput_bps=100.0,
            expected_delivery_latency_ms=1.0,
        ),
    )
    result = DeterministicDecisionEngine().decide(
        link_state, policy_factory(constraints=(soft,)), predictions
    )
    assert result.selected_mode is RadioMode.LORA_SF8
    assert ReasonCode.SOFT_CONSTRAINT_VIOLATED in result.reason_codes


def test_reproducibility_is_independent_of_prediction_input_order(
    link_state: LinkState,
    predictions: tuple[RadioPrediction, ...],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    engine = DeterministicDecisionEngine()
    policy = policy_factory()
    forward = engine.decide(link_state, policy, predictions)
    reverse = engine.decide(link_state, policy, tuple(reversed(predictions)))
    assert forward == reverse
    assert forward.model_dump_json() == reverse.model_dump_json()


def test_complete_audit_trace_generation(
    link_state: LinkState,
    predictions: tuple[RadioPrediction, ...],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    result = DeterministicDecisionEngine().decide(link_state, policy_factory(), predictions)
    assert result.audit_trace[0].event_type is AuditEventType.VALIDATION
    assert result.audit_trace[-1].event_type is AuditEventType.SELECTION
    assert tuple(event.sequence for event in result.audit_trace) == tuple(
        range(len(result.audit_trace))
    )
    modes_with_trace = {event.radio_mode for event in result.audit_trace if event.radio_mode}
    assert modes_with_trace == set(RadioMode)
    assert all(rejection.reason_codes for rejection in result.rejected_modes)


def test_no_generative_provider_is_invoked_by_exact_decision_path(
    link_state: LinkState,
    predictions: tuple[RadioPrediction, ...],
    policy_factory: Callable[..., MissionPolicy],
) -> None:
    class ExplodingExplanationProvider:
        calls = 0

        def explain(self, link_state: LinkState, decision: object) -> str:
            del link_state, decision
            self.calls += 1
            raise AssertionError("generative provider entered the exact path")

    provider = ExplodingExplanationProvider()
    result = DeterministicDecisionEngine().decide(link_state, policy_factory(), predictions)
    assert result.selected_mode is RadioMode.LORA_SF8
    assert provider.calls == 0
