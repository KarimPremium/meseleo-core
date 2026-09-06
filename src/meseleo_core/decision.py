"""Exact deterministic reliability-constrained decision engine."""

from __future__ import annotations

from collections.abc import Callable

from .exceptions import (
    DuplicatePredictionError,
    FallbackUnavailableError,
    UnsupportedConstraintError,
)
from .models import (
    RADIO_MODE_ORDER,
    RADIO_MODE_RANK,
    AuditEvent,
    AuditEventType,
    ConfidenceStatus,
    ConstraintEvaluation,
    ConstraintEvaluationStatus,
    ConstraintKind,
    ConstraintMetric,
    ConstraintOperator,
    ConstraintSatisfactionStatus,
    DecisionConstraint,
    DecisionResult,
    DistributionStatus,
    LinkState,
    MetadataItem,
    MissionPolicy,
    ModelVersion,
    ModeRejection,
    OptimizationObjective,
    PredictionProvenance,
    RadioMode,
    RadioPrediction,
    ReasonCode,
    UncertaintyAction,
)

DECISION_ENGINE_VERSION = ModelVersion(
    name="meseleo-deterministic-decision-engine",
    version="1.1.0",
)


class _AuditBuilder:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def add(
        self,
        event_type: AuditEventType,
        reason_code: ReasonCode,
        message: str,
        radio_mode: RadioMode | None = None,
        details: tuple[MetadataItem, ...] = (),
    ) -> None:
        self._events.append(
            AuditEvent(
                sequence=len(self._events),
                event_type=event_type,
                reason_code=reason_code,
                message=message,
                radio_mode=radio_mode,
                details=details,
            )
        )

    def result(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)


def _append_unique[T](items: list[T], item: T) -> None:
    if item not in items:
        items.append(item)


def _compare(value: float, operator: ConstraintOperator, threshold: float) -> bool:
    functions: dict[ConstraintOperator, Callable[[float, float], bool]] = {
        ConstraintOperator.LT: lambda left, right: left < right,
        ConstraintOperator.LTE: lambda left, right: left <= right,
        ConstraintOperator.GT: lambda left, right: left > right,
        ConstraintOperator.GTE: lambda left, right: left >= right,
        ConstraintOperator.EQ: lambda left, right: left == right,
    }
    try:
        return functions[operator](value, threshold)
    except KeyError as error:  # Defensive if a future enum reaches an older engine.
        raise UnsupportedConstraintError(f"unsupported operator: {operator}") from error


def _constraint_value(
    constraint: DecisionConstraint,
    link_state: LinkState,
    prediction: RadioPrediction,
) -> float | None:
    calibrated_confidence = (
        prediction.confidence
        if prediction.confidence_status is ConfidenceStatus.CALIBRATED
        else None
    )
    values: dict[ConstraintMetric, float | None] = {
        ConstraintMetric.PREDICTED_PER: prediction.predicted_per,
        ConstraintMetric.PREDICTED_GOODPUT_BPS: prediction.predicted_goodput_bps,
        ConstraintMetric.PREDICTED_PACKETS_PER_WINDOW: prediction.predicted_packets_per_window,
        ConstraintMetric.EXPECTED_DELIVERY_LATENCY_MS: (prediction.expected_delivery_latency_ms),
        ConstraintMetric.ENERGY_PER_ATTEMPT_MILLIJOULES: (
            prediction.energy_per_attempt_millijoules
        ),
        ConstraintMetric.CONFIDENCE: calibrated_confidence,
        ConstraintMetric.ELEVATION_DEG: link_state.elevation_deg,
    }
    try:
        return values[constraint.metric]
    except KeyError as error:
        raise UnsupportedConstraintError(
            f"unsupported constraint metric: {constraint.metric}"
        ) from error


def _standard_constraints(policy: MissionPolicy) -> tuple[DecisionConstraint, ...]:
    constraints: list[DecisionConstraint] = [
        DecisionConstraint(
            constraint_id="policy.maximum_admissible_per",
            kind=ConstraintKind.HARD,
            metric=ConstraintMetric.PREDICTED_PER,
            operator=ConstraintOperator.LTE,
            threshold=policy.maximum_admissible_per,
            unit="ratio",
        )
    ]
    if policy.maximum_expected_delivery_latency_ms is not None:
        constraints.append(
            DecisionConstraint(
                constraint_id="policy.maximum_expected_delivery_latency_ms",
                kind=ConstraintKind.HARD,
                metric=ConstraintMetric.EXPECTED_DELIVERY_LATENCY_MS,
                operator=ConstraintOperator.LTE,
                threshold=policy.maximum_expected_delivery_latency_ms,
                unit="ms",
            )
        )
    if policy.maximum_energy_per_attempt_millijoules is not None:
        constraints.append(
            DecisionConstraint(
                constraint_id="policy.maximum_energy_per_attempt_millijoules",
                kind=ConstraintKind.HARD,
                metric=ConstraintMetric.ENERGY_PER_ATTEMPT_MILLIJOULES,
                operator=ConstraintOperator.LTE,
                threshold=policy.maximum_energy_per_attempt_millijoules,
                unit="mJ/attempt",
            )
        )
    if policy.minimum_confidence is not None:
        constraints.append(
            DecisionConstraint(
                constraint_id="policy.minimum_confidence",
                kind=ConstraintKind.HARD,
                metric=ConstraintMetric.CONFIDENCE,
                operator=ConstraintOperator.GTE,
                threshold=policy.minimum_confidence,
                unit="ratio",
            )
        )
    constraints.extend(policy.constraints)
    return tuple(constraints)


def _specific_failure_code(constraint: DecisionConstraint) -> ReasonCode:
    standard_codes = {
        "policy.maximum_admissible_per": ReasonCode.PER_EXCEEDS_MAXIMUM,
        "policy.maximum_expected_delivery_latency_ms": ReasonCode.LATENCY_EXCEEDS_MAXIMUM,
        "policy.maximum_energy_per_attempt_millijoules": ReasonCode.ENERGY_EXCEEDS_MAXIMUM,
        "policy.minimum_confidence": ReasonCode.CONFIDENCE_BELOW_MINIMUM,
    }
    return standard_codes.get(constraint.constraint_id, ReasonCode.HARD_CONSTRAINT_FAILED)


def _objective_data_missing(policy: MissionPolicy, prediction: RadioPrediction) -> bool:
    objective = policy.optimization_objective
    if objective is OptimizationObjective.MAXIMIZE_GOODPUT_BPS:
        return prediction.predicted_goodput_bps is None
    if objective is OptimizationObjective.MAXIMIZE_PACKETS_PER_WINDOW:
        return prediction.predicted_packets_per_window is None
    if objective is OptimizationObjective.MINIMIZE_EXPECTED_DELIVERY_LATENCY:
        return prediction.expected_delivery_latency_ms is None
    if objective is OptimizationObjective.MINIMIZE_ENERGY_PER_ATTEMPT:
        return prediction.energy_per_attempt_millijoules is None
    if objective is OptimizationObjective.BALANCED_WEIGHTED_SCORE:
        weights = policy.balanced_weights
        if weights is None:
            raise UnsupportedConstraintError("balanced objective requires policy weights")
        return (
            (weights.goodput_bps > 0.0 and prediction.predicted_goodput_bps is None)
            or (
                weights.packets_per_window > 0.0 and prediction.predicted_packets_per_window is None
            )
            or (
                weights.expected_delivery_latency > 0.0
                and prediction.expected_delivery_latency_ms is None
            )
            or (
                weights.energy_per_attempt > 0.0
                and prediction.energy_per_attempt_millijoules is None
            )
        )
    if objective is OptimizationObjective.MINIMIZE_PER:
        return False
    raise UnsupportedConstraintError(f"unsupported objective: {objective}")


class DeterministicDecisionEngine:
    """Filter hard constraints, optimize admissible modes, and audit every outcome."""

    version = DECISION_ENGINE_VERSION

    def decide(
        self,
        link_state: LinkState,
        mission_policy: MissionPolicy,
        predictions: tuple[RadioPrediction, ...],
        available_modes: frozenset[RadioMode] | None = None,
    ) -> DecisionResult:
        available = available_modes if available_modes is not None else frozenset(RADIO_MODE_ORDER)
        required_fallbacks = {mission_policy.fallback_mode}
        if mission_policy.low_elevation_safety is not None:
            required_fallbacks.add(mission_policy.low_elevation_safety.forced_mode)
        unavailable_fallbacks = required_fallbacks - available
        if unavailable_fallbacks:
            names = ", ".join(sorted(mode.value for mode in unavailable_fallbacks))
            raise FallbackUnavailableError(f"configured safety fallback unavailable: {names}")

        prediction_by_mode: dict[RadioMode, RadioPrediction] = {}
        for input_prediction in predictions:
            if input_prediction.radio_mode in prediction_by_mode:
                raise DuplicatePredictionError(
                    f"duplicate prediction for {input_prediction.radio_mode.value}"
                )
            prediction_by_mode[input_prediction.radio_mode] = input_prediction

        audit = _AuditBuilder()
        audit.add(
            AuditEventType.VALIDATION,
            ReasonCode.INPUT_VALIDATED,
            "Validated link state, mission policy, availability, and prediction-set identity.",
            details=(
                MetadataItem(key="prediction_count", value=len(predictions)),
                MetadataItem(key="message_priority", value=mission_policy.message_priority.value),
            ),
        )

        all_constraints = _standard_constraints(mission_policy)
        constraint_evaluations: list[ConstraintEvaluation] = []
        rejections: dict[RadioMode, list[ReasonCode]] = {mode: [] for mode in RADIO_MODE_ORDER}
        rejection_constraint_ids: dict[RadioMode, list[str]] = {
            mode: [] for mode in RADIO_MODE_ORDER
        }
        global_reasons: list[ReasonCode] = []
        safe_fallback_triggered = False

        for mode in RADIO_MODE_ORDER:
            prediction = prediction_by_mode.get(mode)
            if mode not in mission_policy.allowed_radio_modes:
                _append_unique(rejections[mode], ReasonCode.MODE_FORBIDDEN)
                audit.add(
                    AuditEventType.PREDICTION,
                    ReasonCode.MODE_FORBIDDEN,
                    "Rejected mode because the policy forbids it.",
                    mode,
                )
                continue
            if mode not in available:
                _append_unique(rejections[mode], ReasonCode.MODE_UNAVAILABLE)
                audit.add(
                    AuditEventType.PREDICTION,
                    ReasonCode.MODE_UNAVAILABLE,
                    "Rejected mode because it is unavailable for this decision.",
                    mode,
                )
                continue
            if prediction is None:
                _append_unique(rejections[mode], ReasonCode.MISSING_PREDICTION)
                audit.add(
                    AuditEventType.PREDICTION,
                    ReasonCode.MISSING_PREDICTION,
                    "Rejected mode because no prediction was supplied.",
                    mode,
                )
                for constraint in all_constraints:
                    if constraint.modes is not None and mode not in constraint.modes:
                        continue
                    constraint_evaluations.append(
                        ConstraintEvaluation(
                            constraint_id=constraint.constraint_id,
                            radio_mode=mode,
                            metric=constraint.metric,
                            operator=constraint.operator,
                            threshold=constraint.threshold,
                            unit=constraint.unit,
                            observed_value=None,
                            status=ConstraintEvaluationStatus.NOT_EVALUATED,
                            kind=constraint.kind,
                            reason_code=ReasonCode.MISSING_PREDICTION_METRIC,
                            explanation=(
                                "Constraint was not evaluated because the mode "
                                "prediction is missing."
                            ),
                        )
                    )
                continue

            if prediction.ood.status is not DistributionStatus.IN_DISTRIBUTION:
                if mission_policy.uncertainty_action is UncertaintyAction.REJECT:
                    _append_unique(rejections[mode], ReasonCode.OOD_REJECTED)
                    audit.add(
                        AuditEventType.PREDICTION,
                        ReasonCode.OOD_REJECTED,
                        "Rejected prediction because it is not in-distribution.",
                        mode,
                        (
                            MetadataItem(
                                key="distribution_status",
                                value=prediction.ood.status.value,
                            ),
                        ),
                    )
                elif mission_policy.uncertainty_action is UncertaintyAction.ALLOW_WITH_WARNING:
                    _append_unique(global_reasons, ReasonCode.OOD_ALLOWED_WITH_WARNING)
                    audit.add(
                        AuditEventType.PREDICTION,
                        ReasonCode.OOD_ALLOWED_WITH_WARNING,
                        "Allowed non-in-distribution prediction with a policy warning.",
                        mode,
                        (
                            MetadataItem(
                                key="distribution_status",
                                value=prediction.ood.status.value,
                            ),
                        ),
                    )
                else:
                    safe_fallback_triggered = True
                    _append_unique(rejections[mode], ReasonCode.OOD_SAFE_FALLBACK)
                    _append_unique(global_reasons, ReasonCode.OOD_SAFE_FALLBACK)
                    audit.add(
                        AuditEventType.FALLBACK,
                        ReasonCode.OOD_SAFE_FALLBACK,
                        "Prediction distribution status triggered the configured safe fallback.",
                        mode,
                        (
                            MetadataItem(
                                key="distribution_status",
                                value=prediction.ood.status.value,
                            ),
                        ),
                    )

            for constraint in all_constraints:
                if constraint.modes is not None and mode not in constraint.modes:
                    continue
                value = _constraint_value(constraint, link_state, prediction)
                if value is None:
                    constraint_evaluations.append(
                        ConstraintEvaluation(
                            constraint_id=constraint.constraint_id,
                            radio_mode=mode,
                            metric=constraint.metric,
                            operator=constraint.operator,
                            threshold=constraint.threshold,
                            unit=constraint.unit,
                            observed_value=None,
                            status=ConstraintEvaluationStatus.NOT_EVALUATED,
                            kind=constraint.kind,
                            reason_code=ReasonCode.MISSING_PREDICTION_METRIC,
                            explanation=(
                                "Constraint was not evaluated because its prediction "
                                "metric is missing."
                            ),
                            predictor_version=prediction.predictor_version,
                        )
                    )
                    if constraint.kind is ConstraintKind.HARD:
                        _append_unique(rejections[mode], ReasonCode.MISSING_CONSTRAINT_DATA)
                        _append_unique(
                            rejection_constraint_ids[mode],
                            constraint.constraint_id,
                        )
                        audit.add(
                            AuditEventType.CONSTRAINT,
                            ReasonCode.MISSING_CONSTRAINT_DATA,
                            "Rejected mode because data required by a hard constraint is missing.",
                            mode,
                            (MetadataItem(key="constraint_id", value=constraint.constraint_id),),
                        )
                    else:
                        _append_unique(global_reasons, ReasonCode.SOFT_CONSTRAINT_NOT_EVALUATED)
                        audit.add(
                            AuditEventType.CONSTRAINT,
                            ReasonCode.SOFT_CONSTRAINT_NOT_EVALUATED,
                            "Soft constraint could not be evaluated because data is missing.",
                            mode,
                            (MetadataItem(key="constraint_id", value=constraint.constraint_id),),
                        )
                    continue
                if _compare(value, constraint.operator, constraint.threshold):
                    constraint_evaluations.append(
                        ConstraintEvaluation(
                            constraint_id=constraint.constraint_id,
                            radio_mode=mode,
                            metric=constraint.metric,
                            operator=constraint.operator,
                            threshold=constraint.threshold,
                            unit=constraint.unit,
                            observed_value=value,
                            status=ConstraintEvaluationStatus.PASSED,
                            kind=constraint.kind,
                            reason_code=ReasonCode.CONSTRAINT_PASSED,
                            explanation="Observed value satisfies the configured constraint.",
                            predictor_version=prediction.predictor_version,
                        )
                    )
                    continue
                if constraint.kind is ConstraintKind.HARD:
                    failure = _specific_failure_code(constraint)
                    constraint_evaluations.append(
                        ConstraintEvaluation(
                            constraint_id=constraint.constraint_id,
                            radio_mode=mode,
                            metric=constraint.metric,
                            operator=constraint.operator,
                            threshold=constraint.threshold,
                            unit=constraint.unit,
                            observed_value=value,
                            status=ConstraintEvaluationStatus.FAILED,
                            kind=constraint.kind,
                            reason_code=failure,
                            explanation="Observed value violates a hard constraint.",
                            predictor_version=prediction.predictor_version,
                        )
                    )
                    _append_unique(rejections[mode], failure)
                    _append_unique(rejection_constraint_ids[mode], constraint.constraint_id)
                    audit.add(
                        AuditEventType.CONSTRAINT,
                        failure,
                        "Rejected mode because a hard constraint failed.",
                        mode,
                        (
                            MetadataItem(key="constraint_id", value=constraint.constraint_id),
                            MetadataItem(key="observed", value=value),
                            MetadataItem(key="threshold", value=constraint.threshold),
                        ),
                    )
                else:
                    constraint_evaluations.append(
                        ConstraintEvaluation(
                            constraint_id=constraint.constraint_id,
                            radio_mode=mode,
                            metric=constraint.metric,
                            operator=constraint.operator,
                            threshold=constraint.threshold,
                            unit=constraint.unit,
                            observed_value=value,
                            status=ConstraintEvaluationStatus.FAILED,
                            kind=constraint.kind,
                            reason_code=ReasonCode.SOFT_CONSTRAINT_VIOLATED,
                            explanation=(
                                "Observed value violates a soft constraint; "
                                "admissibility is unchanged."
                            ),
                            predictor_version=prediction.predictor_version,
                        )
                    )
                    _append_unique(global_reasons, ReasonCode.SOFT_CONSTRAINT_VIOLATED)
                    audit.add(
                        AuditEventType.CONSTRAINT,
                        ReasonCode.SOFT_CONSTRAINT_VIOLATED,
                        "Recorded a soft-constraint violation without changing admissibility.",
                        mode,
                        (MetadataItem(key="constraint_id", value=constraint.constraint_id),),
                    )

            if _objective_data_missing(mission_policy, prediction):
                _append_unique(rejections[mode], ReasonCode.MISSING_OBJECTIVE_DATA)
                audit.add(
                    AuditEventType.OPTIMIZATION,
                    ReasonCode.MISSING_OBJECTIVE_DATA,
                    "Rejected mode because data required by the configured objective is missing.",
                    mode,
                )

        admissible = tuple(
            mode for mode in RADIO_MODE_ORDER if mode in prediction_by_mode and not rejections[mode]
        )
        for mode in admissible:
            audit.add(
                AuditEventType.CONSTRAINT,
                ReasonCode.MODE_ADMISSIBLE,
                "Mode passed every applicable hard constraint.",
                mode,
            )

        low_rule = mission_policy.low_elevation_safety
        if low_rule is not None and link_state.elevation_deg < low_rule.below_elevation_deg:
            selected_mode = low_rule.forced_mode
            fallback_applied = True
            _append_unique(global_reasons, ReasonCode.LOW_ELEVATION_SAFETY_FALLBACK)
            _append_unique(global_reasons, ReasonCode.FALLBACK_APPLIED)
            audit.add(
                AuditEventType.FALLBACK,
                ReasonCode.LOW_ELEVATION_SAFETY_FALLBACK,
                "Applied the configured low-elevation safety mode.",
                selected_mode,
                (
                    MetadataItem(key="elevation_deg", value=link_state.elevation_deg),
                    MetadataItem(key="threshold_deg", value=low_rule.below_elevation_deg),
                ),
            )
        elif safe_fallback_triggered:
            selected_mode = mission_policy.fallback_mode
            fallback_applied = True
            _append_unique(global_reasons, ReasonCode.FALLBACK_APPLIED)
            audit.add(
                AuditEventType.FALLBACK,
                ReasonCode.FALLBACK_APPLIED,
                "Applied the configured fallback after an OOD policy trigger.",
                selected_mode,
            )
        elif admissible:
            selected_mode, tied = self._optimize(
                mission_policy,
                tuple(prediction_by_mode[mode] for mode in admissible),
            )
            fallback_applied = False
            if tied:
                _append_unique(global_reasons, ReasonCode.DETERMINISTIC_TIE_BREAK)
                audit.add(
                    AuditEventType.OPTIMIZATION,
                    ReasonCode.DETERMINISTIC_TIE_BREAK,
                    "Resolved an objective tie with deterministic secondary keys.",
                    selected_mode,
                )
        else:
            selected_mode = mission_policy.fallback_mode
            fallback_applied = True
            _append_unique(global_reasons, ReasonCode.NO_ADMISSIBLE_MODE)
            _append_unique(global_reasons, ReasonCode.FALLBACK_APPLIED)
            audit.add(
                AuditEventType.FALLBACK,
                ReasonCode.NO_ADMISSIBLE_MODE,
                "No mode passed all hard constraints.",
            )
            audit.add(
                AuditEventType.FALLBACK,
                ReasonCode.FALLBACK_APPLIED,
                "Applied the configured safety fallback.",
                selected_mode,
            )

        selected_prediction = prediction_by_mode.get(selected_mode)
        if not fallback_applied:
            status = ConstraintSatisfactionStatus.SATISFIED
        elif selected_prediction is None:
            status = ConstraintSatisfactionStatus.FALLBACK_UNVERIFIED
            _append_unique(global_reasons, ReasonCode.FALLBACK_PREDICTION_MISSING)
            audit.add(
                AuditEventType.FALLBACK,
                ReasonCode.FALLBACK_PREDICTION_MISSING,
                "Fallback was selected without prediction data.",
                selected_mode,
            )
        elif selected_mode in admissible:
            status = ConstraintSatisfactionStatus.SATISFIED
        else:
            status = ConstraintSatisfactionStatus.FALLBACK_WITH_VIOLATIONS

        _append_unique(global_reasons, ReasonCode.MODE_SELECTED)
        audit.add(
            AuditEventType.SELECTION,
            ReasonCode.MODE_SELECTED,
            "Selected the final radio mode deterministically.",
            selected_mode,
            (
                MetadataItem(
                    key="optimization_objective",
                    value=mission_policy.optimization_objective.value,
                ),
                MetadataItem(key="fallback_applied", value=fallback_applied),
            ),
        )

        rejected_modes = tuple(
            ModeRejection(
                radio_mode=mode,
                reason_codes=tuple(rejections[mode]),
                constraint_ids=tuple(rejection_constraint_ids[mode]),
            )
            for mode in RADIO_MODE_ORDER
            if rejections[mode]
        )
        for rejection in rejected_modes:
            for reason in rejection.reason_codes:
                _append_unique(global_reasons, reason)

        provenance = tuple(
            PredictionProvenance(
                radio_mode=mode,
                predictor_version=prediction_by_mode[mode].predictor_version,
            )
            for mode in RADIO_MODE_ORDER
            if mode in prediction_by_mode
        )
        return DecisionResult(
            selected_mode=selected_mode,
            selected_prediction=selected_prediction,
            admissible_modes=admissible,
            rejected_modes=rejected_modes,
            constraint_satisfaction_status=status,
            fallback_applied=fallback_applied,
            reason_codes=tuple(global_reasons),
            audit_trace=audit.result(),
            policy_version=mission_policy.version,
            prediction_provenance=provenance,
            constraint_evaluations=tuple(constraint_evaluations),
            deterministic_decision_engine_version=self.version,
        )

    def _optimize(
        self,
        policy: MissionPolicy,
        predictions: tuple[RadioPrediction, ...],
    ) -> tuple[RadioMode, bool]:
        objective = policy.optimization_objective
        primary: dict[RadioMode, float]

        if objective is OptimizationObjective.MAXIMIZE_GOODPUT_BPS:
            primary = {
                item.radio_mode: _required(item.predicted_goodput_bps) for item in predictions
            }
            selected = min(
                predictions,
                key=lambda item: (
                    -primary[item.radio_mode],
                    item.predicted_per,
                    RADIO_MODE_RANK[item.radio_mode],
                ),
            )
            best_primary = max(primary.values())
        elif objective is OptimizationObjective.MAXIMIZE_PACKETS_PER_WINDOW:
            primary = {
                item.radio_mode: _required(item.predicted_packets_per_window)
                for item in predictions
            }
            selected = min(
                predictions,
                key=lambda item: (
                    -primary[item.radio_mode],
                    item.predicted_per,
                    RADIO_MODE_RANK[item.radio_mode],
                ),
            )
            best_primary = max(primary.values())
        elif objective is OptimizationObjective.MINIMIZE_PER:
            primary = {item.radio_mode: item.predicted_per for item in predictions}
            selected = min(
                predictions,
                key=lambda item: (
                    item.predicted_per,
                    -_optional_for_tie(item.predicted_goodput_bps),
                    RADIO_MODE_RANK[item.radio_mode],
                ),
            )
            best_primary = min(primary.values())
        elif objective is OptimizationObjective.MINIMIZE_EXPECTED_DELIVERY_LATENCY:
            primary = {
                item.radio_mode: _required(item.expected_delivery_latency_ms)
                for item in predictions
            }
            selected = min(
                predictions,
                key=lambda item: (
                    primary[item.radio_mode],
                    item.predicted_per,
                    RADIO_MODE_RANK[item.radio_mode],
                ),
            )
            best_primary = min(primary.values())
        elif objective is OptimizationObjective.MINIMIZE_ENERGY_PER_ATTEMPT:
            primary = {
                item.radio_mode: _required(item.energy_per_attempt_millijoules)
                for item in predictions
            }
            selected = min(
                predictions,
                key=lambda item: (
                    primary[item.radio_mode],
                    item.predicted_per,
                    RADIO_MODE_RANK[item.radio_mode],
                ),
            )
            best_primary = min(primary.values())
        elif objective is OptimizationObjective.BALANCED_WEIGHTED_SCORE:
            weights = policy.balanced_weights
            if weights is None:
                raise UnsupportedConstraintError("balanced objective requires policy weights")
            max_goodput = max(_optional_for_tie(item.predicted_goodput_bps) for item in predictions)
            max_packets = max(
                _optional_for_tie(item.predicted_packets_per_window) for item in predictions
            )
            latencies = [
                _required(item.expected_delivery_latency_ms)
                for item in predictions
                if weights.expected_delivery_latency > 0.0
            ]
            energies = [
                _required(item.energy_per_attempt_millijoules)
                for item in predictions
                if weights.energy_per_attempt > 0.0
            ]
            min_latency = min(latencies) if latencies else 0.0
            min_energy = min(energies) if energies else 0.0

            def score(item: RadioPrediction) -> float:
                goodput = _normalized_max(item.predicted_goodput_bps, max_goodput)
                packets = _normalized_max(item.predicted_packets_per_window, max_packets)
                latency = _normalized_min(item.expected_delivery_latency_ms, min_latency)
                energy = _normalized_min(item.energy_per_attempt_millijoules, min_energy)
                return (
                    weights.goodput_bps * goodput
                    + weights.packets_per_window * packets
                    + weights.reliability * (1.0 - item.predicted_per)
                    + weights.expected_delivery_latency * latency
                    + weights.energy_per_attempt * energy
                )

            primary = {item.radio_mode: score(item) for item in predictions}
            selected = min(
                predictions,
                key=lambda item: (
                    -primary[item.radio_mode],
                    item.predicted_per,
                    RADIO_MODE_RANK[item.radio_mode],
                ),
            )
            best_primary = max(primary.values())
        else:  # Defensive against future enum values with an older engine.
            raise UnsupportedConstraintError(f"unsupported objective: {objective}")

        tied = sum(value == best_primary for value in primary.values()) > 1
        return selected.radio_mode, tied


def _required(value: float | None) -> float:
    if value is None:
        raise UnsupportedConstraintError("required objective data was not filtered")
    return value


def _optional_for_tie(value: float | None) -> float:
    return value if value is not None else 0.0


def _normalized_max(value: float | None, maximum: float) -> float:
    if value is None:
        return 0.0
    return value / maximum if maximum > 0.0 else 1.0


def _normalized_min(value: float | None, minimum: float) -> float:
    if value is None:
        return 0.0
    if minimum == 0.0:
        return 1.0 if value == 0.0 else 0.0
    return minimum / value
