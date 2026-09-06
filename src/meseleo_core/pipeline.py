"""Framework-independent dependency-injected MESELEO scientific/product pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field

from .explanations import ExplanationContext, StructuredExplanation
from .integration_errors import IncompletePredictionSetError
from .interfaces import (
    CommandAdapter,
    DecisionEngine,
    ExplanationProvider,
    GeometryEngine,
    PolicyEngine,
    PredictionEngine,
)
from .models import (
    RADIO_MODE_ORDER,
    DecisionResult,
    FrozenModel,
    GeometryResult,
    LinkAvailabilityStatus,
    LinkState,
    MetadataItem,
    MissionPolicy,
    ModemCommand,
    OperationalRequest,
    RadioMode,
    RadioPrediction,
    RadioScenario,
    TLEStateRequest,
    TransmissionEligibility,
    TransmissionEligibilityReason,
    TransmissionEligibilityStatus,
)
from .prediction import RetrievedContext


class PipelineStage(StrEnum):
    GEOMETRY = "geometry"
    ELIGIBILITY = "eligibility"
    POLICY = "policy"
    PREDICTION = "prediction"
    DECISION = "decision"
    COMMAND = "command"
    EXPLANATION = "explanation"


class PipelineAuditEvent(FrozenModel):
    sequence: Annotated[int, Field(ge=0)]
    stage: PipelineStage
    message: Annotated[str, Field(min_length=1)]
    metadata: tuple[MetadataItem, ...] = ()


class PipelineResult(FrozenModel):
    geometry: GeometryResult
    scenario: RadioScenario
    operational_request: OperationalRequest
    mission_policy: MissionPolicy
    predictions: tuple[RadioPrediction, ...] = ()
    decision: DecisionResult | None = None
    transmission_eligibility: TransmissionEligibility
    command: ModemCommand | None = None
    explanation: StructuredExplanation | None = None
    audit_trace: tuple[PipelineAuditEvent, ...]

    @classmethod
    def _decision_required_message(cls) -> str:
        return "transmission-ready or research-only results require a decision"

    @property
    def production_ready(self) -> bool:
        return self.transmission_eligibility.eligible and self.command is not None

    def model_post_init(self, __context: Any) -> None:
        if self.transmission_eligibility.eligible:
            if self.decision is None or self.command is None:
                raise ValueError(self._decision_required_message())
        elif self.command is not None:
            raise ValueError("ineligible transmission cannot contain a modem command")
        if self.transmission_eligibility.research_only and self.decision is None:
            raise ValueError(self._decision_required_message())
        if self.decision is None and (self.predictions or self.explanation is not None):
            raise ValueError(
                "a result without a decision cannot contain predictions or explanation"
            )


class MeseleoPipeline:
    """Run geometry, prediction, policy, exact decision, command, then explanation."""

    def __init__(
        self,
        geometry_engine: GeometryEngine,
        prediction_engine: PredictionEngine,
        policy_engine: PolicyEngine,
        decision_engine: DecisionEngine,
        command_adapter: CommandAdapter,
        explanation_provider: ExplanationProvider | None = None,
    ) -> None:
        self.geometry_engine = geometry_engine
        self.prediction_engine = prediction_engine
        self.policy_engine = policy_engine
        self.decision_engine = decision_engine
        self.command_adapter = command_adapter
        self.explanation_provider = explanation_provider

    def run_manual_state(
        self,
        state: LinkState,
        scenario: RadioScenario,
        operational_request: OperationalRequest,
    ) -> PipelineResult:
        geometry = self.geometry_engine.map_manual_state(state, scenario)
        return self._run(geometry, scenario, operational_request)

    def run_tle_state(
        self,
        request: TLEStateRequest,
        scenario: RadioScenario,
        operational_request: OperationalRequest,
        allow_non_visible_state_for_research: bool = False,
    ) -> PipelineResult:
        geometry = self.geometry_engine.calculate_tle_state(request, scenario)
        return self._run(
            geometry,
            scenario,
            operational_request,
            evaluate_ineligible_state=allow_non_visible_state_for_research,
        )

    def _run(
        self,
        geometry: GeometryResult,
        scenario: RadioScenario,
        operational_request: OperationalRequest,
        evaluate_ineligible_state: bool = True,
    ) -> PipelineResult:
        audit: list[PipelineAuditEvent] = []

        def record(
            stage: PipelineStage, message: str, metadata: tuple[MetadataItem, ...] = ()
        ) -> None:
            audit.append(
                PipelineAuditEvent(
                    sequence=len(audit),
                    stage=stage,
                    message=message,
                    metadata=metadata,
                )
            )

        record(
            PipelineStage.GEOMETRY,
            "Produced a validated link state.",
            (MetadataItem(key="visible", value=geometry.visible),),
        )
        eligibility = self._transmission_eligibility(
            geometry,
            research_only=(not geometry.visible and evaluate_ineligible_state),
        )
        record(
            PipelineStage.ELIGIBILITY,
            (
                "Validated a production-ready link window."
                if eligibility.eligible
                else "Blocked production transmission because no valid link window is available."
            ),
            (
                MetadataItem(key="status", value=eligibility.status.value),
                MetadataItem(
                    key="link_availability",
                    value=eligibility.link_availability.status.value,
                ),
                MetadataItem(key="production_ready", value=eligibility.eligible),
            ),
        )
        policy = self.policy_engine.compile_policy(operational_request, geometry.link_state)
        policy_resolution = policy.policy_resolution
        policy_metadata = [MetadataItem(key="policy_name", value=policy.policy_name)]
        if policy_resolution is not None:
            policy_metadata.extend(
                (
                    MetadataItem(
                        key="profile_selection_mode",
                        value=policy_resolution.requested_mode.value,
                    ),
                    MetadataItem(
                        key="resolved_profile",
                        value=policy_resolution.resolved_profile,
                    ),
                    MetadataItem(
                        key="profile_resolution_reason",
                        value=policy_resolution.resolution_reason,
                    ),
                    MetadataItem(
                        key="profile_resolution_source_fields",
                        value=",".join(policy_resolution.source_fields),
                    ),
                    MetadataItem(
                        key="effective_maximum_per",
                        value=policy_resolution.effective_maximum_per,
                    ),
                    MetadataItem(
                        key="profile_resolution_warnings",
                        value=" | ".join(policy_resolution.warnings),
                    ),
                )
            )
        record(
            PipelineStage.POLICY,
            "Compiled operational intent into a versioned mission policy.",
            tuple(policy_metadata),
        )
        if not eligibility.eligible and not evaluate_ineligible_state:
            record(
                PipelineStage.COMMAND,
                "Suppressed command generation because transmission eligibility is blocked.",
                (
                    MetadataItem(
                        key="blocking_reason",
                        value=eligibility.reason_codes[0].value,
                    ),
                ),
            )
            return PipelineResult(
                geometry=geometry,
                scenario=scenario,
                operational_request=operational_request,
                mission_policy=policy,
                transmission_eligibility=eligibility,
                audit_trace=tuple(audit),
            )
        raw_predictions = self.prediction_engine.predict(
            geometry.link_state,
            scenario,
            RADIO_MODE_ORDER,
        )
        predictions = self._validate_predictions(raw_predictions)
        record(
            PipelineStage.PREDICTION,
            "Produced exactly one prediction for each V1 radio mode.",
            (MetadataItem(key="prediction_count", value=len(predictions)),),
        )
        decision = self.decision_engine.decide(
            geometry.link_state,
            policy,
            predictions,
        )
        record(
            PipelineStage.DECISION,
            "Applied exact deterministic hard constraints and optimization.",
            (MetadataItem(key="selected_mode", value=decision.selected_mode.value),),
        )
        command: ModemCommand | None = None
        if eligibility.eligible:
            command = self.command_adapter.translate(decision, eligibility)
            record(
                PipelineStage.COMMAND,
                "Translated the validated decision into a modem command without re-optimizing.",
                (MetadataItem(key="command", value=command.command),),
            )
        else:
            record(
                PipelineStage.COMMAND,
                "Suppressed command generation for research-only non-visible analysis.",
                (
                    MetadataItem(
                        key="blocking_reason",
                        value=eligibility.reason_codes[0].value,
                    ),
                ),
            )

        explanation: StructuredExplanation | None = None
        if self.explanation_provider is not None:
            retrieved_context = self._retrieved_context()
            explanation = self.explanation_provider.explain(
                ExplanationContext(
                    link_state=geometry.link_state,
                    scenario=scenario,
                    operational_request=operational_request,
                    mission_policy=policy,
                    predictions=predictions,
                    decision=decision,
                    retrieved_context=retrieved_context,
                )
            )
            record(
                PipelineStage.EXPLANATION,
                "Generated a post-decision explanation; the decision remained immutable.",
            )
        return PipelineResult(
            geometry=geometry,
            scenario=scenario,
            operational_request=operational_request,
            mission_policy=policy,
            predictions=predictions,
            decision=decision,
            transmission_eligibility=eligibility,
            command=command,
            explanation=explanation,
            audit_trace=tuple(audit),
        )

    @staticmethod
    def _transmission_eligibility(
        geometry: GeometryResult,
        research_only: bool,
    ) -> TransmissionEligibility:
        availability = geometry.link_availability
        if availability.status is LinkAvailabilityStatus.VISIBLE:
            return TransmissionEligibility(
                status=TransmissionEligibilityStatus.READY,
                eligible=True,
                link_availability=availability,
                reason_codes=(TransmissionEligibilityReason.LINK_WINDOW_VALID,),
            )
        reason = {
            LinkAvailabilityStatus.BELOW_HORIZON: (
                TransmissionEligibilityReason.LINK_NOT_VISIBLE
            ),
            LinkAvailabilityStatus.BELOW_ELEVATION_MASK: (
                TransmissionEligibilityReason.BELOW_ELEVATION_MASK
            ),
            LinkAvailabilityStatus.OUTSIDE_VISIBILITY_WINDOW: (
                TransmissionEligibilityReason.OUTSIDE_VISIBILITY_WINDOW
            ),
            LinkAvailabilityStatus.UNKNOWN: (
                TransmissionEligibilityReason.LINK_AVAILABILITY_UNKNOWN
            ),
        }[availability.status]
        reasons = [TransmissionEligibilityReason.NO_LINK_WINDOW, reason]
        if research_only:
            reasons.append(TransmissionEligibilityReason.RESEARCH_ONLY_NON_VISIBLE_STATE)
        return TransmissionEligibility(
            status=TransmissionEligibilityStatus.BLOCKED,
            eligible=False,
            link_availability=availability,
            reason_codes=tuple(reasons),
            research_only=research_only,
        )

    @staticmethod
    def _validate_predictions(
        predictions: tuple[RadioPrediction, ...],
    ) -> tuple[RadioPrediction, ...]:
        by_mode: dict[RadioMode, RadioPrediction] = {}
        duplicates: list[RadioMode] = []
        for prediction in predictions:
            if prediction.radio_mode in by_mode:
                duplicates.append(prediction.radio_mode)
            by_mode[prediction.radio_mode] = prediction
        missing = set(RADIO_MODE_ORDER) - set(by_mode)
        extra = set(by_mode) - set(RADIO_MODE_ORDER)
        if duplicates or missing or extra or len(predictions) != len(RADIO_MODE_ORDER):
            raise IncompletePredictionSetError(
                "prediction set must contain each V1 mode exactly once; "
                f"duplicates={[mode.value for mode in duplicates]}, "
                f"missing={[mode.value for mode in sorted(missing, key=lambda item: item.value)]}, "
                f"extra={[mode.value for mode in sorted(extra, key=lambda item: item.value)]}"
            )
        return tuple(by_mode[mode] for mode in RADIO_MODE_ORDER)

    def _retrieved_context(self) -> tuple[RetrievedContext, ...]:
        value: Any = getattr(self.prediction_engine, "last_retrieved_context", ())
        return value if isinstance(value, tuple) else ()
