"""Post-decision explanation contracts and deterministic template provider."""

from __future__ import annotations

from .models import (
    DecisionResult,
    FrozenModel,
    LinkState,
    MissionPolicy,
    OperationalRequest,
    RadioPrediction,
    RadioScenario,
    ReasonCode,
)
from .prediction import RetrievedContext


class ExplanationContext(FrozenModel):
    link_state: LinkState
    scenario: RadioScenario
    operational_request: OperationalRequest
    mission_policy: MissionPolicy
    predictions: tuple[RadioPrediction, ...]
    decision: DecisionResult
    retrieved_context: tuple[RetrievedContext, ...] = ()


class StructuredExplanation(FrozenModel):
    summary: str
    dominant_factors: tuple[str, ...]
    constraint_reasoning: tuple[str, ...]
    fallback_explanation: str | None = None
    warnings: tuple[str, ...] = ()
    provenance: tuple[str, ...]


class TemplateExplanationProvider:
    """Explain an immutable decision without modifying or recomputing it."""

    def explain(self, context: ExplanationContext) -> StructuredExplanation:
        decision = context.decision
        selected = decision.selected_prediction
        factors = (
            f"elevation={context.link_state.elevation_deg} deg",
            f"absolute radial speed={abs(context.link_state.radial_speed_km_s)} km/s",
            f"active nodes={context.link_state.active_nodes}",
            f"policy objective={context.mission_policy.optimization_objective.value}",
        )
        reasoning = tuple(
            f"{rejection.radio_mode.value}: "
            + ", ".join(reason.value for reason in rejection.reason_codes)
            for rejection in decision.rejected_modes
        ) or ("all supplied candidates satisfied applicable hard constraints",)
        warnings: list[str] = []
        if selected is not None and selected.confidence is None:
            warnings.append("predictor confidence is unavailable")
        if ReasonCode.OOD_ALLOWED_WITH_WARNING in decision.reason_codes:
            warnings.append("a non-in-distribution prediction was allowed by policy")
        fallback = None
        if decision.fallback_applied:
            fallback = (
                f"configured fallback selected with status "
                f"{decision.constraint_satisfaction_status.value}"
            )
        return StructuredExplanation(
            summary=(
                f"Selected {decision.selected_mode.value} using the exact deterministic "
                f"decision engine."
            ),
            dominant_factors=factors,
            constraint_reasoning=reasoning,
            fallback_explanation=fallback,
            warnings=tuple(warnings),
            provenance=tuple(
                f"{item.radio_mode.value}:{item.predictor_version.name}@"
                f"{item.predictor_version.version}"
                for item in decision.prediction_provenance
            ),
        )
