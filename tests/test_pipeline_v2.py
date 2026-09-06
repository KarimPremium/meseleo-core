from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from meseleo_core import (
    RADIO_MODE_ORDER,
    ConstraintSatisfactionStatus,
    DefaultPolicyEngine,
    DeterministicDecisionEngine,
    GroundStation,
    IncompletePredictionSetError,
    LinkState,
    MeseleoPipeline,
    MessagePriority,
    OperationalRequest,
    PipelineStage,
    PromptTechnique,
    RadioMode,
    RadioScenario,
    ReasonCode,
    RetrievedContext,
    RetrievedModeContext,
    StructuredExplanation,
    TemplateExplanationProvider,
    TLEInput,
    TLEStateRequest,
    V1ATCommandAdapter,
    V1GeometryAdapter,
    V1KnnPredictionEngine,
    V1RagConfiguration,
    V1RagPredictionEngine,
)

RAG_RESPONSE = (
    '{"per_sf7":5,"per_sf8":4,"per_sf9":3,"per_sf10":2,'
    '"per_sf11":1,"per_sf12":0.5,"per_dr8":0.2,"per_dr9":0.4}'
)


class PipelineRetriever:
    def retrieve(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        k: int,
    ) -> tuple[RetrievedContext, ...]:
        return (
            RetrievedContext(
                context_id="pipeline-rag-fixture",
                llm_text="Mocked complete V1 state",
                elevation_deg=link_state.elevation_deg,
                v_rel_km_s=abs(link_state.radial_speed_km_s),
                active_nodes=link_state.active_nodes,
                distance=0.0,
                modes=tuple(
                    RetrievedModeContext(
                        radio_mode=mode,
                        per_percent=1.0,
                        airtime_seconds=0.05 + index * 0.01,
                        max_packets_per_window=10_000.0 - index * 100.0,
                    )
                    for index, mode in enumerate(RADIO_MODE_ORDER)
                ),
            ),
        )


class PipelineLLM:
    def complete(self, prompt: str, timeout_seconds: float) -> str:
        return RAG_RESPONSE


class CountingExplanationProvider:
    def __init__(self) -> None:
        self.calls = 0

    def explain(self, context: object) -> StructuredExplanation:
        self.calls += 1
        return StructuredExplanation(
            summary="Explanation observes but cannot replace DecisionResult.",
            dominant_factors=("fixture",),
            constraint_reasoning=("exact engine already completed",),
            provenance=("test-provider",),
        )


def build_pipeline(
    predictor: object,
    explanation_provider: object | None = None,
) -> MeseleoPipeline:
    return MeseleoPipeline(
        geometry_engine=V1GeometryAdapter(),
        prediction_engine=predictor,  # type: ignore[arg-type]
        policy_engine=DefaultPolicyEngine(),
        decision_engine=DeterministicDecisionEngine(),
        command_adapter=V1ATCommandAdapter(),
        explanation_provider=explanation_provider,  # type: ignore[arg-type]
    )


def test_manual_knn_exact_decision_and_command(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    pipeline = build_pipeline(v1_knn)

    result = pipeline.run_manual_state(
        LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000),
        radio_scenario,
        OperationalRequest(message_priority=MessagePriority.ROUTINE),
    )

    assert len(result.predictions) == 8
    assert result.decision.selected_mode is RadioMode.LR_FHSS_DR9
    assert result.command.command == "AT+MOD=LR-FHSS-DR9"
    assert not result.decision.fallback_applied
    assert result.decision.constraint_satisfaction_status is ConstraintSatisfactionStatus.SATISFIED


def test_low_elevation_configured_dr8_fallback(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    result = build_pipeline(v1_knn).run_manual_state(
        LinkState(elevation_deg=5.0, radial_speed_km_s=0.0, active_nodes=10_000),
        radio_scenario,
        OperationalRequest(message_priority=MessagePriority.ROUTINE),
    )

    assert result.decision.selected_mode is RadioMode.LR_FHSS_DR8
    assert result.decision.fallback_applied
    assert ReasonCode.LOW_ELEVATION_SAFETY_FALLBACK in result.decision.reason_codes


def test_fixed_tle_knn_pipeline_acceptance_path(
    v1_knn: V1KnnPredictionEngine,
    fixed_tle: TLEInput,
    paris_station: GroundStation,
    radio_scenario: RadioScenario,
) -> None:
    result = build_pipeline(v1_knn).run_tle_state(
        TLEStateRequest(
            tle=fixed_tle,
            ground_station=paris_station,
            timestamp=datetime(2024, 1, 1, 13, 25, tzinfo=UTC),
            active_nodes=50_000,
            minimum_elevation_deg=5.0,
        ),
        radio_scenario,
        OperationalRequest(message_priority=MessagePriority.ROUTINE),
    )

    assert result.geometry.visible
    assert result.geometry.link_state.satellite_id == fixed_tle.satellite_id
    assert len(result.predictions) == 8
    assert result.command.radio_mode is result.decision.selected_mode


def test_mocked_rag_exact_decision_command_and_explanation(
    tmp_path: Path,
    radio_scenario: RadioScenario,
) -> None:
    rag = V1RagPredictionEngine(
        V1RagConfiguration(
            vector_store_path=tmp_path / "not-read",
            dataset_version="mocked-v1",
            prompt_technique=PromptTechnique.ZERO_SHOT,
        ),
        retrieval_provider=PipelineRetriever(),
        llm_provider=PipelineLLM(),
    )
    pipeline = build_pipeline(rag, TemplateExplanationProvider())

    result = pipeline.run_manual_state(
        LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000),
        radio_scenario,
        OperationalRequest(message_priority=MessagePriority.ROUTINE),
    )

    assert result.decision.selected_mode is RadioMode.LORA_SF7
    assert result.command.command == "AT+MOD=LORA-SF7"
    assert result.explanation is not None
    assert "exact deterministic" in result.explanation.summary
    assert result.explanation.warnings == (
        "predictor confidence is unavailable",
        "a non-in-distribution prediction was allowed by policy",
    )


def test_no_admissible_mode_uses_configured_fallback(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    result = build_pipeline(v1_knn).run_manual_state(
        LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000),
        radio_scenario,
        OperationalRequest(
            message_priority=MessagePriority.ROUTINE,
            maximum_admissible_per=0.0,
        ),
    )

    assert result.decision.selected_mode is RadioMode.LR_FHSS_DR8
    assert result.decision.fallback_applied
    assert (
        result.decision.constraint_satisfaction_status
        is ConstraintSatisfactionStatus.FALLBACK_WITH_VIOLATIONS
    )
    assert ReasonCode.NO_ADMISSIBLE_MODE in result.decision.reason_codes


def test_ood_safe_fallback_is_exact_and_explicit(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    result = build_pipeline(v1_knn).run_manual_state(
        LinkState(elevation_deg=90.0, radial_speed_km_s=8.0, active_nodes=200_000),
        radio_scenario,
        OperationalRequest(message_priority=MessagePriority.CRITICAL),
    )

    assert result.decision.selected_mode is RadioMode.LR_FHSS_DR8
    assert result.decision.fallback_applied
    assert ReasonCode.OOD_SAFE_FALLBACK in result.decision.reason_codes


def test_incomplete_prediction_set_fails_before_decision(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    class IncompletePredictor:
        def predict(
            self,
            link_state: LinkState,
            scenario: RadioScenario,
            radio_modes: tuple[RadioMode, ...],
        ) -> object:
            return v1_knn.predict(link_state, scenario, radio_modes[:-1])

    with pytest.raises(IncompletePredictionSetError):
        build_pipeline(IncompletePredictor()).run_manual_state(
            LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000),
            radio_scenario,
            OperationalRequest(message_priority=MessagePriority.ROUTINE),
        )


def test_pipeline_is_reproducible_and_audit_chain_is_complete(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    pipeline = build_pipeline(v1_knn)
    inputs = (
        LinkState(elevation_deg=27.0, radial_speed_km_s=2.5, active_nodes=10_000),
        radio_scenario,
        OperationalRequest(message_priority=MessagePriority.ROUTINE),
    )

    first = pipeline.run_manual_state(*inputs)
    second = pipeline.run_manual_state(*inputs)

    assert first == second
    assert tuple(item.sequence for item in first.audit_trace) == tuple(
        range(len(first.audit_trace))
    )
    assert tuple(item.stage for item in first.audit_trace) == (
        PipelineStage.GEOMETRY,
        PipelineStage.ELIGIBILITY,
        PipelineStage.POLICY,
        PipelineStage.PREDICTION,
        PipelineStage.DECISION,
        PipelineStage.COMMAND,
    )
    assert tuple(item.radio_mode for item in first.decision.prediction_provenance) == (
        RADIO_MODE_ORDER
    )


def test_explanation_boundary_cannot_change_selected_decision(
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    provider = CountingExplanationProvider()
    state = LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000)
    request = OperationalRequest(message_priority=MessagePriority.ROUTINE)

    exact_only = build_pipeline(v1_knn).run_manual_state(state, radio_scenario, request)
    explained = build_pipeline(v1_knn, provider).run_manual_state(
        state,
        radio_scenario,
        request,
    )

    assert provider.calls == 1
    assert explained.decision == exact_only.decision
    assert explained.command == exact_only.command
    assert explained.explanation is not None
