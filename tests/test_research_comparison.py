from __future__ import annotations

from pathlib import Path

from meseleo_core import (
    RADIO_MODE_ORDER,
    DefaultPolicyEngine,
    DeterministicDecisionEngine,
    LinkState,
    MessagePriority,
    OperationalRequest,
    PromptTechnique,
    RadioMode,
    RadioScenario,
    ResearchComparisonService,
    RetrievedContext,
    RetrievedModeContext,
    V1KnnPredictionEngine,
    V1RagConfiguration,
    V1RagPredictionEngine,
)


class ComparisonRetriever:
    def retrieve(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        k: int,
    ) -> tuple[RetrievedContext, ...]:
        return (
            RetrievedContext(
                context_id="comparison-context",
                llm_text="Mocked V1 context for research comparison",
                elevation_deg=link_state.elevation_deg,
                v_rel_km_s=abs(link_state.radial_speed_km_s),
                active_nodes=link_state.active_nodes,
                modes=tuple(
                    RetrievedModeContext(
                        radio_mode=mode,
                        per_percent=1.0,
                        airtime_seconds=0.05 + index * 0.01,
                        max_packets_per_window=20_000.0 - index * 1_000.0,
                    )
                    for index, mode in enumerate(RADIO_MODE_ORDER)
                ),
            ),
        )


class ComparisonLLM:
    def complete(self, prompt: str, timeout_seconds: float) -> str:
        return (
            '{"per_sf7":5,"per_sf8":4,"per_sf9":3,"per_sf10":2,'
            '"per_sf11":1,"per_sf12":0.5,"per_dr8":0.2,"per_dr9":0.4}'
        )


def test_research_mode_reports_disagreement_and_oracle(
    tmp_path: Path,
    v1_knn: V1KnnPredictionEngine,
    radio_scenario: RadioScenario,
) -> None:
    rag = V1RagPredictionEngine(
        V1RagConfiguration(
            vector_store_path=tmp_path / "not-read",
            dataset_version="mocked-v1",
            prompt_technique=PromptTechnique.FEW_SHOT,
        ),
        retrieval_provider=ComparisonRetriever(),
        llm_provider=ComparisonLLM(),
    )
    service = ResearchComparisonService(
        predictors=(("knn", v1_knn), ("rag-few-shot", rag)),
        policy_engine=DefaultPolicyEngine(),
        decision_engine=DeterministicDecisionEngine(),
    )
    state = LinkState(elevation_deg=45.0, radial_speed_km_s=4.0, active_nodes=50_000)
    oracle = v1_knn.predict(state, radio_scenario, RADIO_MODE_ORDER)

    result = service.compare(
        state,
        radio_scenario,
        OperationalRequest(message_priority=MessagePriority.ROUTINE),
        oracle_predictions=oracle,
    )

    assert tuple(item.predictor_name for item in result.comparisons) == (
        "knn",
        "rag-few-shot",
    )
    assert result.comparisons[0].decision.selected_mode is RadioMode.LR_FHSS_DR9
    assert result.comparisons[1].decision.selected_mode is RadioMode.LORA_SF7
    assert result.disagreement_matrix[0].selected_modes_differ
    assert result.oracle_mode is RadioMode.LR_FHSS_DR9
    assert result.oracle_matches == (("knn", True), ("rag-few-shot", False))
    assert all(item.latency_ms >= 0.0 for item in result.comparisons)
    assert all(item.predictor_provenance for item in result.comparisons)
