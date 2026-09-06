from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from meseleo_core import (
    RADIO_MODE_ORDER,
    ChromaV1Retriever,
    ConfidenceStatus,
    DistributionStatus,
    LinkState,
    MalformedLLMResponseError,
    MissingVectorStoreError,
    OllamaLLMProvider,
    PredictionTimeoutError,
    PredictorRegistry,
    PromptTechnique,
    RadioMode,
    RadioScenario,
    RagProviderUnavailableError,
    RetrievedContext,
    RetrievedModeContext,
    V1DatasetConfiguration,
    V1KnnPredictionEngine,
    V1RagConfiguration,
    V1RagPredictionEngine,
    build_v1_predictor_registry,
    parse_v1_per_response,
)

VALID_RESPONSE = (
    '{"per_sf7":5,"per_sf8":4,"per_sf9":3,"per_sf10":2,'
    '"per_sf11":1,"per_sf12":0.5,"per_dr8":0.2,"per_dr9":0.4}'
)


def test_base_core_import_does_not_load_optional_rag_packages() -> None:
    assert "chromadb" not in sys.modules
    assert "langchain_ollama" not in sys.modules


def retrieved_context() -> tuple[RetrievedContext, ...]:
    return (
        RetrievedContext(
            context_id="v1-state-45-4-50000",
            llm_text="Complete eight-mode V1 pooled simulation state.",
            elevation_deg=45.0,
            v_rel_km_s=4.0,
            active_nodes=50_000,
            distance=0.01,
            modes=tuple(
                RetrievedModeContext(
                    radio_mode=mode,
                    per_percent=float(index + 1),
                    airtime_seconds=0.05 + index * 0.01,
                    max_packets_per_window=1_000.0 - index * 50.0,
                )
                for index, mode in enumerate(RADIO_MODE_ORDER)
            ),
        ),
    )


class FakeRetriever:
    def __init__(self, contexts: tuple[RetrievedContext, ...] | None = None) -> None:
        self.contexts = contexts if contexts is not None else retrieved_context()
        self.calls = 0

    def retrieve(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        k: int,
    ) -> tuple[RetrievedContext, ...]:
        self.calls += 1
        assert k == 3
        return self.contexts


class FakeLLM:
    def __init__(self, response: str = VALID_RESPONSE, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.prompts: list[str] = []

    def complete(self, prompt: str, timeout_seconds: float) -> str:
        self.prompts.append(prompt)
        assert timeout_seconds == 5.0
        if self.error is not None:
            raise self.error
        return self.response


def rag_configuration(tmp_path: Path, technique: PromptTechnique) -> V1RagConfiguration:
    return V1RagConfiguration(
        vector_store_path=tmp_path / "unused-in-injected-test",
        dataset_version="v1-fixture",
        prompt_technique=technique,
        timeout_seconds=5.0,
        model_name="mocked-v1-model",
        embedding_model="mocked-v1-embedding",
    )


@pytest.mark.parametrize("technique", tuple(PromptTechnique))
def test_mocked_rag_supports_all_v1_prompt_techniques(
    technique: PromptTechnique,
    tmp_path: Path,
    link_state: LinkState,
    radio_scenario: RadioScenario,
) -> None:
    llm = FakeLLM()
    engine = V1RagPredictionEngine(
        rag_configuration(tmp_path, technique),
        retrieval_provider=FakeRetriever(),
        llm_provider=llm,
    )

    predictions = engine.predict(link_state, radio_scenario, RADIO_MODE_ORDER)

    assert tuple(item.radio_mode for item in predictions) == RADIO_MODE_ORDER
    assert predictions[0].predicted_per == 0.05
    assert predictions[-1].predicted_per == 0.004
    assert predictions[0].predicted_packets_per_window == 1_000.0
    assert predictions[0].confidence is None
    assert predictions[0].confidence_status is ConfidenceStatus.UNAVAILABLE
    assert predictions[0].ood.status is DistributionStatus.UNCERTAIN
    assert f"Technique: {technique.value}" in llm.prompts[0]
    assert engine.last_retrieved_context == retrieved_context()


@pytest.mark.parametrize(
    ("response", "message"),
    (
        ("no object", "no JSON"),
        ('{"per_sf7":1}', "all eight modes"),
        (
            VALID_RESPONSE.replace('"per_sf8":4', '"per_sf7":4'),
            "duplicate JSON key",
        ),
        (VALID_RESPONSE.replace('"per_dr9":0.4', '"per_dr9":101'), r"in \[0, 100\]"),
        (VALID_RESPONSE.replace('"per_sf10":2', '"per_sf10":"2"'), "must be numeric"),
    ),
)
def test_rag_parser_rejects_malformed_output(response: str, message: str) -> None:
    with pytest.raises(MalformedLLMResponseError, match=message):
        parse_v1_per_response(response)


def test_rag_parser_allows_cot_provenance_field() -> None:
    parsed = parse_v1_per_response(
        VALID_RESPONSE.replace("{", '{"interpolation_logic":"internal",', 1)
    )

    assert parsed[RadioMode.LR_FHSS_DR8] == 0.002


def test_empty_retrieval_raises_missing_store(
    tmp_path: Path,
    link_state: LinkState,
    radio_scenario: RadioScenario,
) -> None:
    engine = V1RagPredictionEngine(
        rag_configuration(tmp_path, PromptTechnique.ZERO_SHOT),
        retrieval_provider=FakeRetriever(()),
        llm_provider=FakeLLM(),
    )

    with pytest.raises(MissingVectorStoreError):
        engine.predict(link_state, radio_scenario, RADIO_MODE_ORDER)


def test_provider_timeout_is_structured(
    tmp_path: Path,
    link_state: LinkState,
    radio_scenario: RadioScenario,
) -> None:
    engine = V1RagPredictionEngine(
        rag_configuration(tmp_path, PromptTechnique.ZERO_SHOT),
        retrieval_provider=FakeRetriever(),
        llm_provider=FakeLLM(error=TimeoutError("mock timeout")),
    )

    with pytest.raises(PredictionTimeoutError):
        engine.predict(link_state, radio_scenario, RADIO_MODE_ORDER)


def test_provider_failure_is_not_silently_replaced_by_knn(
    tmp_path: Path,
    link_state: LinkState,
    radio_scenario: RadioScenario,
) -> None:
    engine = V1RagPredictionEngine(
        rag_configuration(tmp_path, PromptTechnique.ZERO_SHOT),
        retrieval_provider=FakeRetriever(),
        llm_provider=FakeLLM(error=RuntimeError("model unavailable")),
    )

    with pytest.raises(RagProviderUnavailableError):
        engine.predict(link_state, radio_scenario, RADIO_MODE_ORDER)


def test_ollama_import_is_lazy_and_missing_dependency_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "langchain_ollama":
            raise ImportError("intentionally absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RagProviderUnavailableError):
        OllamaLLMProvider("not-installed", 0.0).complete("prompt", 1.0)


def test_predictor_registry_is_framework_independent() -> None:
    registry = PredictorRegistry()
    marker = object()
    registry.register("knn", lambda: marker)

    assert registry.names == ("knn",)
    assert registry.create("knn") is marker


def test_v1_registry_selects_all_required_predictor_modes(
    tmp_path: Path,
    v1_knn: V1KnnPredictionEngine,
) -> None:
    engines = tuple(
        V1RagPredictionEngine(
            rag_configuration(tmp_path, technique),
            retrieval_provider=FakeRetriever(),
            llm_provider=FakeLLM(),
        )
        for technique in PromptTechnique
    )
    registry = build_v1_predictor_registry(v1_knn, *engines)

    assert registry.names == ("knn", "rag-cot", "rag-few-shot", "rag-zero-shot")
    assert registry.create("knn") is v1_knn
    assert registry.create("rag-zero-shot") is engines[0]
    assert registry.create("rag-few-shot") is engines[1]
    assert registry.create("rag-cot") is engines[2]


def test_real_retriever_reproduces_v1_enriched_query_text(
    tmp_path: Path,
    scientific_fixture_path: Path,
    link_state: LinkState,
    radio_scenario: RadioScenario,
) -> None:
    configuration = rag_configuration(tmp_path, PromptTechnique.ZERO_SHOT).model_copy(
        update={
            "state_dataset": V1DatasetConfiguration(
                path=scientific_fixture_path,
                version="v1-scientific-fixture-1",
                sha256_digest=("474f20043c479852ae96a0e72f2fbc1b93323dad8e306c3e36fa36ac8e3779e6"),
            )
        }
    )
    retriever = ChromaV1Retriever(configuration, embedding_provider=object())  # type: ignore[arg-type]

    text = retriever.build_query_text(link_state, radio_scenario)

    assert text == (
        "LEO uplink at elevation 45° (mid), relative speed 4.0 km/s (mid Doppler), "
        "network of 50000 nodes (busy). Link: distance 683 km, kappa 12.5, "
        "RSSI -123 dBm, SNR LoRa -6 dB / LR-FHSS 18 dB."
    )
