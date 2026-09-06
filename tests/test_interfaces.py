from __future__ import annotations

from meseleo_core import (
    CommandAdapter,
    DecisionResult,
    ExplanationContext,
    ExplanationProvider,
    LinkState,
    MissionPolicy,
    ModelVersion,
    ModemCommand,
    OperationalRequest,
    PolicyEngine,
    PredictionEngine,
    RadioMode,
    RadioPrediction,
    RadioScenario,
    StructuredExplanation,
)


class ExampleCommandAdapter:
    def translate(self, decision: DecisionResult) -> ModemCommand:
        return ModemCommand(
            command=f"MODE={decision.selected_mode.value}",
            radio_mode=decision.selected_mode,
            adapter_version=ModelVersion(name="test-adapter", version="1"),
        )


class ExampleExplanationProvider:
    def explain(self, context: ExplanationContext) -> StructuredExplanation:
        return StructuredExplanation(
            summary=context.decision.selected_mode.value,
            dominant_factors=(),
            constraint_reasoning=(),
            provenance=("structural-test",),
        )


class ExamplePredictionEngine:
    def predict(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        radio_modes: tuple[RadioMode, ...],
    ) -> tuple[RadioPrediction, ...]:
        del link_state, scenario, radio_modes
        return ()


class ExamplePolicyEngine:
    def compile_policy(
        self,
        request: OperationalRequest,
        link_state: LinkState,
    ) -> MissionPolicy:
        del request, link_state
        raise NotImplementedError


def test_command_and_explanation_boundaries_are_structural() -> None:
    assert isinstance(ExampleCommandAdapter(), CommandAdapter)
    assert isinstance(ExampleExplanationProvider(), ExplanationProvider)
    assert isinstance(ExamplePredictionEngine(), PredictionEngine)
    assert isinstance(ExamplePolicyEngine(), PolicyEngine)
