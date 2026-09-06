"""Research-only predictor and decision comparison outside the exact engine."""

from __future__ import annotations

from time import perf_counter_ns
from typing import Annotated

from pydantic import Field

from .interfaces import DecisionEngine, PolicyEngine, PredictionEngine
from .models import (
    RADIO_MODE_ORDER,
    ConstraintSatisfactionStatus,
    DecisionResult,
    FrozenModel,
    LinkState,
    MissionPolicy,
    OperationalRequest,
    PredictionProvenance,
    RadioMode,
    RadioPrediction,
    RadioScenario,
)
from .pipeline import MeseleoPipeline


class PredictorComparison(FrozenModel):
    predictor_name: Annotated[str, Field(min_length=1)]
    predictions: tuple[RadioPrediction, ...]
    decision: DecisionResult
    latency_ms: Annotated[float, Field(ge=0.0)]
    constraint_satisfied: bool
    predictor_provenance: tuple[PredictionProvenance, ...]


class PredictorDisagreement(FrozenModel):
    left_predictor: Annotated[str, Field(min_length=1)]
    right_predictor: Annotated[str, Field(min_length=1)]
    selected_modes_differ: bool


class ResearchComparisonResult(FrozenModel):
    link_state: LinkState
    scenario: RadioScenario
    mission_policy: MissionPolicy
    comparisons: tuple[PredictorComparison, ...]
    disagreement_matrix: tuple[PredictorDisagreement, ...]
    oracle_mode: RadioMode | None = None
    oracle_matches: tuple[tuple[str, bool], ...] = ()


class ResearchComparisonService:
    """Run configured predictors on the same state and compare exact decisions."""

    def __init__(
        self,
        predictors: tuple[tuple[str, PredictionEngine], ...],
        policy_engine: PolicyEngine,
        decision_engine: DecisionEngine,
    ) -> None:
        names = tuple(name for name, _ in predictors)
        if not names or len(names) != len(set(names)):
            raise ValueError("research predictor names must be non-empty and unique")
        self.predictors = predictors
        self.policy_engine = policy_engine
        self.decision_engine = decision_engine

    def compare(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        operational_request: OperationalRequest,
        oracle_predictions: tuple[RadioPrediction, ...] | None = None,
    ) -> ResearchComparisonResult:
        policy = self.policy_engine.compile_policy(operational_request, link_state)
        comparisons: list[PredictorComparison] = []
        for name, predictor in self.predictors:
            started = perf_counter_ns()
            raw = predictor.predict(link_state, scenario, RADIO_MODE_ORDER)
            predictions = MeseleoPipeline._validate_predictions(raw)
            decision = self.decision_engine.decide(link_state, policy, predictions)
            latency_ms = (perf_counter_ns() - started) / 1_000_000.0
            comparisons.append(
                PredictorComparison(
                    predictor_name=name,
                    predictions=predictions,
                    decision=decision,
                    latency_ms=latency_ms,
                    constraint_satisfied=(
                        decision.constraint_satisfaction_status
                        is ConstraintSatisfactionStatus.SATISFIED
                    ),
                    predictor_provenance=decision.prediction_provenance,
                )
            )

        disagreements = tuple(
            PredictorDisagreement(
                left_predictor=left.predictor_name,
                right_predictor=right.predictor_name,
                selected_modes_differ=(
                    left.decision.selected_mode is not right.decision.selected_mode
                ),
            )
            for left_index, left in enumerate(comparisons)
            for right in comparisons[left_index + 1 :]
        )
        oracle_mode: RadioMode | None = None
        oracle_matches: tuple[tuple[str, bool], ...] = ()
        if oracle_predictions is not None:
            oracle = self.decision_engine.decide(
                link_state,
                policy,
                MeseleoPipeline._validate_predictions(oracle_predictions),
            )
            oracle_mode = oracle.selected_mode
            oracle_matches = tuple(
                (item.predictor_name, item.decision.selected_mode is oracle_mode)
                for item in comparisons
            )
        return ResearchComparisonResult(
            link_state=link_state,
            scenario=scenario,
            mission_policy=policy,
            comparisons=tuple(comparisons),
            disagreement_matrix=disagreements,
            oracle_mode=oracle_mode,
            oracle_matches=oracle_matches,
        )
