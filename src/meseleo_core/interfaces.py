"""Dependency-inversion protocols around the exact MESELEO core."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from .explanations import ExplanationContext, StructuredExplanation
from .models import (
    DecisionResult,
    GeometryResult,
    LinkState,
    MissionPolicy,
    ModemCommand,
    OperationalRequest,
    RadioMode,
    RadioPrediction,
    RadioScenario,
    SatellitePass,
    SatellitePassRequest,
    TLEInput,
    TLEStateRequest,
    TransmissionEligibility,
)


@runtime_checkable
class TLEProvider(Protocol):
    """Optional external TLE source boundary; the core provides no network implementation."""

    def get_tle(self, satellite_id: str, timestamp: datetime | None = None) -> TLEInput: ...


@runtime_checkable
class GeometryEngine(Protocol):
    def map_manual_state(
        self,
        state: LinkState,
        scenario: RadioScenario,
        minimum_elevation_deg: float = 0.0,
    ) -> GeometryResult: ...

    def calculate_tle_state(
        self,
        request: TLEStateRequest,
        scenario: RadioScenario,
    ) -> GeometryResult: ...

    def predict_passes(
        self,
        request: SatellitePassRequest,
        scenario: RadioScenario,
    ) -> tuple[SatellitePass, ...]: ...


@runtime_checkable
class PredictionEngine(Protocol):
    def predict(
        self,
        link_state: LinkState,
        scenario: RadioScenario,
        radio_modes: tuple[RadioMode, ...],
    ) -> tuple[RadioPrediction, ...]: ...


@runtime_checkable
class PolicyEngine(Protocol):
    def compile_policy(
        self,
        request: OperationalRequest,
        link_state: LinkState,
    ) -> MissionPolicy: ...


@runtime_checkable
class DecisionEngine(Protocol):
    def decide(
        self,
        link_state: LinkState,
        mission_policy: MissionPolicy,
        predictions: tuple[RadioPrediction, ...],
        available_modes: frozenset[RadioMode] | None = None,
    ) -> DecisionResult: ...


@runtime_checkable
class ExplanationProvider(Protocol):
    def explain(self, context: ExplanationContext) -> StructuredExplanation: ...


@runtime_checkable
class CommandAdapter(Protocol):
    def translate(
        self,
        decision: DecisionResult,
        transmission_eligibility: TransmissionEligibility,
    ) -> ModemCommand: ...
