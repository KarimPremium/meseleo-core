"""Strict immutable domain contracts for the framework-independent MESELEO core."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer, model_validator

type MetadataValue = str | int | float | bool | None
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


class FrozenModel(BaseModel):
    """Base model with no coercion, extra fields, mutation, NaN, or infinity."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class MetadataItem(FrozenModel):
    key: Annotated[str, Field(min_length=1)]
    value: MetadataValue


class ModelVersion(FrozenModel):
    name: Annotated[str, Field(min_length=1)]
    version: Annotated[str, Field(min_length=1)]
    digest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None


class RadioMode(StrEnum):
    LORA_SF7 = "lora_sf7"
    LORA_SF8 = "lora_sf8"
    LORA_SF9 = "lora_sf9"
    LORA_SF10 = "lora_sf10"
    LORA_SF11 = "lora_sf11"
    LORA_SF12 = "lora_sf12"
    LR_FHSS_DR8 = "lr_fhss_dr8"
    LR_FHSS_DR9 = "lr_fhss_dr9"


RADIO_MODE_ORDER: tuple[RadioMode, ...] = tuple(RadioMode)
RADIO_MODE_RANK: dict[RadioMode, int] = {mode: rank for rank, mode in enumerate(RADIO_MODE_ORDER)}


def _require_aware(value: datetime | None, field_name: str) -> None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{field_name} must be timezone-aware")


class VisibilityWindow(FrozenModel):
    start: datetime
    end: datetime
    window_id: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        _require_aware(self.start, "start")
        _require_aware(self.end, "end")
        if self.end <= self.start:
            raise ValueError("visibility window end must be after start")
        return self


class LinkAvailabilityStatus(StrEnum):
    VISIBLE = "visible"
    BELOW_HORIZON = "below_horizon"
    BELOW_ELEVATION_MASK = "below_elevation_mask"
    OUTSIDE_VISIBILITY_WINDOW = "outside_visibility_window"
    UNKNOWN = "unknown"


class LinkAvailability(FrozenModel):
    """Geometry-owned visibility result; it does not select a radio mode."""

    status: LinkAvailabilityStatus
    elevation_deg: Annotated[float, Field(ge=-90.0, le=90.0)]
    minimum_elevation_deg: Annotated[float, Field(ge=-90.0, le=90.0)] | None = None
    source: Annotated[str, Field(min_length=1)]
    details: tuple[MetadataItem, ...] = ()


class TransmissionEligibilityStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class TransmissionEligibilityReason(StrEnum):
    LINK_WINDOW_VALID = "link_window_valid"
    MANUAL_STATE_EXPLICIT = "manual_state_explicit"
    NO_LINK_WINDOW = "no_link_window"
    LINK_NOT_VISIBLE = "link_not_visible"
    BELOW_ELEVATION_MASK = "below_elevation_mask"
    OUTSIDE_VISIBILITY_WINDOW = "outside_visibility_window"
    LINK_AVAILABILITY_UNKNOWN = "link_availability_unknown"
    RESEARCH_ONLY_NON_VISIBLE_STATE = "research_only_non_visible_state"


class TransmissionEligibility(FrozenModel):
    """Validated permission boundary between mode selection and command generation."""

    status: TransmissionEligibilityStatus
    eligible: bool
    link_availability: LinkAvailability
    reason_codes: tuple[TransmissionEligibilityReason, ...]
    research_only: bool = False

    @model_validator(mode="after")
    def validate_eligibility(self) -> Self:
        if not self.reason_codes or len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("transmission eligibility requires unique reason codes")
        if self.eligible is not (self.status is TransmissionEligibilityStatus.READY):
            raise ValueError("eligible must match transmission eligibility status")
        if self.eligible and self.research_only:
            raise ValueError("research-only analysis cannot be transmission-ready")
        return self


class GeometryRequest(FrozenModel):
    timestamp: datetime
    satellite_id: Annotated[str, Field(min_length=1)]
    ground_station_id: Annotated[str, Field(min_length=1)]
    metadata: tuple[MetadataItem, ...] = ()

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        _require_aware(self.timestamp, "timestamp")
        return self


class LinkState(FrozenModel):
    elevation_deg: Annotated[float, Field(ge=-90.0, le=90.0)]
    radial_speed_km_s: float
    active_nodes: Annotated[int, Field(ge=0)]
    timestamp: datetime | None = None
    satellite_id: Annotated[str, Field(min_length=1)] | None = None
    ground_station_id: Annotated[str, Field(min_length=1)] | None = None
    # RSSI and SNR are descriptive context only: the V1 k-NN predictor keys on elevation, radial
    # speed, and active nodes exclusively (see V1KnnPredictionEngine). These values are recorded and
    # displayed for interpretation but are NOT model inputs and never change a prediction.
    rssi_dbm: float | None = Field(
        default=None,
        description=(
            "Descriptive only: measured/estimated RSSI in dBm. Not a predictor feature; the V1 "
            "k-NN model keys solely on elevation, radial speed, and active nodes."
        ),
    )
    snr_db: float | None = Field(
        default=None,
        description=(
            "Descriptive only: measured/estimated SNR in dB. Not a predictor feature; the V1 "
            "k-NN model keys solely on elevation, radial speed, and active nodes."
        ),
    )
    doppler_hz: float | None = None
    visibility_window: VisibilityWindow | None = None
    link_availability_status: LinkAvailabilityStatus | None = None
    metadata: tuple[MetadataItem, ...] = ()

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        _require_aware(self.timestamp, "timestamp")
        return self


class TLEInput(FrozenModel):
    satellite_id: Annotated[str, Field(min_length=1)]
    line1: Annotated[str, Field(min_length=1)]
    line2: Annotated[str, Field(min_length=1)]
    source: Annotated[str, Field(min_length=1)] | None = None
    epoch_provenance: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def validate_lines(self) -> Self:
        if not self.line1.startswith("1 ") or not self.line2.startswith("2 "):
            raise ValueError("TLE lines must start with '1 ' and '2 ' respectively")
        return self


class GroundStation(FrozenModel):
    ground_station_id: Annotated[str, Field(min_length=1)]
    latitude_deg: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude_deg: Annotated[float, Field(ge=-180.0, le=180.0)]
    altitude_m: float = 0.0
    metadata: tuple[MetadataItem, ...] = ()


class TLEStateRequest(FrozenModel):
    tle: TLEInput
    ground_station: GroundStation
    timestamp: datetime
    active_nodes: Annotated[int, Field(ge=0)]
    minimum_elevation_deg: Annotated[float, Field(ge=-90.0, le=90.0)] = 0.0
    metadata: tuple[MetadataItem, ...] = ()

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        _require_aware(self.timestamp, "timestamp")
        return self


class SatellitePassRequest(FrozenModel):
    tle: TLEInput
    ground_station: GroundStation
    start: datetime
    end: datetime
    active_nodes: Annotated[int, Field(ge=0)]
    minimum_elevation_deg: Annotated[float, Field(ge=-90.0, le=90.0)] = 0.0
    sampling_interval_seconds: PositiveFloat = 30.0
    metadata: tuple[MetadataItem, ...] = ()

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        _require_aware(self.start, "start")
        _require_aware(self.end, "end")
        if self.end <= self.start:
            raise ValueError("satellite-pass end must be after start")
        return self


class PassState(FrozenModel):
    timestamp: datetime
    elevation_deg: Annotated[float, Field(ge=-90.0, le=90.0)]
    azimuth_deg: Annotated[float, Field(ge=0.0, lt=360.0)]
    slant_range_km: PositiveFloat
    radial_speed_km_s: float
    doppler_hz: float
    visible: bool

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        _require_aware(self.timestamp, "timestamp")
        return self


class SatellitePass(FrozenModel):
    satellite_id: Annotated[str, Field(min_length=1)]
    ground_station_id: Annotated[str, Field(min_length=1)]
    acquisition_of_signal: datetime
    loss_of_signal: datetime
    culmination: datetime
    maximum_elevation_deg: Annotated[float, Field(ge=-90.0, le=90.0)]
    duration_seconds: PositiveFloat
    states: tuple[PassState, ...]
    metadata: tuple[MetadataItem, ...] = ()

    @model_validator(mode="after")
    def validate_pass(self) -> Self:
        for name, value in (
            ("acquisition_of_signal", self.acquisition_of_signal),
            ("loss_of_signal", self.loss_of_signal),
            ("culmination", self.culmination),
        ):
            _require_aware(value, name)
        if self.loss_of_signal <= self.acquisition_of_signal:
            raise ValueError("loss_of_signal must be after acquisition_of_signal")
        if not self.acquisition_of_signal <= self.culmination <= self.loss_of_signal:
            raise ValueError("culmination must fall inside the pass")
        if not self.states:
            raise ValueError("a satellite pass requires sampled states")
        timestamps = tuple(state.timestamp for state in self.states)
        if timestamps != tuple(sorted(timestamps)) or len(timestamps) != len(set(timestamps)):
            raise ValueError("pass states must have unique increasing timestamps")
        if not all(state.visible for state in self.states):
            raise ValueError("satellite-pass states must satisfy the elevation mask")
        expected_duration = (self.loss_of_signal - self.acquisition_of_signal).total_seconds()
        if abs(self.duration_seconds - expected_duration) > 1e-6:
            raise ValueError("duration_seconds must match acquisition/loss timestamps")
        return self


class GeometryResult(FrozenModel):
    link_state: LinkState
    azimuth_deg: Annotated[float, Field(ge=0.0, lt=360.0)] | None = None
    slant_range_km: PositiveFloat | None = None
    visible: bool
    link_availability: LinkAvailability
    metadata: tuple[MetadataItem, ...] = ()

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        expected_visible = self.link_availability.status is LinkAvailabilityStatus.VISIBLE
        if self.visible is not expected_visible:
            raise ValueError("visible must match link_availability.status")
        return self


class RadioScenario(FrozenModel):
    """Physical and payload context required by a prediction implementation."""

    carrier_frequency_hz: PositiveFloat
    payload_bytes: Annotated[int, Field(ge=0)]
    transmit_power_dbm: float
    bandwidth_hz: PositiveFloat | None = None
    channel_plan: Annotated[str, Field(min_length=1)] | None = None
    device_profile_id: Annotated[str, Field(min_length=1)] | None = None
    metadata: tuple[MetadataItem, ...] = ()


class DistributionStatus(StrEnum):
    IN_DISTRIBUTION = "in_distribution"
    UNCERTAIN = "uncertain"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    INSUFFICIENT_DATA = "insufficient_data"


class OutOfDistributionResult(FrozenModel):
    status: DistributionStatus
    score: Ratio | None = None
    detector_version: ModelVersion | None = None
    reasons: tuple[str, ...] = ()


class ConfidenceStatus(StrEnum):
    UNAVAILABLE = "unavailable"
    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"


class RadioPrediction(FrozenModel):
    radio_mode: RadioMode
    predicted_per: Ratio
    predicted_goodput_bps: NonNegativeFloat | None = Field(
        default=None,
        validation_alias=AliasChoices("predicted_goodput_bps", "predicted_throughput_bps"),
    )
    predicted_packets_per_window: NonNegativeFloat | None = None
    airtime_ms: PositiveFloat | None = None
    expected_delivery_latency_ms: NonNegativeFloat | None = Field(
        default=None,
        validation_alias=AliasChoices("expected_delivery_latency_ms", "latency_ms"),
    )
    energy_per_attempt_millijoules: NonNegativeFloat | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "energy_per_attempt_millijoules",
            "energy_millijoules",
        ),
    )
    confidence: Ratio | None = None
    confidence_status: ConfidenceStatus = ConfidenceStatus.UNAVAILABLE
    predictor_version: ModelVersion
    ood: OutOfDistributionResult = OutOfDistributionResult(
        status=DistributionStatus.IN_DISTRIBUTION
    )
    metadata: tuple[MetadataItem, ...] = ()

    @model_validator(mode="after")
    def validate_confidence_contract(self) -> Self:
        if self.confidence_status is ConfidenceStatus.UNAVAILABLE and self.confidence is not None:
            raise ValueError("unavailable confidence status requires confidence=None")
        if self.confidence_status is not ConfidenceStatus.UNAVAILABLE and self.confidence is None:
            raise ValueError("uncalibrated or calibrated confidence status requires a value")
        return self

    @property
    def is_out_of_distribution(self) -> bool:
        """Derived convenience flag; ``ood.status`` remains the source of truth."""

        return self.ood.status is DistributionStatus.OUT_OF_DISTRIBUTION


class MessagePriority(StrEnum):
    ROUTINE = "routine"
    IMPORTANT = "important"
    URGENT = "urgent"
    CRITICAL = "critical"


class EnergyMode(StrEnum):
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    CONSERVE_ENERGY = "conserve_energy"


class ProfileSelectionMode(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class OptimizationObjective(StrEnum):
    MAXIMIZE_GOODPUT_BPS = "maximize_goodput_bps"
    MAXIMIZE_PACKETS_PER_WINDOW = "maximize_packets_per_window"
    MINIMIZE_PER = "minimize_per"
    MINIMIZE_EXPECTED_DELIVERY_LATENCY = "minimize_expected_delivery_latency_ms"
    MINIMIZE_ENERGY_PER_ATTEMPT = "minimize_energy_per_attempt_millijoules"
    BALANCED_WEIGHTED_SCORE = "balanced_weighted_score"


class OperationalRequest(FrozenModel):
    """Operational intent to be compiled into a generic mission policy."""

    message_priority: MessagePriority
    deadline: datetime | None = None
    delivery_reliability_target: Ratio | None = None
    maximum_admissible_per: Ratio | None = None
    energy_mode: EnergyMode | None = None
    profile_selection_mode: ProfileSelectionMode = ProfileSelectionMode.AUTOMATIC
    policy_profile: Annotated[str, Field(min_length=1)] | None = None
    optimization_objective: OptimizationObjective | None = None
    allowed_radio_modes: frozenset[RadioMode] | None = None
    fallback_mode: RadioMode | None = None
    low_elevation_threshold_deg: Annotated[float, Field(ge=-90.0, le=90.0)] | None = None
    organization_id: Annotated[str, Field(min_length=1)] | None = None
    site_id: Annotated[str, Field(min_length=1)] | None = None
    device_id: Annotated[str, Field(min_length=1)] | None = None
    metadata: tuple[MetadataItem, ...] = ()

    @field_serializer("allowed_radio_modes", when_used="json")
    def serialize_allowed_radio_modes(
        self,
        value: frozenset[RadioMode] | None,
    ) -> tuple[RadioMode, ...] | None:
        if value is None:
            return None
        return tuple(mode for mode in RADIO_MODE_ORDER if mode in value)

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_explicit_profile(cls, value: object) -> object:
        if (
            isinstance(value, dict)
            and value.get("policy_profile") is not None
            and "profile_selection_mode" not in value
        ):
            migrated = dict(value)
            migrated["profile_selection_mode"] = ProfileSelectionMode.MANUAL
            return migrated
        return value

    @model_validator(mode="after")
    def validate_deadline(self) -> Self:
        _require_aware(self.deadline, "deadline")
        if self.allowed_radio_modes is not None and not self.allowed_radio_modes:
            raise ValueError("allowed_radio_modes cannot be an empty set")
        if (
            self.allowed_radio_modes is not None
            and self.fallback_mode is not None
            and self.fallback_mode not in self.allowed_radio_modes
        ):
            raise ValueError("fallback_mode must be allowed")
        if self.profile_selection_mode is ProfileSelectionMode.MANUAL:
            if self.policy_profile is None:
                raise ValueError("manual profile selection requires policy_profile")
        elif self.policy_profile is not None:
            raise ValueError("automatic profile selection must not include policy_profile")
        return self


class UncertaintyAction(StrEnum):
    REJECT = "reject"
    ALLOW_WITH_WARNING = "allow_with_warning"
    SAFE_FALLBACK = "safe_fallback"


class BalancedWeights(FrozenModel):
    goodput_bps: NonNegativeFloat
    packets_per_window: NonNegativeFloat
    reliability: NonNegativeFloat
    expected_delivery_latency: NonNegativeFloat
    energy_per_attempt: NonNegativeFloat

    @model_validator(mode="after")
    def require_nonzero_weight(self) -> Self:
        total = (
            self.goodput_bps
            + self.packets_per_window
            + self.reliability
            + self.expected_delivery_latency
            + self.energy_per_attempt
        )
        if total <= 0.0:
            raise ValueError("at least one balanced-score weight must be greater than zero")
        return self


class LowElevationSafetyRule(FrozenModel):
    below_elevation_deg: Annotated[float, Field(ge=-90.0, le=90.0)]
    forced_mode: RadioMode


class ConstraintKind(StrEnum):
    HARD = "hard"
    SOFT = "soft"


class ConstraintMetric(StrEnum):
    PREDICTED_PER = "predicted_per"
    PREDICTED_GOODPUT_BPS = "predicted_goodput_bps"
    PREDICTED_PACKETS_PER_WINDOW = "predicted_packets_per_window"
    EXPECTED_DELIVERY_LATENCY_MS = "expected_delivery_latency_ms"
    ENERGY_PER_ATTEMPT_MILLIJOULES = "energy_per_attempt_millijoules"
    CONFIDENCE = "confidence"
    ELEVATION_DEG = "elevation_deg"


class ConstraintOperator(StrEnum):
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    EQ = "eq"


_EXPECTED_UNITS: dict[ConstraintMetric, str] = {
    ConstraintMetric.PREDICTED_PER: "ratio",
    ConstraintMetric.PREDICTED_GOODPUT_BPS: "bit/s",
    ConstraintMetric.PREDICTED_PACKETS_PER_WINDOW: "packets/window",
    ConstraintMetric.EXPECTED_DELIVERY_LATENCY_MS: "ms",
    ConstraintMetric.ENERGY_PER_ATTEMPT_MILLIJOULES: "mJ/attempt",
    ConstraintMetric.CONFIDENCE: "ratio",
    ConstraintMetric.ELEVATION_DEG: "deg",
}


class DecisionConstraint(FrozenModel):
    constraint_id: Annotated[str, Field(min_length=1)]
    kind: ConstraintKind
    metric: ConstraintMetric
    operator: ConstraintOperator
    threshold: float
    unit: Annotated[str, Field(min_length=1)]
    modes: frozenset[RadioMode] | None = None
    description: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        expected = _EXPECTED_UNITS[self.metric]
        if self.unit != expected:
            raise ValueError(f"{self.metric.value} constraints must use unit {expected!r}")
        if self.modes is not None and not self.modes:
            raise ValueError("constraint modes cannot be an empty set")
        if self.metric in {ConstraintMetric.PREDICTED_PER, ConstraintMetric.CONFIDENCE}:
            if not 0.0 <= self.threshold <= 1.0:
                raise ValueError(f"{self.metric.value} threshold must be in [0, 1]")
        elif self.metric is ConstraintMetric.ELEVATION_DEG:
            if not -90.0 <= self.threshold <= 90.0:
                raise ValueError("elevation_deg threshold must be in [-90, 90]")
        elif self.threshold < 0.0:
            raise ValueError(f"{self.metric.value} threshold must be greater than or equal to 0")
        return self


class PolicyResolution(FrozenModel):
    requested_mode: ProfileSelectionMode
    resolved_profile: Annotated[str, Field(min_length=1)]
    resolution_reason: Annotated[str, Field(min_length=1)]
    source_fields: tuple[Annotated[str, Field(min_length=1)], ...]
    warnings: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    profile_default_maximum_per: Ratio
    requested_maximum_per_override: Ratio | None = None
    effective_maximum_per: Ratio


class MissionPolicy(FrozenModel):
    policy_name: Annotated[str, Field(min_length=1)]
    version: ModelVersion
    message_priority: MessagePriority
    maximum_admissible_per: Ratio
    maximum_expected_delivery_latency_ms: NonNegativeFloat | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "maximum_expected_delivery_latency_ms",
            "maximum_latency_ms",
        ),
    )
    maximum_energy_per_attempt_millijoules: NonNegativeFloat | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "maximum_energy_per_attempt_millijoules",
            "maximum_energy_millijoules",
        ),
    )
    optimization_objective: OptimizationObjective
    allowed_radio_modes: frozenset[RadioMode]
    fallback_mode: RadioMode
    minimum_confidence: Ratio | None = None
    uncertainty_action: UncertaintyAction = UncertaintyAction.REJECT
    low_elevation_safety: LowElevationSafetyRule | None = None
    balanced_weights: BalancedWeights | None = None
    constraints: tuple[DecisionConstraint, ...] = ()
    source_request: OperationalRequest | None = None
    policy_resolution: PolicyResolution | None = None
    assumptions: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    metadata: tuple[MetadataItem, ...] = ()

    @field_serializer("allowed_radio_modes", when_used="json")
    def serialize_allowed_radio_modes(
        self,
        value: frozenset[RadioMode],
    ) -> tuple[RadioMode, ...]:
        return tuple(mode for mode in RADIO_MODE_ORDER if mode in value)

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if not self.allowed_radio_modes:
            raise ValueError("allowed_radio_modes cannot be empty")
        if self.fallback_mode not in self.allowed_radio_modes:
            raise ValueError("fallback_mode must be in allowed_radio_modes")
        if (
            self.low_elevation_safety is not None
            and self.low_elevation_safety.forced_mode not in self.allowed_radio_modes
        ):
            raise ValueError("low-elevation forced mode must be allowed")
        is_balanced = self.optimization_objective is OptimizationObjective.BALANCED_WEIGHTED_SCORE
        if is_balanced and self.balanced_weights is None:
            raise ValueError("balanced_weights are required for balanced_weighted_score")
        if not is_balanced and self.balanced_weights is not None:
            raise ValueError("balanced_weights are only supported by balanced_weighted_score")
        constraint_ids = [item.constraint_id for item in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint_id values must be unique")
        return self


class ReasonCode(StrEnum):
    INPUT_VALIDATED = "input_validated"
    MODE_ADMISSIBLE = "mode_admissible"
    MODE_FORBIDDEN = "mode_forbidden"
    MODE_UNAVAILABLE = "mode_unavailable"
    MISSING_PREDICTION = "missing_prediction"
    PER_EXCEEDS_MAXIMUM = "per_exceeds_maximum"
    LATENCY_EXCEEDS_MAXIMUM = "latency_exceeds_maximum"
    ENERGY_EXCEEDS_MAXIMUM = "energy_exceeds_maximum"
    CONFIDENCE_BELOW_MINIMUM = "confidence_below_minimum"
    OOD_REJECTED = "ood_rejected"
    OOD_ALLOWED_WITH_WARNING = "ood_allowed_with_warning"
    OOD_SAFE_FALLBACK = "ood_safe_fallback"
    MISSING_CONSTRAINT_DATA = "missing_constraint_data"
    HARD_CONSTRAINT_FAILED = "hard_constraint_failed"
    SOFT_CONSTRAINT_VIOLATED = "soft_constraint_violated"
    SOFT_CONSTRAINT_NOT_EVALUATED = "soft_constraint_not_evaluated"
    CONSTRAINT_PASSED = "constraint_passed"
    MISSING_PREDICTION_METRIC = "missing_prediction_metric"
    MISSING_OBJECTIVE_DATA = "missing_objective_data"
    DETERMINISTIC_TIE_BREAK = "deterministic_tie_break"
    MODE_SELECTED = "mode_selected"
    NO_ADMISSIBLE_MODE = "no_admissible_mode"
    FALLBACK_APPLIED = "fallback_applied"
    FALLBACK_PREDICTION_MISSING = "fallback_prediction_missing"
    LOW_ELEVATION_SAFETY_FALLBACK = "low_elevation_safety_fallback"


class ConstraintEvaluationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


class ConstraintEvaluation(FrozenModel):
    constraint_id: Annotated[str, Field(min_length=1)]
    radio_mode: RadioMode
    metric: ConstraintMetric
    operator: ConstraintOperator
    threshold: float
    unit: Annotated[str, Field(min_length=1)]
    observed_value: float | None
    status: ConstraintEvaluationStatus
    kind: ConstraintKind
    reason_code: ReasonCode
    explanation: Annotated[str, Field(min_length=1)]
    predictor_version: ModelVersion | None = None

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        if self.status is ConstraintEvaluationStatus.NOT_EVALUATED:
            if self.observed_value is not None:
                raise ValueError("not_evaluated requires observed_value=None")
            if self.reason_code is not ReasonCode.MISSING_PREDICTION_METRIC:
                raise ValueError("not_evaluated requires missing_prediction_metric")
        elif self.observed_value is None:
            raise ValueError("passed or failed constraints require an observed value")
        return self


class AuditEventType(StrEnum):
    VALIDATION = "validation"
    PREDICTION = "prediction"
    CONSTRAINT = "constraint"
    OPTIMIZATION = "optimization"
    FALLBACK = "fallback"
    SELECTION = "selection"


class AuditEvent(FrozenModel):
    sequence: Annotated[int, Field(ge=0)]
    event_type: AuditEventType
    reason_code: ReasonCode
    message: Annotated[str, Field(min_length=1)]
    radio_mode: RadioMode | None = None
    details: tuple[MetadataItem, ...] = ()


class ModeRejection(FrozenModel):
    radio_mode: RadioMode
    reason_codes: tuple[ReasonCode, ...]
    constraint_ids: tuple[Annotated[str, Field(min_length=1)], ...] = ()

    @model_validator(mode="after")
    def validate_rejection(self) -> Self:
        if not self.reason_codes:
            raise ValueError("a rejected mode requires at least one reason")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("rejection reason_codes must be unique")
        if len(self.constraint_ids) != len(set(self.constraint_ids)):
            raise ValueError("rejection constraint_ids must be unique")
        return self


class ConstraintSatisfactionStatus(StrEnum):
    SATISFIED = "satisfied"
    FALLBACK_WITH_VIOLATIONS = "fallback_with_violations"
    FALLBACK_UNVERIFIED = "fallback_unverified"


class PredictionProvenance(FrozenModel):
    radio_mode: RadioMode
    predictor_version: ModelVersion


class DecisionResult(FrozenModel):
    selected_mode: RadioMode
    selected_prediction: RadioPrediction | None
    admissible_modes: tuple[RadioMode, ...]
    rejected_modes: tuple[ModeRejection, ...]
    constraint_satisfaction_status: ConstraintSatisfactionStatus
    fallback_applied: bool
    reason_codes: tuple[ReasonCode, ...]
    audit_trace: tuple[AuditEvent, ...]
    policy_version: ModelVersion
    prediction_provenance: tuple[PredictionProvenance, ...]
    constraint_evaluations: tuple[ConstraintEvaluation, ...] = ()
    deterministic_decision_engine_version: ModelVersion

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (
            self.selected_prediction is not None
            and self.selected_prediction.radio_mode is not self.selected_mode
        ):
            raise ValueError("selected_prediction must match selected_mode")
        if not self.reason_codes:
            raise ValueError("decision result requires at least one reason code")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("decision reason_codes must be unique")
        if ReasonCode.MODE_SELECTED not in self.reason_codes:
            raise ValueError("decision reason_codes must include mode_selected")

        if len(self.admissible_modes) != len(set(self.admissible_modes)):
            raise ValueError("admissible_modes cannot contain duplicates")
        if tuple(sorted(self.admissible_modes, key=RADIO_MODE_RANK.__getitem__)) != (
            self.admissible_modes
        ):
            raise ValueError("admissible_modes must use deterministic radio-mode order")

        rejected_mode_values = tuple(item.radio_mode for item in self.rejected_modes)
        if len(rejected_mode_values) != len(set(rejected_mode_values)):
            raise ValueError("rejected_modes cannot contain duplicate radio modes")
        if tuple(sorted(rejected_mode_values, key=RADIO_MODE_RANK.__getitem__)) != (
            rejected_mode_values
        ):
            raise ValueError("rejected_modes must use deterministic radio-mode order")
        if set(self.admissible_modes) & set(rejected_mode_values):
            raise ValueError("a radio mode cannot be both admissible and rejected")
        rejection_reasons = {
            reason for rejection in self.rejected_modes for reason in rejection.reason_codes
        }
        if not rejection_reasons.issubset(set(self.reason_codes)):
            raise ValueError("decision reason_codes must include every rejection reason")

        provenance_modes = tuple(item.radio_mode for item in self.prediction_provenance)
        if len(provenance_modes) != len(set(provenance_modes)):
            raise ValueError("prediction_provenance cannot repeat a radio mode")
        if tuple(sorted(provenance_modes, key=RADIO_MODE_RANK.__getitem__)) != provenance_modes:
            raise ValueError("prediction_provenance must use deterministic radio-mode order")
        if self.selected_prediction is not None:
            matching = [
                item for item in self.prediction_provenance if item.radio_mode is self.selected_mode
            ]
            if not matching or matching[0].predictor_version != (
                self.selected_prediction.predictor_version
            ):
                raise ValueError("selected prediction must match prediction provenance")

        expected_sequences = tuple(range(len(self.audit_trace)))
        actual_sequences = tuple(item.sequence for item in self.audit_trace)
        if actual_sequences != expected_sequences:
            raise ValueError("audit_trace sequences must be contiguous and start at zero")

        has_fallback_reason = ReasonCode.FALLBACK_APPLIED in self.reason_codes
        if has_fallback_reason is not self.fallback_applied:
            raise ValueError("fallback_applied must match the fallback_applied reason code")
        fallback_only_reasons = {
            ReasonCode.NO_ADMISSIBLE_MODE,
            ReasonCode.OOD_SAFE_FALLBACK,
            ReasonCode.FALLBACK_PREDICTION_MISSING,
            ReasonCode.LOW_ELEVATION_SAFETY_FALLBACK,
        }
        if not self.fallback_applied and fallback_only_reasons & set(self.reason_codes):
            raise ValueError("fallback-only reason code requires fallback_applied")
        if not self.fallback_applied:
            if self.constraint_satisfaction_status is not ConstraintSatisfactionStatus.SATISFIED:
                raise ValueError("a non-fallback decision must satisfy all hard constraints")
            if self.selected_prediction is None or self.selected_mode not in self.admissible_modes:
                raise ValueError("a non-fallback selection must be a predicted admissible mode")
        elif self.constraint_satisfaction_status is ConstraintSatisfactionStatus.SATISFIED:
            if self.selected_prediction is None or self.selected_mode not in self.admissible_modes:
                raise ValueError("a satisfied fallback must be a predicted admissible mode")
        elif (
            self.constraint_satisfaction_status
            is ConstraintSatisfactionStatus.FALLBACK_WITH_VIOLATIONS
        ):
            if self.selected_prediction is None or self.selected_mode not in rejected_mode_values:
                raise ValueError("a fallback with violations must select a rejected prediction")
        else:
            if self.selected_prediction is not None:
                raise ValueError("an unverified fallback cannot have a selected prediction")
            if ReasonCode.FALLBACK_PREDICTION_MISSING not in self.reason_codes:
                raise ValueError("an unverified fallback requires fallback_prediction_missing")
            selected_rejection = next(
                (item for item in self.rejected_modes if item.radio_mode is self.selected_mode),
                None,
            )
            if (
                self.selected_mode in self.admissible_modes
                or selected_rejection is None
                or ReasonCode.MISSING_PREDICTION not in selected_rejection.reason_codes
            ):
                raise ValueError("an unverified fallback must select a missing prediction")
        return self


class CommandValidationStatus(StrEnum):
    VALIDATED = "validated"


class ModemCommand(FrozenModel):
    command: Annotated[str, Field(min_length=1)]
    radio_mode: RadioMode
    adapter_version: ModelVersion
    modem_profile: Annotated[str, Field(min_length=1)] | None = None
    validation_status: CommandValidationStatus = CommandValidationStatus.VALIDATED
    metadata: tuple[MetadataItem, ...] = ()
