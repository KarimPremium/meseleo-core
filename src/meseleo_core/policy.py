"""Deterministic compilation of operational intent into generic mission policy."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from .models import (
    RADIO_MODE_ORDER,
    BalancedWeights,
    EnergyMode,
    FrozenModel,
    LinkState,
    LowElevationSafetyRule,
    MessagePriority,
    MetadataItem,
    MissionPolicy,
    ModelVersion,
    OperationalRequest,
    OptimizationObjective,
    PolicyResolution,
    ProfileSelectionMode,
    RadioMode,
    UncertaintyAction,
)


class PolicyProfile(FrozenModel):
    name: Annotated[str, Field(min_length=1)]
    version: ModelVersion
    maximum_admissible_per: Annotated[float, Field(ge=0.0, le=1.0)]
    optimization_objective: OptimizationObjective
    allowed_radio_modes: frozenset[RadioMode] = frozenset(RADIO_MODE_ORDER)
    fallback_mode: RadioMode = RadioMode.LR_FHSS_DR8
    low_elevation_threshold_deg: Annotated[float, Field(ge=-90.0, le=90.0)] | None = 15.0
    uncertainty_action: UncertaintyAction = UncertaintyAction.ALLOW_WITH_WARNING
    balanced_weights: BalancedWeights | None = None

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        if not self.allowed_radio_modes or self.fallback_mode not in self.allowed_radio_modes:
            raise ValueError("profile fallback must be an allowed mode")
        balanced = self.optimization_objective is OptimizationObjective.BALANCED_WEIGHTED_SCORE
        if balanced is not (self.balanced_weights is not None):
            raise ValueError("balanced profile objective and weights must be configured together")
        return self


DEFAULT_POLICY_PROFILES: tuple[PolicyProfile, ...] = (
    PolicyProfile(
        name="routine_telemetry",
        version=ModelVersion(name="routine-telemetry", version="1.0.0"),
        maximum_admissible_per=0.10,
        optimization_objective=OptimizationObjective.MAXIMIZE_PACKETS_PER_WINDOW,
    ),
    PolicyProfile(
        name="reliable_telemetry",
        version=ModelVersion(name="reliable-telemetry", version="1.0.0"),
        maximum_admissible_per=0.05,
        optimization_objective=OptimizationObjective.MAXIMIZE_PACKETS_PER_WINDOW,
    ),
    PolicyProfile(
        name="critical_alert",
        version=ModelVersion(name="critical-alert", version="1.0.0"),
        maximum_admissible_per=0.01,
        optimization_objective=OptimizationObjective.MINIMIZE_PER,
        uncertainty_action=UncertaintyAction.SAFE_FALLBACK,
    ),
    PolicyProfile(
        name="battery_saving",
        version=ModelVersion(name="battery-saving", version="1.0.0"),
        maximum_admissible_per=0.10,
        optimization_objective=OptimizationObjective.MINIMIZE_ENERGY_PER_ATTEMPT,
    ),
    PolicyProfile(
        name="maximum_throughput",
        version=ModelVersion(name="maximum-throughput", version="1.0.0"),
        maximum_admissible_per=0.20,
        optimization_objective=OptimizationObjective.MAXIMIZE_GOODPUT_BPS,
    ),
)


class DefaultPolicyEngine:
    """Compile explicit operational requests without an LLM or sector-specific rules."""

    def __init__(self, profiles: tuple[PolicyProfile, ...] = DEFAULT_POLICY_PROFILES) -> None:
        names = tuple(profile.name for profile in profiles)
        if len(names) != len(set(names)):
            raise ValueError("policy profile names must be unique")
        self._profiles = {profile.name: profile for profile in profiles}

    @property
    def profile_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def compile_policy(
        self,
        request: OperationalRequest,
        link_state: LinkState,
    ) -> MissionPolicy:
        profile, resolution_reason, source_fields, resolution_warnings = (
            self._resolve_profile(request)
        )
        assumptions = [
            "delivery reliability is mapped to single-attempt reliability as 1-PER",
            "V1 packet-window capacity is distinct from bit/s goodput",
        ]

        maximum_per = profile.maximum_admissible_per
        if request.maximum_admissible_per is not None:
            if request.maximum_admissible_per > profile.maximum_admissible_per:
                raise ValueError("request cannot weaken the profile maximum PER")
            maximum_per = request.maximum_admissible_per
        if request.delivery_reliability_target is not None:
            maximum_per = min(maximum_per, 1.0 - request.delivery_reliability_target)

        maximum_latency_ms: float | None = None
        if request.deadline is not None:
            if link_state.timestamp is None:
                raise ValueError("a deadline requires a timestamped link state")
            remaining = (request.deadline - link_state.timestamp).total_seconds()
            if remaining <= 0.0:
                raise ValueError("operational deadline must be after link-state timestamp")
            maximum_latency_ms = remaining * 1000.0
            assumptions.append("deadline is compiled as maximum expected delivery latency")

        allowed_modes = profile.allowed_radio_modes
        if request.allowed_radio_modes is not None:
            if not request.allowed_radio_modes.issubset(profile.allowed_radio_modes):
                raise ValueError("request allowed modes must be a subset of the profile")
            allowed_modes = request.allowed_radio_modes
        fallback_mode = request.fallback_mode or profile.fallback_mode
        if fallback_mode not in allowed_modes:
            raise ValueError("compiled fallback mode must be allowed")

        objective = request.optimization_objective or profile.optimization_objective
        balanced_weights = profile.balanced_weights
        if objective is OptimizationObjective.BALANCED_WEIGHTED_SCORE and balanced_weights is None:
            raise ValueError("balanced objective override requires an explicitly weighted profile")
        if objective is not OptimizationObjective.BALANCED_WEIGHTED_SCORE:
            balanced_weights = None

        low_threshold = (
            request.low_elevation_threshold_deg
            if request.low_elevation_threshold_deg is not None
            else profile.low_elevation_threshold_deg
        )
        low_rule = (
            LowElevationSafetyRule(
                below_elevation_deg=low_threshold,
                forced_mode=fallback_mode,
            )
            if low_threshold is not None
            else None
        )
        return MissionPolicy(
            policy_name=profile.name,
            version=profile.version,
            message_priority=request.message_priority,
            maximum_admissible_per=maximum_per,
            maximum_expected_delivery_latency_ms=maximum_latency_ms,
            optimization_objective=objective,
            allowed_radio_modes=allowed_modes,
            fallback_mode=fallback_mode,
            uncertainty_action=profile.uncertainty_action,
            low_elevation_safety=low_rule,
            balanced_weights=balanced_weights,
            source_request=request,
            policy_resolution=PolicyResolution(
                requested_mode=request.profile_selection_mode,
                resolved_profile=profile.name,
                resolution_reason=resolution_reason,
                source_fields=source_fields,
                warnings=resolution_warnings,
                profile_default_maximum_per=profile.maximum_admissible_per,
                requested_maximum_per_override=request.maximum_admissible_per,
                effective_maximum_per=maximum_per,
            ),
            assumptions=tuple(assumptions),
            metadata=(
                MetadataItem(key="policy_compiler", value="default-policy-engine-v1"),
                MetadataItem(key="profile_name", value=profile.name),
                MetadataItem(key="profile_version", value=profile.version.version),
                MetadataItem(
                    key="operational_energy_mode",
                    value=request.energy_mode.value if request.energy_mode else None,
                ),
                MetadataItem(key="generated_maximum_per", value=maximum_per),
                MetadataItem(
                    key="profile_selection_mode",
                    value=request.profile_selection_mode.value,
                ),
                MetadataItem(key="profile_resolution_reason", value=resolution_reason),
            ),
        )

    def _resolve_profile(
        self,
        request: OperationalRequest,
    ) -> tuple[PolicyProfile, str, tuple[str, ...], tuple[str, ...]]:
        automatic_name, automatic_reason, automatic_fields = self._automatic_profile(request)
        if request.profile_selection_mode is ProfileSelectionMode.MANUAL:
            if request.policy_profile is None:  # Defensive; the model already enforces this.
                raise ValueError("manual profile selection requires policy_profile")
            try:
                profile = self._profiles[request.policy_profile]
            except KeyError as error:
                raise ValueError(f"unknown policy profile: {request.policy_profile}") from error
            warnings: tuple[str, ...] = ()
            if profile.name != automatic_name:
                warnings = (
                    "manual profile overrides automatic resolution "
                    f"({automatic_name} from {automatic_reason})",
                )
            return profile, "manual_profile_selected", ("policy_profile",), warnings
        return self._profiles[automatic_name], automatic_reason, automatic_fields, ()

    @staticmethod
    def _automatic_profile(
        request: OperationalRequest,
    ) -> tuple[str, str, tuple[str, ...]]:
        if request.message_priority is MessagePriority.CRITICAL:
            return "critical_alert", "critical_priority", ("message_priority",)
        if request.energy_mode is EnergyMode.CONSERVE_ENERGY:
            return "battery_saving", "conserve_energy", ("energy_mode",)
        if request.message_priority in {MessagePriority.IMPORTANT, MessagePriority.URGENT}:
            return "reliable_telemetry", "elevated_priority", ("message_priority",)
        return "routine_telemetry", "routine_default", ("message_priority",)
