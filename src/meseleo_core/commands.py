"""Validated V1 AT-command translation and future execution boundary."""

from __future__ import annotations

from typing import Protocol

from .configuration import V1CommandConfiguration
from .integration_errors import CommandMappingUnavailableError, TransmissionIneligibleError
from .models import DecisionResult, MetadataItem, ModemCommand, TransmissionEligibility


class V1ATCommandAdapter:
    """Translate an exact decision; never select or optimize a radio mode."""

    def __init__(self, configuration: V1CommandConfiguration | None = None) -> None:
        self.configuration = configuration or V1CommandConfiguration()
        self._mapping = {item.radio_mode: item.command for item in self.configuration.mappings}

    def translate(
        self,
        decision: DecisionResult,
        transmission_eligibility: TransmissionEligibility,
    ) -> ModemCommand:
        if not transmission_eligibility.eligible:
            raise TransmissionIneligibleError(
                "validated transmission eligibility blocks command generation"
            )
        try:
            command = self._mapping[decision.selected_mode]
        except KeyError as error:
            raise CommandMappingUnavailableError(
                f"no AT command configured for {decision.selected_mode.value}"
            ) from error
        return ModemCommand(
            command=command,
            radio_mode=decision.selected_mode,
            adapter_version=self.configuration.adapter_version,
            modem_profile=self.configuration.modem_profile,
            metadata=(
                MetadataItem(
                    key="decision_engine_version",
                    value=decision.deterministic_decision_engine_version.version,
                ),
                MetadataItem(key="policy_version", value=decision.policy_version.version),
                MetadataItem(
                    key="transmission_eligibility",
                    value=transmission_eligibility.status.value,
                ),
            ),
        )


class ModemExecutor(Protocol):
    """Future hardware boundary; Sprint 2 provides no implementation."""

    def execute(self, command: ModemCommand) -> None: ...
