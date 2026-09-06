"""Offline Skyfield adapter derived from the academic V1 dashboard geometry."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from .configuration import V1_GIT_COMMIT, V1GeometryConfiguration
from .integration_errors import GeometryComputationError, PredictionEngineUnavailableError
from .models import (
    GeometryResult,
    LinkAvailability,
    LinkAvailabilityStatus,
    LinkState,
    MetadataItem,
    PassState,
    RadioScenario,
    SatellitePass,
    SatellitePassRequest,
    TLEInput,
    TLEStateRequest,
)


class V1GeometryAdapter:
    """Adapt V1 Skyfield geometry without network reads or implicit current time."""

    def __init__(self, configuration: V1GeometryConfiguration | None = None) -> None:
        self.configuration = configuration or V1GeometryConfiguration()
        try:
            from skyfield.api import load  # type: ignore[import-untyped]

            self._timescale = load.timescale(builtin=True)
        except Exception as error:
            raise PredictionEngineUnavailableError(
                "Skyfield is required for the V1 geometry adapter"
            ) from error

    def map_manual_state(
        self,
        state: LinkState,
        scenario: RadioScenario,
        minimum_elevation_deg: float = 0.0,
    ) -> GeometryResult:
        """Validate a manual state and derive Doppler only when it is absent."""

        doppler_hz = state.doppler_hz
        if doppler_hz is None:
            doppler_hz = self._doppler_hz(state.radial_speed_km_s, scenario)
            state = state.model_copy(update={"doppler_hz": doppler_hz})
        availability = self._link_availability(
            state.elevation_deg,
            minimum_elevation_deg,
            "manual",
            state.link_availability_status,
        )
        return GeometryResult(
            link_state=state,
            visible=availability.status is LinkAvailabilityStatus.VISIBLE,
            link_availability=availability,
            metadata=(
                MetadataItem(key="geometry_source", value="manual"),
                MetadataItem(key="doppler_frequency_hz", value=scenario.carrier_frequency_hz),
            ),
        )

    def calculate_tle_state(
        self,
        request: TLEStateRequest,
        scenario: RadioScenario,
    ) -> GeometryResult:
        try:
            satellite = self._make_satellite(request.tle)
            state = self._observe(
                satellite,
                request,
                scenario,
                request.minimum_elevation_deg,
            )
        except GeometryComputationError:
            raise
        except Exception as error:
            raise GeometryComputationError("failed to calculate TLE link geometry") from error
        link_state = LinkState(
            elevation_deg=state.elevation_deg,
            radial_speed_km_s=state.radial_speed_km_s,
            active_nodes=request.active_nodes,
            timestamp=request.timestamp,
            satellite_id=request.tle.satellite_id,
            ground_station_id=request.ground_station.ground_station_id,
            doppler_hz=state.doppler_hz,
            metadata=(
                MetadataItem(key="geometry_adapter", value="v1-skyfield"),
                MetadataItem(key="source_git_commit", value=V1_GIT_COMMIT),
            ),
        )
        return GeometryResult(
            link_state=link_state,
            azimuth_deg=state.azimuth_deg,
            slant_range_km=state.slant_range_km,
            visible=state.visible,
            link_availability=self._link_availability(
                state.elevation_deg,
                request.minimum_elevation_deg,
                "tle",
            ),
            metadata=(
                MetadataItem(key="tle_source", value=request.tle.source),
                MetadataItem(key="minimum_elevation_deg", value=request.minimum_elevation_deg),
                MetadataItem(key="doppler_frequency_hz", value=scenario.carrier_frequency_hz),
            ),
        )

    def predict_passes(
        self,
        request: SatellitePassRequest,
        scenario: RadioScenario,
    ) -> tuple[SatellitePass, ...]:
        """Sample and segment all visible passes in the requested interval."""

        try:
            satellite = self._make_satellite(request.tle)
            samples = tuple(
                self._observe_at(
                    satellite,
                    request.ground_station.latitude_deg,
                    request.ground_station.longitude_deg,
                    request.ground_station.altitude_m,
                    timestamp,
                    scenario,
                    request.minimum_elevation_deg,
                )
                for timestamp in self._sample_times(request)
            )
        except GeometryComputationError:
            raise
        except Exception as error:
            raise GeometryComputationError("failed to calculate satellite passes") from error

        segments: list[tuple[PassState, ...]] = []
        current: list[PassState] = []
        for sample in samples:
            if sample.visible:
                current.append(sample)
            elif current:
                if len(current) >= 2:
                    segments.append(tuple(current))
                current = []
        if len(current) >= 2:
            segments.append(tuple(current))

        passes: list[SatellitePass] = []
        for index, states in enumerate(segments):
            culmination = max(states, key=lambda item: item.elevation_deg)
            acquisition = states[0].timestamp
            loss = states[-1].timestamp
            passes.append(
                SatellitePass(
                    satellite_id=request.tle.satellite_id,
                    ground_station_id=request.ground_station.ground_station_id,
                    acquisition_of_signal=acquisition,
                    loss_of_signal=loss,
                    culmination=culmination.timestamp,
                    maximum_elevation_deg=culmination.elevation_deg,
                    duration_seconds=(loss - acquisition).total_seconds(),
                    states=states,
                    metadata=(
                        MetadataItem(key="pass_index", value=index),
                        MetadataItem(
                            key="sampling_interval_seconds",
                            value=request.sampling_interval_seconds,
                        ),
                        MetadataItem(
                            key="minimum_elevation_deg",
                            value=request.minimum_elevation_deg,
                        ),
                        MetadataItem(key="source_git_commit", value=V1_GIT_COMMIT),
                    ),
                )
            )
        return tuple(passes)

    @staticmethod
    def _link_availability(
        elevation_deg: float,
        minimum_elevation_deg: float,
        source: str,
        explicit_status: LinkAvailabilityStatus | None = None,
    ) -> LinkAvailability:
        if explicit_status is not None:
            status = explicit_status
        elif elevation_deg < 0.0:
            status = LinkAvailabilityStatus.BELOW_HORIZON
        elif elevation_deg < minimum_elevation_deg:
            status = LinkAvailabilityStatus.BELOW_ELEVATION_MASK
        else:
            status = LinkAvailabilityStatus.VISIBLE
        return LinkAvailability(
            status=status,
            elevation_deg=elevation_deg,
            minimum_elevation_deg=minimum_elevation_deg,
            source=source,
        )

    def _make_satellite(self, tle: TLEInput) -> Any:
        try:
            from skyfield.api import EarthSatellite

            return EarthSatellite(tle.line1, tle.line2, tle.satellite_id, self._timescale)
        except Exception as error:
            raise GeometryComputationError(
                f"invalid TLE for satellite {tle.satellite_id!r}"
            ) from error

    def _observe(
        self,
        satellite: Any,
        request: TLEStateRequest,
        scenario: RadioScenario,
        minimum_elevation_deg: float,
    ) -> PassState:
        return self._observe_at(
            satellite,
            request.ground_station.latitude_deg,
            request.ground_station.longitude_deg,
            request.ground_station.altitude_m,
            request.timestamp,
            scenario,
            minimum_elevation_deg,
        )

    def _observe_at(
        self,
        satellite: Any,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
        timestamp: Any,
        scenario: RadioScenario,
        minimum_elevation_deg: float,
    ) -> PassState:
        try:
            import numpy as np
            from skyfield.api import wgs84

            station = wgs84.latlon(
                latitude_degrees=latitude_deg,
                longitude_degrees=longitude_deg,
                elevation_m=altitude_m,
            )
            time = self._timescale.from_datetime(timestamp)
            topocentric = (satellite - station).at(time)
            altitude, azimuth, distance = topocentric.altaz()
            position = topocentric.position.km
            velocity = topocentric.velocity.km_per_s
            slant_range_km = float(distance.km)
            radial_speed_km_s = float(np.dot(position, velocity) / slant_range_km)
            elevation_deg = float(altitude.degrees)
            return PassState(
                timestamp=timestamp,
                elevation_deg=elevation_deg,
                azimuth_deg=float(azimuth.degrees) % 360.0,
                slant_range_km=slant_range_km,
                radial_speed_km_s=radial_speed_km_s,
                doppler_hz=self._doppler_hz(radial_speed_km_s, scenario),
                visible=elevation_deg >= max(0.0, minimum_elevation_deg),
            )
        except Exception as error:
            raise GeometryComputationError(
                f"geometry computation failed at {timestamp.isoformat()}"
            ) from error

    def _doppler_hz(self, radial_speed_km_s: float, scenario: RadioScenario) -> float:
        radial_speed_m_s = radial_speed_km_s * 1000.0
        return (
            -(radial_speed_m_s / self.configuration.speed_of_light_m_s)
            * scenario.carrier_frequency_hz
        )

    @staticmethod
    def _sample_times(request: SatellitePassRequest) -> tuple[Any, ...]:
        interval = request.sampling_interval_seconds
        duration = (request.end - request.start).total_seconds()
        count = int(duration // interval)
        times = [request.start + timedelta(seconds=index * interval) for index in range(count + 1)]
        if times[-1] < request.end:
            times.append(request.end)
        return tuple(times)
