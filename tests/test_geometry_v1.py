from __future__ import annotations

import socket
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from meseleo_core import (
    GeometryComputationError,
    GroundStation,
    LinkState,
    RadioScenario,
    SatellitePassRequest,
    TLEInput,
    TLEStateRequest,
    V1GeometryAdapter,
)


def test_manual_state_mapping_preserves_independent_axes(
    radio_scenario: RadioScenario,
) -> None:
    state = LinkState(
        elevation_deg=22.0,
        radial_speed_km_s=-3.25,
        active_nodes=50_000,
        rssi_dbm=-120.0,
        snr_db=-4.0,
    )

    result = V1GeometryAdapter().map_manual_state(state, radio_scenario)

    assert result.link_state.elevation_deg == 22.0
    assert result.link_state.radial_speed_km_s == -3.25
    assert result.link_state.active_nodes == 50_000
    assert result.link_state.doppler_hz == pytest.approx(9_409.842105, rel=1e-6)
    assert result.visible


def test_doppler_uses_scenario_frequency() -> None:
    adapter = V1GeometryAdapter()
    state = LinkState(elevation_deg=30.0, radial_speed_km_s=2.0, active_nodes=10_000)
    low = RadioScenario(
        carrier_frequency_hz=400_000_000.0,
        payload_bytes=12,
        transmit_power_dbm=14.0,
    )
    high = low.model_copy(update={"carrier_frequency_hz": 800_000_000.0})

    low_doppler = adapter.map_manual_state(state, low).link_state.doppler_hz
    high_doppler = adapter.map_manual_state(state, high).link_state.doppler_hz

    assert low_doppler is not None
    assert high_doppler == pytest.approx(low_doppler * 2.0)


def test_fixed_tle_instantaneous_state_is_deterministic(
    fixed_tle: TLEInput,
    paris_station: GroundStation,
    radio_scenario: RadioScenario,
) -> None:
    request = TLEStateRequest(
        tle=fixed_tle,
        ground_station=paris_station,
        timestamp=datetime(2024, 1, 1, 12, tzinfo=UTC),
        active_nodes=50_000,
    )
    adapter = V1GeometryAdapter()

    first = adapter.calculate_tle_state(request, radio_scenario)
    second = adapter.calculate_tle_state(request, radio_scenario)

    assert first == second
    assert first.link_state.elevation_deg == pytest.approx(-14.5874620583)
    assert first.azimuth_deg == pytest.approx(61.3703913767)
    assert first.slant_range_km == pytest.approx(4_465.17523678)
    assert first.link_state.radial_speed_km_s == pytest.approx(6.45723843026)
    assert first.link_state.doppler_hz == pytest.approx(-18_695.877124)
    assert not first.visible


def test_fixed_full_pass_prediction_and_mask(
    fixed_tle: TLEInput,
    paris_station: GroundStation,
    radio_scenario: RadioScenario,
) -> None:
    adapter = V1GeometryAdapter()
    base = dict(
        tle=fixed_tle,
        ground_station=paris_station,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 3, tzinfo=UTC),
        active_nodes=10_000,
        sampling_interval_seconds=60.0,
    )
    five_degree = adapter.predict_passes(
        SatellitePassRequest(**base, minimum_elevation_deg=5.0),
        radio_scenario,
    )
    twenty_degree = adapter.predict_passes(
        SatellitePassRequest(**base, minimum_elevation_deg=20.0),
        radio_scenario,
    )

    assert len(five_degree) == 12
    assert five_degree[0].acquisition_of_signal == datetime(2024, 1, 1, 11, 47, tzinfo=UTC)
    assert five_degree[0].loss_of_signal == datetime(2024, 1, 1, 11, 52, tzinfo=UTC)
    assert five_degree[0].maximum_elevation_deg == pytest.approx(13.6141028693)
    assert all(state.elevation_deg >= 5.0 for item in five_degree for state in item.states)
    assert all(state.elevation_deg >= 20.0 for item in twenty_degree for state in item.states)
    assert len(twenty_degree) < len(five_degree)


def test_geometry_path_requires_no_network(
    monkeypatch: pytest.MonkeyPatch,
    fixed_tle: TLEInput,
    paris_station: GroundStation,
    radio_scenario: RadioScenario,
) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden in geometry tests")

    monkeypatch.setattr(socket, "create_connection", blocked)
    adapter = V1GeometryAdapter()
    result = adapter.calculate_tle_state(
        TLEStateRequest(
            tle=fixed_tle,
            ground_station=paris_station,
            timestamp=datetime(2024, 1, 1, 13, 25, tzinfo=UTC),
            active_nodes=10_000,
        ),
        radio_scenario,
    )

    assert result.visible


def test_invalid_tle_is_wrapped_as_geometry_error(
    paris_station: GroundStation,
    radio_scenario: RadioScenario,
) -> None:
    invalid = TLEInput(satellite_id="invalid", line1="1 broken", line2="2 broken")
    request = TLEStateRequest(
        tle=invalid,
        ground_station=paris_station,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        active_nodes=1,
    )

    with pytest.raises(GeometryComputationError):
        V1GeometryAdapter().calculate_tle_state(request, radio_scenario)


@pytest.mark.parametrize(
    ("field", "value"),
    (("latitude_deg", 90.01), ("longitude_deg", -180.01)),
)
def test_invalid_ground_station_is_rejected(field: str, value: float) -> None:
    values: dict[str, object] = {
        "ground_station_id": "invalid",
        "latitude_deg": 0.0,
        "longitude_deg": 0.0,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        GroundStation(**values)  # type: ignore[arg-type]
