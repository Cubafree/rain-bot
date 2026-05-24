"""Tests for weather service with mocked HTTP."""
import pytest
import respx
import httpx
from datetime import date

from services.weather import get_forecast, _build_summary, WeatherSummary


MOCK_RESPONSE = {
    "latitude": 32.847,
    "longitude": -96.852,
    "daily": {
        "time": ["2026-06-05"],
        "temperature_2m_max": [96.2, 94.8, 98.1, 95.5, 97.0],
        "temperature_2m_min": [75.0, 74.5, 76.2, 74.8, 75.5],
        "precipitation_sum": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
}


@pytest.mark.asyncio
@respx.mock
async def test_get_forecast_returns_summary():
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=MOCK_RESPONSE)
    )

    result = await get_forecast(
        station="KDAL",
        target_date=date(2026, 6, 5),
        latitude=32.847,
        longitude=-96.852,
        threshold=95.0,
        threshold_unit="F",
    )

    assert result is not None
    assert result.station == "KDAL"
    assert result.mean_max_f is not None
    assert result.pct_above_threshold is not None


@pytest.mark.asyncio
@respx.mock
async def test_get_forecast_handles_api_error():
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(500)
    )

    result = await get_forecast(
        station="KDAL",
        target_date=date(2026, 6, 5),
        latitude=32.847,
        longitude=-96.852,
    )

    assert result is None


def test_build_summary_pct_above():
    data = {
        "daily": {
            "time": ["2026-06-05"],
            "temperature_2m_max": [96.0, 94.0, 97.0, 93.0, 98.0],
            "temperature_2m_min": [75.0, 74.0, 76.0, 73.0, 77.0],
            "precipitation_sum": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }
    summary = _build_summary(data, "KDAL", date(2026, 6, 5), threshold=95.0, threshold_unit="F")

    assert summary.pct_above_threshold == 0.6
    assert summary.pct_below_threshold == 0.4
    assert summary.threshold == 95.0


def test_build_summary_no_threshold():
    data = {
        "daily": {
            "time": ["2026-06-05"],
            "temperature_2m_max": [90.0],
            "temperature_2m_min": [70.0],
            "precipitation_sum": [0.0],
        }
    }
    summary = _build_summary(data, "KDAL", date(2026, 6, 5), threshold=None, threshold_unit="F")

    assert summary.pct_above_threshold is None
    assert summary.mean_max_f == 90.0
