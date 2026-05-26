"""Tests for weather service with mocked HTTP."""
import math
import pytest
import respx
import httpx
from datetime import date

from services.weather import (
    get_forecast,
    _build_summary,
    _add_forecast_noise,
    _apply_bias,
    WeatherSummary,
)


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


def test_build_summary_single_member_p10_p90_equal_mean():
    # ERA5-style: one observation — percentile indices both collapse to index 0
    data = {
        "daily": {
            "time": ["2026-06-05"],
            "temperature_2m_max": [97.0],
            "temperature_2m_min": [74.0],
            "precipitation_sum": [0.0],
        }
    }
    summary = _build_summary(data, "KDAL", date(2026, 6, 5), threshold=95.0, threshold_unit="F")

    assert summary.p10_max_f == summary.p90_max_f == summary.mean_max_f
    # Single value above threshold → pct_above must be exactly 1.0
    assert summary.pct_above_threshold == 1.0
    assert summary.pct_below_threshold == 0.0


def test_build_summary_single_member_below_threshold_pct():
    data = {
        "daily": {
            "time": ["2026-06-05"],
            "temperature_2m_max": [90.0],
            "temperature_2m_min": [70.0],
            "precipitation_sum": [0.0],
        }
    }
    summary = _build_summary(data, "KDAL", date(2026, 6, 5), threshold=95.0, threshold_unit="F")

    # Single value below threshold → pct_above must be exactly 0.0
    assert summary.pct_above_threshold == 0.0
    assert summary.pct_below_threshold == 1.0


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


# ---------------------------------------------------------------------------
# _add_forecast_noise
# ---------------------------------------------------------------------------

def _make_era5_summary(mean_max_f: float, threshold: float | None = None) -> WeatherSummary:
    return WeatherSummary(
        station="KDAL",
        target_date="2026-06-05",
        members=1,
        mean_max_f=mean_max_f,
        mean_min_f=70.0,
        p10_max_f=mean_max_f,
        p90_max_f=mean_max_f,
        precipitation_mm=0.0,
        pct_above_threshold=(1.0 if (threshold is not None and mean_max_f > threshold) else
                              (0.0 if threshold is not None else None)),
        pct_below_threshold=(0.0 if (threshold is not None and mean_max_f > threshold) else
                              (1.0 if threshold is not None else None)),
        threshold=threshold,
        hours_ahead=48.0,
        data_source="era5_fallback_raw",
    )


@pytest.mark.parametrize("hours_ahead,expected_sigma", [
    (6.0, 1.5),
    (18.0, 2.5),
    (36.0, 3.5),
    (72.0, 4.5),
])
def test_add_forecast_noise_p90_gt_p10(hours_ahead, expected_sigma):
    summary = _make_era5_summary(mean_max_f=95.0, threshold=95.0)
    noisy = _add_forecast_noise(summary, hours_ahead)

    assert noisy.p90_max_f is not None
    assert noisy.p10_max_f is not None
    assert noisy.p90_max_f > noisy.p10_max_f
    # Spread must equal 2 * 1.28 * sigma_f (within rounding)
    expected_spread = round(2 * 1.28 * expected_sigma, 1)
    actual_spread = round(noisy.p90_max_f - noisy.p10_max_f, 1)
    assert abs(actual_spread - expected_spread) < 0.2


def test_add_forecast_noise_data_source():
    summary = _make_era5_summary(mean_max_f=95.0, threshold=95.0)
    noisy = _add_forecast_noise(summary, hours_ahead=48.0)
    assert noisy.data_source == "era5_with_noise"


def test_add_forecast_noise_pct_above_strictly_between_0_and_1_near_threshold(monkeypatch):
    # With noise=0, mean stays 2°F below threshold; CDF must give an interior value.
    monkeypatch.setattr("services.weather.random.gauss", lambda *a, **k: 0.0)
    threshold = 95.0
    mean_near = 93.0  # z = (93 - 95) / 3.5 ≈ -0.57 → pct_above ≈ 0.28
    summary = _make_era5_summary(mean_max_f=mean_near, threshold=threshold)

    noisy = _add_forecast_noise(summary, hours_ahead=48.0)
    assert noisy.pct_above_threshold is not None
    assert 0.0 < noisy.pct_above_threshold < 1.0
    assert noisy.pct_below_threshold is not None
    assert 0.0 < noisy.pct_below_threshold < 1.0


def test_add_forecast_noise_no_threshold_pct_unchanged():
    summary = _make_era5_summary(mean_max_f=95.0, threshold=None)
    noisy = _add_forecast_noise(summary, hours_ahead=48.0)

    assert noisy.pct_above_threshold is None
    assert noisy.pct_below_threshold is None


# ---------------------------------------------------------------------------
# _apply_bias
# ---------------------------------------------------------------------------

def _make_noisy_summary(mean_max_f: float, threshold: float | None, sigma_f: float = 3.5) -> WeatherSummary:
    p10 = round(mean_max_f - 1.28 * sigma_f, 1)
    p90 = round(mean_max_f + 1.28 * sigma_f, 1)
    if threshold is not None:
        z = (mean_max_f - threshold) / sigma_f
        pct_above = round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)
        pct_below = round(1.0 - pct_above, 3)
    else:
        pct_above = pct_below = None
    return WeatherSummary(
        station="KDAL",
        target_date="2026-06-05",
        members=1,
        mean_max_f=mean_max_f,
        mean_min_f=70.0,
        p10_max_f=p10,
        p90_max_f=p90,
        precipitation_mm=0.0,
        pct_above_threshold=pct_above,
        pct_below_threshold=pct_below,
        threshold=threshold,
        hours_ahead=48.0,
        data_source="era5_with_noise",
    )


def test_apply_bias_reduces_mean_max():
    summary = _make_noisy_summary(mean_max_f=97.0, threshold=95.0)
    biased = _apply_bias(summary, bias_f=2.0)

    assert biased.mean_max_f == 95.0
    assert biased.mean_max_f < summary.mean_max_f


def test_apply_bias_pct_above_decreases_when_bias_pushes_mean_below_threshold():
    # mean_max starts above threshold; after bias it drops below → pct_above must fall
    threshold = 95.0
    summary = _make_noisy_summary(mean_max_f=97.0, threshold=threshold)
    original_pct = summary.pct_above_threshold

    biased = _apply_bias(summary, bias_f=4.0)  # 97 - 4 = 93, now below threshold

    assert biased.pct_above_threshold < original_pct


def test_apply_bias_zero_spread_does_not_crash():
    # p10 == p90 → the guard `max(sigma, 0.5)` must prevent division by zero
    summary = WeatherSummary(
        station="KDAL",
        target_date="2026-06-05",
        members=1,
        mean_max_f=97.0,
        mean_min_f=70.0,
        p10_max_f=97.0,
        p90_max_f=97.0,
        precipitation_mm=0.0,
        pct_above_threshold=1.0,
        pct_below_threshold=0.0,
        threshold=95.0,
        hours_ahead=48.0,
        data_source="era5_fallback_raw",
    )

    # Must not raise even though p10 == p90 (zero spread)
    biased = _apply_bias(summary, bias_f=1.0)
    assert biased.mean_max_f == 96.0
    assert biased.pct_above_threshold is not None
