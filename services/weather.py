"""Open-Meteo weather data client with GFS ensemble support."""
import asyncio
import math
import random
from datetime import date, datetime
from typing import Any

import httpx
from cachetools import TTLCache
from loguru import logger
from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception, retry_if_exception_type, stop_after_attempt, wait_exponential

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

_cache: TTLCache = TTLCache(maxsize=200, ttl=3600 * 6)
_cache_lock = asyncio.Lock()


class WeatherSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    station: str
    target_date: str
    members: int = 31
    mean_max_f: float | None = None
    mean_min_f: float | None = None
    p10_max_f: float | None = None
    p90_max_f: float | None = None
    precipitation_mm: float | None = None
    pct_above_threshold: float | None = None
    pct_below_threshold: float | None = None
    threshold: float | None = None
    hours_ahead: float | None = None
    # ECMWF blend fields (populated by get_forecast when ECMWF call succeeds)
    ecmwf_mean_max_f: float | None = None
    model_agreement_delta_f: float | None = None
    # Provenance for backtest transparency
    data_source: str = "gfs_forecast"  # "gfs_forecast" | "era5_with_noise" | "era5_fallback_raw"


def _is_retryable_weather(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 500, 502, 503):
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable_weather),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
async def _fetch(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    resp = await client.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


async def get_forecast(
    station: str,
    target_date: date,
    latitude: float,
    longitude: float,
    threshold: float | None = None,
    threshold_unit: str = "F",
    bias_f: float = 0.0,
) -> WeatherSummary | None:
    cache_key = f"fc:{station}:{target_date.isoformat()}"
    async with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    base_params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "mm",
        "timezone": "UTC",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }
    gfs_params = {**base_params, "models": "gfs_seamless"}
    ecmwf_params = {**base_params, "models": "ecmwf_ifs04"}

    async with httpx.AsyncClient() as client:
        gfs_task = asyncio.create_task(_fetch(client, FORECAST_URL, gfs_params))
        ecmwf_task = asyncio.create_task(_fetch(client, FORECAST_URL, ecmwf_params))
        gfs_data, ecmwf_data = await asyncio.gather(gfs_task, ecmwf_task, return_exceptions=True)

    if isinstance(gfs_data, Exception):
        logger.error(f"GFS forecast failed: {gfs_data}", station=station)
        return None

    summary = _build_summary(gfs_data, station, target_date, threshold, threshold_unit)

    # Blend ECMWF if available
    if not isinstance(ecmwf_data, Exception):
        ecmwf_summary = _build_summary(ecmwf_data, station, target_date, threshold, threshold_unit)
        if ecmwf_summary.mean_max_f is not None and summary.mean_max_f is not None:
            blended_max = round(summary.mean_max_f * 0.55 + ecmwf_summary.mean_max_f * 0.45, 1)
            delta = round(abs(summary.mean_max_f - ecmwf_summary.mean_max_f), 1)
            summary = summary.model_copy(update={
                "mean_max_f": blended_max,
                "ecmwf_mean_max_f": ecmwf_summary.mean_max_f,
                "model_agreement_delta_f": delta,
            })
    else:
        logger.warning(f"ECMWF forecast failed: {ecmwf_data}", station=station)

    if bias_f != 0.0:
        summary = _apply_bias(summary, bias_f)

    async with _cache_lock:
        _cache[cache_key] = summary
    return summary


async def get_historical_forecast(
    station: str,
    target_date: date,
    as_of_date: date,
    latitude: float,
    longitude: float,
    threshold: float | None = None,
    threshold_unit: str = "F",
    hours_ahead_override: float | None = None,
    bias_f: float = 0.0,
) -> WeatherSummary | None:
    """Return forecast/archive data for target_date (backtesting).

    Tries the Historical Forecast API first (returns what GFS predicted at as_of_date).
    Falls back to the ERA5 Archive API (actual observations) when that endpoint fails
    — which it will on the free Open-Meteo tier.
    """
    cache_key = f"hf:{station}:{target_date.isoformat()}:{as_of_date.isoformat()}"
    async with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    base_params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "mm",
        "timezone": "UTC",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }

    data: dict | None = None
    used_era5 = False
    async with httpx.AsyncClient() as client:
        # 1st attempt: Historical Forecast API (premium endpoint, archived GFS runs)
        try:
            params = {**base_params, "models": "gfs_seamless"}
            data = await _fetch(client, HISTORICAL_FORECAST_URL, params)
        except httpx.HTTPStatusError as e:
            logger.warning(
                "Historical forecast API error — falling back to archive",
                station=station,
                status=e.response.status_code,
                body=e.response.text[:300],
            )
        except Exception as e:
            logger.warning(
                "Historical forecast API unavailable — falling back to archive",
                station=station,
                error=repr(e),
            )

        # 2nd attempt: ERA5 Archive API (actual observations, always free)
        if data is None:
            try:
                data = await _fetch(client, HISTORICAL_WEATHER_URL, base_params)
                used_era5 = True
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Archive weather API error",
                    station=station,
                    status=e.response.status_code,
                    body=e.response.text[:300],
                )
                return None
            except Exception as e:
                logger.error("Archive weather API failed", station=station, error=repr(e))
                return None

    summary = _build_summary(data, station, target_date, threshold, threshold_unit, hours_ahead_override)

    if used_era5 and hours_ahead_override is not None:
        summary = _add_forecast_noise(summary, hours_ahead_override)
    elif used_era5:
        summary = summary.model_copy(update={"data_source": "era5_fallback_raw"})

    if bias_f != 0.0:
        summary = _apply_bias(summary, bias_f)

    async with _cache_lock:
        _cache[cache_key] = summary
    return summary


async def get_actual_outcome(
    station: str,
    target_date: date,
    latitude: float,
    longitude: float,
    threshold: float,
    condition: str,
    threshold_unit: str = "F",
) -> str | None:
    """Return YES/NO based on actual observed weather data (for resolution)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }

    async with httpx.AsyncClient() as client:
        try:
            data = await _fetch(client, HISTORICAL_WEATHER_URL, params)
        except Exception as e:
            logger.error("Historical weather failed", station=station, error=str(e))
            return None

    daily = data.get("daily", {})
    max_temps = daily.get("temperature_2m_max", [None])
    actual_max = max_temps[0] if max_temps else None

    if actual_max is None:
        return None

    if condition == "above":
        return "YES" if actual_max > threshold else "NO"
    else:
        return "YES" if actual_max < threshold else "NO"


def _add_forecast_noise(summary: WeatherSummary, hours_ahead: float) -> WeatherSummary:
    """Add realistic GFS forecast uncertainty to ERA5 actuals for honest backtesting."""
    if hours_ahead <= 12:
        sigma_f = 1.5
    elif hours_ahead <= 24:
        sigma_f = 2.5
    elif hours_ahead <= 48:
        sigma_f = 3.5
    else:
        sigma_f = 4.5

    noise = random.gauss(0, sigma_f)
    mean_max = (summary.mean_max_f or 0) + noise
    p10 = (summary.p10_max_f or 0) + noise
    p90 = (summary.p90_max_f or 0) + noise

    # Recompute pct_above using Normal CDF approximation
    if summary.threshold is not None and summary.mean_max_f is not None:
        spread = max(p90 - p10, 1.0) / 2.5  # approx sigma from P10/P90
        z = (mean_max - summary.threshold) / max(spread, 0.5)
        pct_above = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        pct_below = 1.0 - pct_above
    else:
        pct_above = summary.pct_above_threshold
        pct_below = summary.pct_below_threshold

    return summary.model_copy(update={
        "mean_max_f": round(mean_max, 1),
        "p10_max_f": round(p10, 1),
        "p90_max_f": round(p90, 1),
        "pct_above_threshold": round(pct_above, 3) if pct_above is not None else None,
        "pct_below_threshold": round(pct_below, 3) if pct_below is not None else None,
        "data_source": "era5_with_noise",
    })


def _apply_bias(summary: WeatherSummary, bias_f: float) -> WeatherSummary:
    """Subtract station bias (mean forecast - actual) from temperature fields."""
    if bias_f == 0.0:
        return summary
    mean_max = (summary.mean_max_f or 0) - bias_f
    p10 = (summary.p10_max_f or 0) - bias_f
    p90 = (summary.p90_max_f or 0) - bias_f

    pct_above = summary.pct_above_threshold
    pct_below = summary.pct_below_threshold
    if summary.threshold is not None:
        spread = max(p90 - p10, 1.0) / 2.5
        z = (mean_max - summary.threshold) / max(spread, 0.5)
        pct_above = round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)
        pct_below = round(1.0 - pct_above, 3)

    return summary.model_copy(update={
        "mean_max_f": round(mean_max, 1),
        "p10_max_f": round(p10, 1),
        "p90_max_f": round(p90, 1),
        "pct_above_threshold": pct_above,
        "pct_below_threshold": pct_below,
    })


def _build_summary(
    data: dict,
    station: str,
    target_date: date,
    threshold: float | None,
    threshold_unit: str,
    hours_ahead_override: float | None = None,
) -> WeatherSummary:
    daily = data.get("daily", {})
    max_temps: list[float | None] = daily.get("temperature_2m_max", [])
    min_temps: list[float | None] = daily.get("temperature_2m_min", [])
    precip: list[float | None] = daily.get("precipitation_sum", [])

    valid_max = [t for t in max_temps if t is not None]
    valid_min = [t for t in min_temps if t is not None]

    mean_max = sum(valid_max) / len(valid_max) if valid_max else None
    mean_min = sum(valid_min) / len(valid_min) if valid_min else None

    sorted_max = sorted(valid_max) if valid_max else []
    p10 = sorted_max[int(len(sorted_max) * 0.1)] if sorted_max else None
    p90 = sorted_max[int(len(sorted_max) * 0.9)] if sorted_max else None

    pct_above = pct_below = None
    if threshold is not None and valid_max:
        pct_above = sum(1 for t in valid_max if t > threshold) / len(valid_max)
        pct_below = 1.0 - pct_above

    if hours_ahead_override is not None:
        hours_ahead = hours_ahead_override
    else:
        now_utc = datetime.utcnow()
        target_dt = datetime(target_date.year, target_date.month, target_date.day)
        hours_ahead = max(0.0, (target_dt - now_utc).total_seconds() / 3600)

    return WeatherSummary(
        station=station,
        target_date=target_date.isoformat(),
        members=len(valid_max) or 1,
        mean_max_f=round(mean_max, 1) if mean_max is not None else None,
        mean_min_f=round(mean_min, 1) if mean_min is not None else None,
        p10_max_f=round(p10, 1) if p10 is not None else None,
        p90_max_f=round(p90, 1) if p90 is not None else None,
        precipitation_mm=round(precip[0], 1) if precip and precip[0] is not None else None,
        pct_above_threshold=round(pct_above, 3) if pct_above is not None else None,
        pct_below_threshold=round(pct_below, 3) if pct_below is not None else None,
        threshold=threshold,
        hours_ahead=round(hours_ahead, 1) if hours_ahead is not None else None,
    )
