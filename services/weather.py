"""Open-Meteo weather data client with GFS ensemble support."""
import asyncio
from datetime import date, datetime
from typing import Any

import httpx
from cachetools import TTLCache
from loguru import logger
from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def _fetch(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    resp = await client.get(url, params=params, timeout=25)
    resp.raise_for_status()
    return resp.json()


async def get_forecast(
    station: str,
    target_date: date,
    latitude: float,
    longitude: float,
    threshold: float | None = None,
    threshold_unit: str = "F",
) -> WeatherSummary | None:
    cache_key = f"fc:{station}:{target_date.isoformat()}"
    async with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "mm",
        "timezone": "UTC",
        "models": "gfs_seamless",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "forecast_days": 16,
    }

    async with httpx.AsyncClient() as client:
        try:
            data = await _fetch(client, FORECAST_URL, params)
        except Exception as e:
            logger.error("Weather forecast failed", station=station, error=str(e))
            return None

    summary = _build_summary(data, station, target_date, threshold, threshold_unit)
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
) -> WeatherSummary | None:
    """Return what GFS forecast said on as_of_date for target_date (backtesting)."""
    cache_key = f"hf:{station}:{target_date.isoformat()}:{as_of_date.isoformat()}"
    async with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "mm",
        "timezone": "UTC",
        "models": "gfs_seamless",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }

    async with httpx.AsyncClient() as client:
        try:
            data = await _fetch(client, HISTORICAL_FORECAST_URL, params)
        except Exception as e:
            logger.error("Historical forecast failed", station=station, error=str(e))
            return None

    summary = _build_summary(data, station, target_date, threshold, threshold_unit)
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


def _build_summary(
    data: dict,
    station: str,
    target_date: date,
    threshold: float | None,
    threshold_unit: str,
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

    hours_ahead = None
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
