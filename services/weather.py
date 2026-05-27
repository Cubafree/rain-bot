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
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
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

    # GFS: 31-member ensemble (control + 30 perturbed) via dedicated ensemble endpoint
    ensemble_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "mm",
        "timezone": "UTC",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "models": "gfs025",
    }
    # ECMWF deterministic — daily, kept for model agreement delta only
    ecmwf_params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "mm",
        "timezone": "UTC",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "models": "ecmwf_ifs04",
    }

    async with httpx.AsyncClient() as client:
        gfs_task = asyncio.create_task(_fetch(client, ENSEMBLE_URL, ensemble_params))
        ecmwf_task = asyncio.create_task(_fetch(client, FORECAST_URL, ecmwf_params))
        gfs_data, ecmwf_data = await asyncio.gather(gfs_task, ecmwf_task, return_exceptions=True)

    if isinstance(gfs_data, Exception):
        logger.error(f"GFS ensemble forecast failed: {gfs_data}", station=station)
        return None

    summary = _build_summary_from_ensemble(gfs_data, station, target_date, threshold, threshold_unit)

    # ECMWF model agreement delta (blend GFS ensemble mean 55% / ECMWF det. 45%)
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

    # API returns Fahrenheit; convert Celsius threshold to match.
    threshold_f = threshold * 9 / 5 + 32 if threshold_unit == "C" else threshold

    if condition == "above":
        return "YES" if actual_max > threshold_f else "NO"
    else:
        return "YES" if actual_max < threshold_f else "NO"


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
    # P10/P90 represent the ±1.28σ bounds of the synthetic ensemble (80% range)
    p10 = round(mean_max - 1.28 * sigma_f, 1)
    p90 = round(mean_max + 1.28 * sigma_f, 1)

    # Recompute pct_above using Normal CDF with sigma_f as the spread.
    # ERA5 is a single observation so p10==p90 before noise — deriving spread
    # from p90-p10 gives 0, collapsing every z-score to ±∞ and pct_above to
    # 0% or 100%.  Using sigma_f directly gives realistic intermediate values.
    if summary.threshold is not None:
        z = (mean_max - summary.threshold) / sigma_f
        pct_above = round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)
        pct_below = round(1.0 - pct_above, 3)
    else:
        pct_above = summary.pct_above_threshold
        pct_below = summary.pct_below_threshold

    return summary.model_copy(update={
        "mean_max_f": round(mean_max, 1),
        "p10_max_f": p10,
        "p90_max_f": p90,
        "pct_above_threshold": pct_above,
        "pct_below_threshold": pct_below,
        "data_source": "era5_with_noise",
    })


def _apply_bias(summary: WeatherSummary, bias_f: float) -> WeatherSummary:
    """Subtract station bias (mean forecast - actual) from temperature fields."""
    if bias_f == 0.0:
        return summary
    if summary.mean_max_f is None:
        return summary
    mean_max = summary.mean_max_f - bias_f
    p10 = (summary.p10_max_f - bias_f) if summary.p10_max_f is not None else None
    p90 = (summary.p90_max_f - bias_f) if summary.p90_max_f is not None else None

    pct_above = summary.pct_above_threshold
    pct_below = summary.pct_below_threshold
    if summary.threshold is not None and p10 is not None and p90 is not None and p90 > p10:
        # Derive sigma from P10/P90 spread (valid after _add_forecast_noise set them)
        sigma = (p90 - p10) / (2 * 1.28)
        z = (mean_max - summary.threshold) / max(sigma, 0.5)
        pct_above = round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)
        pct_below = round(1.0 - pct_above, 3)

    return summary.model_copy(update={
        "mean_max_f": round(mean_max, 1),
        "p10_max_f": round(p10, 1) if p10 is not None else None,
        "p90_max_f": round(p90, 1) if p90 is not None else None,
        "pct_above_threshold": pct_above,
        "pct_below_threshold": pct_below,
    })


def _extract_member_daily_extremes(
    data: dict,
    target_date: date,
) -> tuple[list[float], list[float], float | None]:
    """Return (member_maxes, member_mins, precip_mm) from GFS025 ensemble hourly response.

    All temperature columns (``temperature_2m`` control + ``temperature_2m_memberXX``)
    are aggregated to a daily max and min.  Precipitation is summed for the control run.
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    date_str = target_date.isoformat()
    target_idx = [i for i, t in enumerate(times) if t.startswith(date_str)]

    if not target_idx:
        return [], [], None

    # Collect control + perturbed member temperature columns
    temp_keys = sorted(
        k for k in hourly
        if k == "temperature_2m"
        or (k.startswith("temperature_2m_member") and k[len("temperature_2m_member"):].isdigit())
    )

    member_maxes: list[float] = []
    member_mins: list[float] = []
    for key in temp_keys:
        vals = hourly[key]
        day_vals = [vals[i] for i in target_idx if i < len(vals) and vals[i] is not None]
        if day_vals:
            member_maxes.append(max(day_vals))
            member_mins.append(min(day_vals))

    # Precipitation — sum control-run hourly values for the day
    precip_vals = hourly.get("precipitation", [])
    precip_day = [precip_vals[i] for i in target_idx if i < len(precip_vals) and precip_vals[i] is not None]
    precip_total = round(sum(precip_day), 1) if precip_day else None

    return member_maxes, member_mins, precip_total


def _build_summary_from_ensemble(
    data: dict,
    station: str,
    target_date: date,
    threshold: float | None,
    threshold_unit: str,
    hours_ahead_override: float | None = None,
) -> WeatherSummary:
    """Build WeatherSummary from GFS025 ensemble hourly API response.

    Computes genuine per-member daily max temperatures so that
    ``pct_above_threshold``, ``p10_max_f``, and ``p90_max_f`` reflect the
    real ensemble spread instead of the degenerate 0 %/100 % values produced
    by a single deterministic run.
    """
    member_maxes, member_mins, precip_mm = _extract_member_daily_extremes(data, target_date)

    if not member_maxes:
        logger.warning("GFS ensemble returned no temperature data", station=station, date=target_date)
        return WeatherSummary(station=station, target_date=target_date.isoformat(), members=0)

    n = len(member_maxes)
    sorted_maxes = sorted(member_maxes)
    mean_max = sum(member_maxes) / n
    mean_min = sum(member_mins) / len(member_mins) if member_mins else None
    p10 = sorted_maxes[max(0, int(n * 0.1))]
    p90 = sorted_maxes[min(n - 1, int(n * 0.9))]

    threshold_f = threshold * 9 / 5 + 32 if (threshold is not None and threshold_unit == "C") else threshold

    pct_above = pct_below = None
    if threshold_f is not None:
        pct_above = round(sum(1 for t in member_maxes if t > threshold_f) / n, 3)
        pct_below = round(1.0 - pct_above, 3)

    if hours_ahead_override is not None:
        hours_ahead = hours_ahead_override
    else:
        from datetime import timezone as _tz
        now_utc = datetime.now(_tz.utc)
        target_dt = datetime(target_date.year, target_date.month, target_date.day, tzinfo=_tz.utc)
        hours_ahead = max(0.0, (target_dt - now_utc).total_seconds() / 3600)

    return WeatherSummary(
        station=station,
        target_date=target_date.isoformat(),
        members=n,
        mean_max_f=round(mean_max, 1),
        mean_min_f=round(mean_min, 1) if mean_min is not None else None,
        p10_max_f=round(p10, 1),
        p90_max_f=round(p90, 1),
        precipitation_mm=precip_mm,
        pct_above_threshold=pct_above,
        pct_below_threshold=pct_below,
        threshold=round(threshold_f, 1) if threshold_f is not None else None,
        hours_ahead=round(hours_ahead, 1),
        data_source="gfs_forecast",
    )


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

    # API always returns Fahrenheit; convert Celsius threshold to match.
    threshold_f = threshold * 9 / 5 + 32 if (threshold is not None and threshold_unit == "C") else threshold

    pct_above = pct_below = None
    if threshold_f is not None and valid_max:
        pct_above = sum(1 for t in valid_max if t > threshold_f) / len(valid_max)
        pct_below = 1.0 - pct_above

    if hours_ahead_override is not None:
        hours_ahead = hours_ahead_override
    else:
        from datetime import timezone as _tz
        now_utc = datetime.now(_tz.utc)
        target_dt = datetime(target_date.year, target_date.month, target_date.day, tzinfo=_tz.utc)
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
        threshold=round(threshold_f, 1) if threshold_f is not None else None,
        hours_ahead=round(hours_ahead, 1) if hours_ahead is not None else None,
    )
