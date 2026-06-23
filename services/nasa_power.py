"""NASA POWER actuals client — independent ground-truth temperature source.

NASA POWER (power.larc.nasa.gov) serves daily observed/reanalysis temperature
(MERRA-2 / GEOS) with no API key. It is NOT a forecast model — values describe
what already happened, with a latency of days to months — so it is used here as
an independent *actuals* source for:
  - station-bias calibration (compare the bot's own past forecasts to truth)
  - cross-checking ERA5-based resolution

All temperatures are returned in Fahrenheit to match the rest of the codebase.
"""
from datetime import date

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
_FILL_VALUE = -999.0  # NASA POWER sentinel for missing data


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=2, max=30),
    reraise=True,
)
async def _fetch(client: httpx.AsyncClient, params: dict) -> dict:
    resp = await client.get(POWER_DAILY_URL, params=params, timeout=40)
    resp.raise_for_status()
    return resp.json()


async def get_actual_max_range_f(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
) -> dict[date, float]:
    """Return {date: observed daily max °F} for [start, end] from NASA POWER.

    Missing days (fill value -999) are omitted.
    """
    params = {
        "parameters": "T2M_MAX",
        "community": "AG",
        "latitude": latitude,
        "longitude": longitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    try:
        async with httpx.AsyncClient() as client:
            data = await _fetch(client, params)
    except Exception as e:
        logger.warning("NASA POWER fetch failed", lat=latitude, lon=longitude, error=repr(e))
        return {}

    series = data.get("properties", {}).get("parameter", {}).get("T2M_MAX", {})
    out: dict[date, float] = {}
    for ymd, val in series.items():
        if val is None or val <= _FILL_VALUE:
            continue
        try:
            d = date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
        except (ValueError, IndexError):
            continue
        out[d] = round(_c_to_f(float(val)), 1)
    return out


async def get_actual_max_f(latitude: float, longitude: float, target_date: date) -> float | None:
    """Return the observed daily max °F for a single date, or None if unavailable."""
    series = await get_actual_max_range_f(latitude, longitude, target_date, target_date)
    return series.get(target_date)
