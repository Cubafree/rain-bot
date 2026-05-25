"""Compute station forecast bias from Visual Crossing Historical Forecast API.

One-time calibration script. Requires VC_API_KEY env var (free tier: 1000 records/day).

Usage:
    python scripts/compute_bias.py --stations KLGA KJFK KORD
    python scripts/compute_bias.py --stations all --months 1-12
    python scripts/compute_bias.py --stations KLGA --months 6 7 8
"""
import argparse
import asyncio
import os
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.models import StationBias
from db.session import db_session
from services.event_router import CITY_STATIONS, STATION_COORDS

VC_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
# Free tier: forecast re-analysis going back up to ~5 years
LOOKBACK_YEARS = 2


def _all_stations() -> list[str]:
    """Return unique station codes from the event router."""
    return sorted(set(CITY_STATIONS.values()))


def _parse_months(months_arg: list[str]) -> list[int]:
    """Accept '1-12' range or individual month numbers."""
    result: list[int] = []
    for token in months_arg:
        if "-" in token:
            start, end = token.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(token))
    return sorted(set(result))


async def _fetch_vc_month(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    year: int,
    month: int,
    api_key: str,
) -> list[tuple[float, float]]:
    """Fetch (forecast_tempmax, actual_tempmax) pairs for one station/month/year.

    Visual Crossing Historical Forecast: forecastBasisDay=-1 means the forecast
    was issued the day before the observation date.
    """
    start = date(year, month, 1)
    # last day of month
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)

    url = f"{VC_BASE}/{lat},{lon}/{start.isoformat()}/{end.isoformat()}"
    params = {
        "unitGroup": "us",
        "elements": "datetime,tempmax",
        "include": "fcst,obs",
        "forecastBasisDay": "-1",
        "key": api_key,
        "contentType": "json",
    }
    try:
        resp = await client.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("VC API failed", lat=lat, lon=lon, year=year, month=month, error=str(e))
        return []

    pairs: list[tuple[float, float]] = []
    for day in data.get("days", []):
        # VC returns forecast and observed in the same day record when include=fcst,obs
        # The 'tempmax' is actual observation; forecast is in 'forecastData' dict
        actual = day.get("tempmax")
        # When forecastBasisDay=-1, the 'source' field distinguishes; alternatively,
        # VC returns a separate 'forecastData' sub-object. We use 'forecastTempmax' if present.
        forecast = day.get("forecastTempmax") or day.get("tempmax")
        if actual is not None and forecast is not None:
            pairs.append((float(forecast), float(actual)))

    return pairs


async def compute_bias_for_station(
    station: str,
    months: list[int],
    api_key: str,
) -> dict[int, tuple[float, float, int]]:
    """Return {month: (bias_f, rmse_f, count)} for the given station."""
    coords = STATION_COORDS.get(station)
    if coords is None:
        logger.error("No coordinates for station", station=station)
        return {}

    lat, lon = coords
    today = date.today()
    years = [today.year - y for y in range(1, LOOKBACK_YEARS + 1)]

    # Accumulate errors per month
    month_errors: dict[int, list[float]] = {m: [] for m in months}

    async with httpx.AsyncClient() as client:
        for year in years:
            for month in months:
                pairs = await _fetch_vc_month(client, lat, lon, year, month, api_key)
                for fcst, obs in pairs:
                    month_errors[month].append(fcst - obs)
                await asyncio.sleep(0.5)  # stay within free tier rate limits

    results: dict[int, tuple[float, float, int]] = {}
    for month, errors in month_errors.items():
        if not errors:
            continue
        n = len(errors)
        mean_err = sum(errors) / n
        rmse = (sum(e * e for e in errors) / n) ** 0.5
        results[month] = (round(mean_err, 2), round(rmse, 2), n)
        logger.info("Bias computed", station=station, month=month, bias_f=mean_err, rmse_f=rmse, n=n)

    return results


async def upsert_bias(station: str, month: int, bias_f: float, rmse_f: float, count: int) -> None:
    """Upsert one station×month row into station_bias."""
    async with db_session() as db:
        stmt = (
            pg_insert(StationBias)
            .values(
                station=station,
                month=month,
                bias_f=Decimal(str(bias_f)),
                rmse_f=Decimal(str(rmse_f)),
                sample_count=count,
                source="visual_crossing",
            )
            .on_conflict_do_update(
                index_elements=["station", "month"],
                set_={
                    "bias_f": Decimal(str(bias_f)),
                    "rmse_f": Decimal(str(rmse_f)),
                    "sample_count": count,
                    "source": "visual_crossing",
                    "updated_at": sa.text("now()"),
                },
            )
        )
        await db.execute(stmt)


async def main(stations: list[str], months: list[int], api_key: str) -> None:
    logger.info("Starting bias computation", stations=stations, months=months)
    for station in stations:
        bias_map = await compute_bias_for_station(station, months, api_key)
        for month, (bias_f, rmse_f, count) in bias_map.items():
            await upsert_bias(station, month, bias_f, rmse_f, count)
            logger.info("Upserted bias", station=station, month=month, bias_f=bias_f, rmse_f=rmse_f, n=count)
    logger.info("Bias computation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute station forecast bias from Visual Crossing")
    parser.add_argument(
        "--stations",
        nargs="+",
        required=True,
        help='Station codes (e.g. KLGA KJFK) or "all" for all known stations',
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=["1-12"],
        help="Month numbers or ranges, e.g. 6 7 8 or 1-12 (default: all)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("VC_API_KEY") or os.environ.get("VISUALCROSSING_API_KEY")
    if not api_key:
        logger.error("VC_API_KEY environment variable not set")
        sys.exit(1)

    station_list = _all_stations() if args.stations == ["all"] else args.stations
    month_list = _parse_months(args.months)

    asyncio.run(main(station_list, month_list, api_key))
