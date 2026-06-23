"""Compute station forecast bias from the bot's own forecasts vs NASA POWER actuals.

This is the leak-free way to calibrate `station_bias`: it compares the mean_max_f
the bot actually predicted (stored in signals.forecast_data) against the observed
daily max from NASA POWER (power.larc.nasa.gov) for the same station and date.
NASA POWER needs no API key, so this replaces the old Visual Crossing flow.

    bias_f[station, month] = mean( predicted_mean_max_f − actual_max_f )

`_apply_bias` later SUBTRACTS bias_f, so a positive bias (forecast runs hot) is
corrected downward. Only past target dates are used (NASA POWER has latency).

Usage:
    python scripts/compute_bias.py                       # all stations, all months
    python scripts/compute_bias.py --stations KLGA KJFK  # specific stations
    python scripts/compute_bias.py --min-samples 5       # require >=5 pairs/month
"""
import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.models import Market, Signal, StationBias
from db.session import db_session
from services import nasa_power
from services.event_router import FORECAST_COORDS, STATION_COORDS


def _coords(station: str) -> tuple[float, float] | None:
    """Coords the live forecast uses: FORECAST_COORDS override, else airport."""
    return FORECAST_COORDS.get(station) or STATION_COORDS.get(station)


async def _load_forecast_pairs(
    stations: list[str] | None,
) -> dict[str, list[tuple[date, float]]]:
    """Return {station: [(target_date, predicted_mean_max_f), ...]} from past markets."""
    today = date.today()
    by_station: dict[str, list[tuple[date, float]]] = defaultdict(list)

    async with db_session() as db:
        stmt = (
            select(Market.weather_station, Market.target_date, Signal.forecast_data)
            .join(Signal, Signal.market_id == Market.id)
            .where(
                Market.weather_station.isnot(None),
                Market.target_date.isnot(None),
                Market.target_date < today,
                Signal.forecast_data.isnot(None),
            )
        )
        if stations:
            stmt = stmt.where(Market.weather_station.in_(stations))
        rows = (await db.execute(stmt)).all()

    # Average predicted mean_max per (station, date) to dedupe multiple signals.
    acc: dict[tuple[str, date], list[float]] = defaultdict(list)
    for station, target_date, fdata in rows:
        if not isinstance(fdata, dict):
            continue
        pred = fdata.get("mean_max_f")
        if pred is None:
            continue
        acc[(station, target_date)].append(float(pred))

    for (station, target_date), preds in acc.items():
        by_station[station].append((target_date, sum(preds) / len(preds)))
    return by_station


async def compute_bias_for_station(
    station: str,
    pairs: list[tuple[date, float]],
    min_samples: int,
) -> dict[int, tuple[float, float, int]]:
    """Return {month: (bias_f, rmse_f, count)} pairing forecasts with NASA POWER actuals."""
    coords = _coords(station)
    if coords is None:
        logger.warning("No coordinates for station — skipping", station=station)
        return {}
    lat, lon = coords

    dates = [d for d, _ in pairs]
    actuals = await nasa_power.get_actual_max_range_f(lat, lon, min(dates), max(dates))
    if not actuals:
        logger.warning("NASA POWER returned no actuals", station=station)
        return {}

    month_errors: dict[int, list[float]] = defaultdict(list)
    for target_date, predicted in pairs:
        actual = actuals.get(target_date)
        if actual is None:
            continue
        month_errors[target_date.month].append(predicted - actual)

    results: dict[int, tuple[float, float, int]] = {}
    for month, errors in month_errors.items():
        n = len(errors)
        if n < min_samples:
            continue
        mean_err = sum(errors) / n
        rmse = (sum(e * e for e in errors) / n) ** 0.5
        results[month] = (round(mean_err, 2), round(rmse, 2), n)
        logger.info("Bias computed", station=station, month=month, bias_f=round(mean_err, 2), rmse_f=round(rmse, 2), n=n)
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
                source="nasa_power",
            )
            .on_conflict_do_update(
                index_elements=["station", "month"],
                set_={
                    "bias_f": Decimal(str(bias_f)),
                    "rmse_f": Decimal(str(rmse_f)),
                    "sample_count": count,
                    "source": "nasa_power",
                    "updated_at": sa.text("now()"),
                },
            )
        )
        await db.execute(stmt)


async def main(stations: list[str] | None, min_samples: int) -> None:
    logger.info("Loading the bot's past forecasts", stations=stations or "all")
    by_station = await _load_forecast_pairs(stations)
    if not by_station:
        logger.warning("No past forecasts found in signals — nothing to calibrate")
        return

    total = 0
    for station, pairs in sorted(by_station.items()):
        bias_map = await compute_bias_for_station(station, pairs, min_samples)
        for month, (bias_f, rmse_f, count) in bias_map.items():
            await upsert_bias(station, month, bias_f, rmse_f, count)
            total += 1
    logger.info("Bias computation complete", rows_upserted=total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate station_bias from bot forecasts vs NASA POWER")
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="Station codes (e.g. KLGA KJFK). Default: all stations seen in signals.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum forecast/actual pairs required per month (default: 3)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.stations, args.min_samples))
