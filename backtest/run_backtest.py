"""Backtest runner: replay strategies against historical resolved markets.

Usage:
    python backtest/run_backtest.py --days 90 --strategies S1 S2 S5
    python backtest/run_backtest.py --days 30 --dry-run
    python backtest/run_backtest.py --days 180 --max-llm-calls 200
"""
import argparse
import asyncio
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import diskcache

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import select

from config import settings
from db.models import Bet, Market, Signal, Strategy
from db.session import db_session
from services import analyzer, polymarket, weather
from services.event_router import parse_market

CACHE = diskcache.Cache("/tmp/backtest_cache", size_limit=500 * 1024 * 1024)


async def run_backtest(
    days: int,
    strategy_codes: list[str] | None,
    dry_run: bool,
    max_llm_calls: int,
) -> None:
    logger.info("Starting backtest", days=days, dry_run=dry_run, max_llm_calls=max_llm_calls)

    resolved_markets = await polymarket.get_resolved_markets(category="weather", days=days)
    logger.info("Fetched resolved markets", count=len(resolved_markets))

    async with db_session() as db:
        strategies = (
            await db.execute(
                select(Strategy).where(
                    Strategy.is_active.is_(True),
                    Strategy.category == "weather",
                )
            )
        ).scalars().all()

    if strategy_codes:
        strategies = [s for s in strategies if s.code in strategy_codes]

    logger.info("Running strategies", codes=[s.code for s in strategies])

    if not dry_run:
        cost_estimate = len(resolved_markets) * len(strategies) * 0.0005
        logger.info(
            "Estimated LLM calls",
            total=len(resolved_markets) * len(strategies),
            max_allowed=max_llm_calls,
            cost_usd=f"${cost_estimate:.2f}",
        )
        if len(resolved_markets) * len(strategies) > max_llm_calls:
            logger.warning(
                "LLM call count exceeds limit — will stop at limit",
                limit=max_llm_calls,
            )

    llm_calls = 0
    results = {"wins": 0, "losses": 0, "total_pnl": 0.0, "skipped": 0}

    for gm in resolved_markets:
        parsed = parse_market(gm.question)
        if parsed.parse_confidence < 0.8 or parsed.station is None:
            results["skipped"] += 1
            continue
        if parsed.target_date is None or parsed.latitude is None:
            results["skipped"] += 1
            continue

        as_of_date = parsed.target_date - timedelta(days=1)

        forecast = await _get_cached_forecast(
            parsed.station,
            parsed.target_date,
            as_of_date,
            parsed.latitude,
            parsed.longitude,
            parsed.threshold,
            parsed.unit or "F",
        )
        if forecast is None:
            results["skipped"] += 1
            continue

        for strategy in strategies:
            if not dry_run and llm_calls >= max_llm_calls:
                logger.warning("LLM call limit reached, stopping")
                break

            # Approximate market price at signal time (24h before close)
            signal_price = gm.yes_price

            # Skip fully-resolved prices — 0.0 causes division by zero, 1.0 gives 0 edge
            if signal_price <= 0.01 or signal_price >= 0.99:
                continue

            if dry_run:
                import random
                signal_result = _fake_signal(signal_price)
            else:
                signal_result = await analyzer.analyze(
                    question=gm.question,
                    yes_price=signal_price,
                    no_price=1 - signal_price,
                    strategy_code=strategy.code,
                    strategy_name=strategy.name,
                    strategy_description=strategy.description or "",
                    strategy_params=strategy.params or {},
                    forecast=forecast,
                    station=parsed.station,
                )
                llm_calls += 1
                # ~30 req/min — stays well within OpenRouter rate limits
                await asyncio.sleep(2.0)

            if signal_result.signal is None or signal_result.signal.direction is None:
                continue
            if signal_result.signal.confidence != "high":
                continue
            if abs(signal_result.signal.edge) < settings.min_edge:
                continue

            # Determine actual outcome (YES won = price resolved to 1.0)
            yes_resolved = gm.yes_price >= 0.99
            market_outcome = "YES" if yes_resolved else "NO"

            bet_won = signal_result.signal.direction == market_outcome
            amount = 5.0
            pnl = round(amount * (1.0 / signal_price - 1.0), 2) if bet_won else -amount

            if bet_won:
                results["wins"] += 1
            else:
                results["losses"] += 1
            results["total_pnl"] += pnl

            async with db_session() as db:
                market_r = await db.execute(
                    select(Market).where(Market.polymarket_id == gm.id)
                )
                market = market_r.scalar_one_or_none()
                if market is None:
                    market = Market(
                        polymarket_id=gm.id,
                        question=gm.question,
                        category="weather",
                        city=parsed.city,
                        weather_station=parsed.station,
                        target_date=parsed.target_date,
                        close_time=gm.endDateIso,
                        resolved=True,
                        outcome=market_outcome,
                    )
                    db.add(market)
                    await db.flush()

                strategy_r = await db.execute(
                    select(Strategy).where(Strategy.code == strategy.code)
                )
                db_strategy = strategy_r.scalar_one_or_none()
                if not db_strategy:
                    continue

                # Skip if a backtest signal already exists for this market+strategy
                # (prevents UniqueViolation on re-runs)
                existing_signal_r = await db.execute(
                    select(Signal.id).where(
                        Signal.market_id == market.id,
                        Signal.strategy_id == db_strategy.id,
                    )
                )
                if existing_signal_r.scalar_one_or_none() is not None:
                    continue

                sig = signal_result.signal
                signal_obj = Signal(
                    market_id=market.id,
                    strategy_id=db_strategy.id,
                    our_probability=sig.our_probability,
                    market_price=signal_price,
                    edge=sig.edge,
                    direction=sig.direction,
                    confidence=sig.confidence,
                    reasoning=sig.reasoning,
                    forecast_data=forecast.model_dump(),
                    llm_model=signal_result.llm_model,
                    tokens_used=signal_result.tokens_used,
                    source="backtest",
                )
                db.add(signal_obj)
                await db.flush()

                bet = Bet(
                    signal_id=signal_obj.id,
                    market_id=market.id,
                    strategy_id=db_strategy.id,
                    mode="backtest",
                    direction=sig.direction,
                    amount_usd=amount,
                    entry_price=signal_price,
                    exit_price=1.0 if bet_won else 0.0,
                    outcome="win" if bet_won else "loss",
                    status="resolved",
                    pnl=pnl,
                    resolved_at=datetime.now(timezone.utc),
                )
                db.add(bet)

    total = results["wins"] + results["losses"]
    win_rate = results["wins"] / total if total > 0 else 0.0

    logger.info(
        "Backtest complete",
        total_bets=total,
        wins=results["wins"],
        losses=results["losses"],
        win_rate=f"{win_rate:.1%}",
        total_pnl=f"${results['total_pnl']:.2f}",
        skipped=results["skipped"],
        llm_calls=llm_calls,
    )


async def _get_cached_forecast(
    station: str,
    target_date: date,
    as_of_date: date,
    lat: float,
    lon: float,
    threshold: float | None,
    unit: str,
):
    key = f"hf:{station}:{target_date.isoformat()}:{as_of_date.isoformat()}"
    cached = CACHE.get(key)
    if cached is not None:
        return cached

    result = await weather.get_historical_forecast(
        station=station,
        target_date=target_date,
        as_of_date=as_of_date,
        latitude=lat,
        longitude=lon,
        threshold=threshold,
        threshold_unit=unit,
    )
    if result is not None:
        CACHE.set(key, result, expire=86400 * 30)
    return result


def _fake_signal(market_price: float):
    """Return a randomised signal for dry runs (no LLM, tests full pipeline)."""
    import random
    from services.analyzer import AnalysisResult, LLMSignal
    direction = random.choice(["YES", "NO"])
    our_prob = random.uniform(0.55, 0.85)
    price = market_price if direction == "YES" else 1 - market_price
    edge = round(our_prob - price, 4)
    signal = LLMSignal(
        our_probability=our_prob,
        confidence="high",
        direction=direction,
        edge=edge,
        reasoning="dry-run fake signal",
        data_quality="sufficient",
    )
    return AnalysisResult(signal=signal, llm_model="dry-run", tokens_used=0)


def main():
    parser = argparse.ArgumentParser(description="Polymarket Bot Backtest Runner")
    parser.add_argument("--days", type=int, default=90, help="Days of history to backtest")
    parser.add_argument("--strategies", nargs="*", help="Strategy codes (default: all active)")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls, use random signals")
    parser.add_argument("--max-llm-calls", type=int, default=500, help="Max LLM API calls")
    args = parser.parse_args()

    asyncio.run(run_backtest(
        days=args.days,
        strategy_codes=args.strategies,
        dry_run=args.dry_run,
        max_llm_calls=args.max_llm_calls,
    ))


if __name__ == "__main__":
    main()
