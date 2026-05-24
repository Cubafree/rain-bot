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
    results = {
        "wins": 0, "losses": 0, "total_pnl": 0.0,
        "skip_parse": 0, "skip_forecast": 0, "skip_price": 0, "skip_resolved": 0,
    }

    # Log the first few questions so we can see what the parser is receiving
    for i, gm in enumerate(resolved_markets[:3]):
        logger.info(f"Sample market[{i}] question: {gm.question[:120]!r}  clobTokenIds={gm.clobTokenIds!r}")

    # ── Phase 1: parse (in-memory, fast) ────────────────────────────────────
    viable: list[tuple] = []
    for gm in resolved_markets:
        parsed = parse_market(gm.question)
        if parsed.parse_confidence < 0.8 or parsed.station is None:
            results["skip_parse"] += 1
            if results["skip_parse"] <= 5:
                logger.warning(
                    f"Skip parse: conf={parsed.parse_confidence:.2f} city={parsed.city!r} "
                    f"station={parsed.station!r} q={gm.question[:80]!r}"
                )
            continue
        if parsed.target_date is None or parsed.latitude is None:
            results["skip_parse"] += 1
            continue
        viable.append((gm, parsed))

    logger.info(f"After parse: {len(viable)} viable / {len(resolved_markets)} total")

    # ── Phase 2: parallel CLOB price fetch (up to 20 concurrent) ────────────
    _price_sem = asyncio.Semaphore(20)

    async def _fetch_price_bounded(gm):
        async with _price_sem:
            return await _get_signal_price(gm)

    logger.info("Fetching CLOB prices in parallel...")
    price_results = await asyncio.gather(*[_fetch_price_bounded(gm) for gm, _ in viable])

    # ── Phase 3: filter, then parallel weather fetch (up to 10 concurrent) ──
    with_price = []
    for (gm, parsed), (price, hours) in zip(viable, price_results):
        if price is None:
            if results["skip_price"] == 0:
                logger.warning(
                    f"First skip_price: clobTokenIds={gm.clobTokenIds!r} "
                    f"conditionId={gm.conditionId!r} endDateIso={gm.endDateIso!r}"
                )
            results["skip_price"] += 1
            continue
        if price <= 0.01 or price >= 0.99:
            results["skip_resolved"] += 1
            continue
        as_of_date = parsed.target_date - timedelta(days=1)
        with_price.append((gm, parsed, as_of_date, price, hours))

    logger.info(f"With CLOB price: {len(with_price)}, fetching weather in parallel...")

    _forecast_sem = asyncio.Semaphore(10)

    async def _fetch_forecast_bounded(item):
        gm, parsed, as_of_date, price, hours = item
        async with _forecast_sem:
            return await _get_cached_forecast(
                parsed.station,
                parsed.target_date,
                as_of_date,
                parsed.latitude,
                parsed.longitude,
                parsed.threshold,
                parsed.unit or "F",
                hours_ahead_override=hours,
            )

    forecast_results = await asyncio.gather(*[_fetch_forecast_bounded(item) for item in with_price])

    ready = []
    for item, forecast in zip(with_price, forecast_results):
        gm, parsed, as_of_date, price, hours = item
        if forecast is None:
            results["skip_forecast"] += 1
            continue
        ready.append((gm, parsed, price, forecast))

    logger.info(f"Ready for LLM: {len(ready)} markets × {len(strategies)} strategies")

    # ── Phase 4: sequential LLM + DB writes ─────────────────────────────────
    limit_reached = False
    for gm, parsed, signal_price, forecast in ready:
        if limit_reached:
            break
        for strategy in strategies:
            if not dry_run and llm_calls >= max_llm_calls:
                logger.warning("LLM call limit reached, stopping")
                limit_reached = True
                break

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
                await asyncio.sleep(2.0)

            if signal_result.signal is None or signal_result.signal.direction is None:
                continue
            if signal_result.signal.confidence != settings.min_confidence:
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
        f"Backtest complete — bets={total} wins={results['wins']} losses={results['losses']} "
        f"pnl=${results['total_pnl']:.2f} win_rate={win_rate:.1%} llm_calls={llm_calls} | "
        f"skips: parse={results['skip_parse']} forecast={results['skip_forecast']} "
        f"price={results['skip_price']} resolved={results['skip_resolved']}"
    )


async def _get_cached_forecast(
    station: str,
    target_date: date,
    as_of_date: date,
    lat: float,
    lon: float,
    threshold: float | None,
    unit: str,
    hours_ahead_override: float | None = None,
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
        hours_ahead_override=hours_ahead_override,
    )
    if result is not None:
        CACHE.set(key, result, expire=86400 * 30)
    return result


async def _get_signal_price(gm) -> tuple[float, float] | tuple[None, None]:
    """Return (price, hours_ahead) for the YES token at the latest viable signal time.

    Tries offsets 24h → 12h → 6h → 2h before market close. Most Polymarket weather
    markets open only a few hours before close, so 24h is rarely available.
    Returns (None, None) when no price can be found.
    """
    if gm.endDateIso is None:
        return None, None

    # Resolve YES token ID (prefer clobTokenIds, fall back to conditionId / Gamma lookup)
    if gm.clobTokenIds:
        yes_token_id = gm.clobTokenIds[0]
    elif gm.conditionId:
        token_cache_key = f"token_id:{gm.conditionId}"
        yes_token_id = CACHE.get(token_cache_key)
        if yes_token_id is None:
            yes_token_id = await polymarket.get_yes_token_id(gm.conditionId)
            if yes_token_id:
                CACHE.set(token_cache_key, yes_token_id, expire=86400 * 90)
        if not yes_token_id:
            return None, None
    else:
        token_cache_key = f"gamma_token:{gm.id}"
        yes_token_id = CACHE.get(token_cache_key)
        if yes_token_id is None:
            cond_id, clob_ids = await polymarket.get_market_token_ids(gm.id)
            if clob_ids:
                yes_token_id = clob_ids[0]
            elif cond_id:
                yes_token_id = await polymarket.get_yes_token_id(cond_id)
            if yes_token_id:
                CACHE.set(token_cache_key, yes_token_id, expire=86400 * 90)
        if not yes_token_id:
            return None, None

    # Try progressively shorter lead times — most markets open <24h before close
    # Cache sentinel ("MISS") for zero-volume tokens to avoid repeated HTTP calls on reruns
    _MISS = "MISS"
    for hours_back in (24, 12, 6, 2):
        signal_time = gm.endDateIso - timedelta(hours=hours_back)
        cache_key = f"clob:{yes_token_id}:{signal_time.strftime('%Y%m%d%H')}"
        cached = CACHE.get(cache_key)
        if cached == _MISS:
            continue
        if cached is not None:
            return cached, float(hours_back)
        price = await polymarket.get_price_at_time(
            token_id=yes_token_id,
            timestamp=signal_time,
        )
        if price is not None:
            CACHE.set(cache_key, price, expire=86400 * 30)
            return price, float(hours_back)
        # Cache the miss so we skip this slot on the next run
        CACHE.set(cache_key, _MISS, expire=86400 * 7)

    return None, None


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
