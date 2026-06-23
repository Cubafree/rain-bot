"""APScheduler v3 — weather cycle synced to GFS publish times."""
import asyncio
import random
from datetime import datetime, timezone
from typing import Any

from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.models import Bet, CycleLog, Market, Signal, Strategy
from db.session import db_session
from services import analyzer, alerter, polymarket, weather
from services.event_router import parse_market
from services.resolver import run_resolution_cycle
from services.trader import can_open_position, place_bet
from services.whale_watcher import run_whale_cycle

_consecutive_empty_cycles = 0


class _CycleCounter:
    """Mutable LLM-call counter shared across one cycle (asyncio-safe, single-threaded)."""
    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0


def create_scheduler() -> AsyncIOScheduler:
    jobstores = {"default": MemoryJobStore()}

    try:
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
        sync_url = settings.database_url
        if "+asyncpg" in sync_url:
            sync_url = sync_url.replace("+asyncpg", "")
        jobstores = {"default": SQLAlchemyJobStore(url=sync_url)}
        logger.info("Using SQLAlchemy job store for scheduler persistence")
    except Exception as e:
        logger.warning("SQLAlchemy job store unavailable, using memory", error=str(e))

    sched = AsyncIOScheduler(jobstores=jobstores, timezone="UTC")

    # GFS nominal runs: 00, 06, 12, 18 UTC. Data available ~4h later.
    # Trigger at 04, 10, 16, 22 UTC (+ 15min buffer).
    sched.add_job(
        run_weather_cycle,
        CronTrigger(hour="4,10,16,22", minute=15, timezone="UTC"),
        id="weather_cycle",
        name="Weather trading cycle",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800,
        replace_existing=True,
    )

    # Resolution check every hour
    sched.add_job(
        run_resolution_cycle,
        CronTrigger(minute=45, timezone="UTC"),
        id="resolution_cycle",
        name="Bet resolution cycle",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    # Daily stats materialization at 00:30 UTC
    sched.add_job(
        materialize_daily_stats,
        CronTrigger(hour=0, minute=30, timezone="UTC"),
        id="daily_stats",
        name="Daily stats rollup",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    return sched


async def run_weather_cycle() -> None:
    global _consecutive_empty_cycles

    # Hard daily LLM budget guard — abort before touching OpenRouter
    today = datetime.now(timezone.utc).date()
    async with db_session() as db:
        spent_r = await db.execute(
            select(func.coalesce(func.sum(CycleLog.llm_cost_usd), 0)).where(
                func.date(CycleLog.started_at) == today
            )
        )
        spent_today = float(spent_r.scalar_one())

    if settings.max_daily_llm_cost_usd > 0 and spent_today >= settings.max_daily_llm_cost_usd:
        logger.warning(
            "Daily LLM budget exhausted — cycle aborted",
            spent_usd=spent_today,
            limit_usd=settings.max_daily_llm_cost_usd,
        )
        await alerter.send_alert(
            "warning",
            f"Daily LLM budget ${settings.max_daily_llm_cost_usd:.2f} reached "
            f"(spent ${spent_today:.3f}) — cycle skipped",
        )
        return

    cycle_log = CycleLog(
        mode=settings.trading_mode,
        started_at=datetime.now(timezone.utc),
        errors=[],
    )

    # LLM health check before spending a full cycle
    llm_ok = await analyzer.health_check()
    if not llm_ok:
        cycle_log.errors.append({"stage": "llm_health_check", "error": "OpenRouter unreachable"})
        cycle_log.finished_at = datetime.now(timezone.utc)
        async with db_session() as db:
            db.add(cycle_log)
        await alerter.send_alert("critical", "Weather cycle aborted — OpenRouter unreachable")
        return

    markets_raw = await polymarket.get_open_markets(category="weather")
    cycle_log.markets_scanned = len(markets_raw)

    async with db_session() as db:
        strategies = (
            await db.execute(
                select(Strategy).where(
                    Strategy.is_active.is_(True),
                    Strategy.category == "weather",
                )
            )
        ).scalars().all()

        llm_counter = _CycleCounter()
        for gm in markets_raw:
            if llm_counter.n >= settings.max_cycle_llm_calls:
                logger.info(
                    "Cycle LLM limit reached — stopping early",
                    limit=settings.max_cycle_llm_calls,
                    markets_scanned=cycle_log.markets_scanned,
                )
                break
            try:
                await _process_market(db, gm, strategies, cycle_log, llm_counter)
            except Exception as e:
                logger.error("Market processing failed", market_id=gm.id, error=str(e))
                cycle_log.errors.append({
                    "stage": "market_processing",
                    "market_id": gm.id,
                    "error": str(e),
                })

        cycle_log.finished_at = datetime.now(timezone.utc)
        db.add(cycle_log)

    if cycle_log.signals_found == 0:
        _consecutive_empty_cycles += 1
        await alerter.alert_cycle_no_signals(_consecutive_empty_cycles)
    else:
        _consecutive_empty_cycles = 0

    logger.info(
        "Weather cycle complete",
        markets=cycle_log.markets_scanned,
        signals=cycle_log.signals_found,
        bets=cycle_log.bets_placed,
        llm_tokens=cycle_log.llm_tokens_used,
    )


async def _process_market(
    db: AsyncSession,
    gm: Any,
    strategies: list[Strategy],
    cycle_log: CycleLog,
    llm_counter: "_CycleCounter",
) -> None:
    if hasattr(gm, 'volume') and gm.volume is not None and gm.volume < settings.min_market_volume_usd:
        logger.debug("Market skipped (volume too low)", question=gm.question[:80], volume=gm.volume)
        return

    # Skip markets with missing or invalid prices — a zero price inflates edge artificially.
    # Also skip near-resolved markets (price at extremes means essentially done).
    yes_p = getattr(gm, 'yes_price', None)
    no_p = getattr(gm, 'no_price', None)
    if yes_p is None or no_p is None or yes_p <= 0.0 or no_p <= 0.0:
        logger.debug("Market skipped (missing/zero price)", question=gm.question[:80], yes_price=yes_p, no_price=no_p)
        return
    if yes_p < 0.04 or yes_p > 0.96:
        logger.debug("Market skipped (price near resolution)", question=gm.question[:80], yes_price=yes_p)
        return

    parsed = parse_market(gm.question)

    if parsed.parse_confidence < 0.8 or parsed.station is None:
        logger.debug("Market skipped (low parse confidence)", question=gm.question[:80])
        return

    if parsed.target_date is None or parsed.latitude is None:
        return

    from datetime import date as _date
    days_until = (parsed.target_date - _date.today()).days
    if days_until < 0:
        logger.debug("Market target date in the past, skipping", station=parsed.station, days_until=days_until)
        return
    if days_until > 16:
        logger.debug("Market beyond 16-day forecast window, skipping", station=parsed.station, days_until=days_until)
        return

    # Upsert market record
    existing = (
        await db.execute(select(Market).where(Market.polymarket_id == gm.id))
    ).scalar_one_or_none()

    if existing is None:
        market = Market(
            polymarket_id=gm.id,
            question=gm.question,
            category="weather",
            city=parsed.city,
            weather_station=parsed.station,
            target_date=parsed.target_date,
            close_time=gm.endDateIso,
        )
        db.add(market)
        await db.flush()
    else:
        market = existing

    if market.resolved or market.parse_failed:
        return

    # Load station bias for this station+month from DB (populated by compute_bias.py)
    from db.models import StationBias
    bias_row = (
        await db.execute(
            select(StationBias).where(
                StationBias.station == parsed.station,
                StationBias.month == parsed.target_date.month,
            )
        )
    ).scalar_one_or_none()
    bias_f = float(bias_row.bias_f) if bias_row else 0.0

    forecast = await weather.get_forecast(
        station=parsed.station,
        target_date=parsed.target_date,
        latitude=parsed.latitude,
        longitude=parsed.longitude,
        threshold=parsed.threshold,
        threshold_unit=parsed.unit or "F",
        bias_f=bias_f,
    )

    if forecast is None:
        cycle_log.errors.append({
            "stage": "weather_fetch",
            "market_id": market.id,
            "error": "forecast unavailable",
        })
        return

    # A forecast with no data_source means the fetch degraded silently — don't bet on it.
    # Live data: forecast_source=None bets won 0/4 (-$20).
    if not getattr(forecast, "data_source", None):
        cycle_log.errors.append({
            "stage": "weather_fetch",
            "market_id": market.id,
            "error": "forecast missing data_source",
        })
        return

    ob = None
    if hasattr(gm, 'clobTokenIds') and gm.clobTokenIds:
        ob = await polymarket.get_order_book(gm.clobTokenIds[0])

    # Collect signals from all ELIGIBLE strategies, then place only the highest-edge
    # bet per market. Each strategy has its own entry gate (see _strategy_eligible) so
    # it only competes on the markets it is designed for — otherwise the ensemble
    # bypass produces a strategy-agnostic signal and S1 (first in the list) always
    # wins the tie, which is why only S1 was ever betting.
    signals: list[tuple[float, "Strategy", Any]] = []  # (|edge|, strategy, result)
    for strategy in strategies:
        if not _strategy_eligible(strategy, gm, market, forecast):
            continue
        try:
            result = await _get_signal(db, market, strategy, gm, forecast, cycle_log, llm_counter, ob)
            if result is not None:
                signals.append(result)
        except Exception as exc:
            logger.warning(
                "Strategy analysis failed",
                market_id=market.id,
                strategy=strategy.code,
                error=str(exc),
            )

    if not signals:
        return

    # Best edge wins — only one bet placed per market per cycle. On (near-)ties,
    # which are common because the ensemble bypass returns the same edge for every
    # eligible strategy, pick at random among the leaders so attribution is fair
    # instead of always crediting the first-loaded strategy (S1).
    top_edge = max(t[0] for t in signals)
    leaders = [t for t in signals if abs(t[0] - top_edge) < 1e-6]
    abs_edge, best_strategy, best_result = random.choice(leaders)
    await _place_signal_bet(db, market, best_strategy, gm, best_result, forecast, cycle_log)


def _strategy_eligible(strategy: Strategy, gm: Any, market: Market, forecast: Any) -> bool:
    """Per-strategy entry gate. A strategy only competes on the markets it targets.

    Gates are derived from each strategy's params so they can be tuned in the DB
    without code changes. Overlap between strategies is fine — the tie-break above
    distributes contested markets. Returns True (eligible) by default for any
    active strategy without an explicit gate.
    """
    code = strategy.code
    params = strategy.params or {}
    yes = gm.yes_price
    hours = forecast.hours_ahead
    spread = forecast.model_spread_f

    if code == "S1":  # Tail Hunter — only extreme-priced (longshot) markets
        if yes is None:
            return False
        max_yes = float(params.get("max_yes_price", 0.15))
        return yes <= max_yes or yes >= (1.0 - max_yes)

    if code == "S2":  # Model Update Sniper — short lead time, fresh model run
        return hours is not None and hours <= float(params.get("max_lead_hours", 36))

    if code == "S3":  # Niche Geo — skip the heavily-traded major cities
        excluded = {c.lower() for c in params.get("excluded_cities", [])}
        return (market.city or "").lower() not in excluded

    if code == "S4":  # Consensus Divergence — require tight multi-model agreement
        return spread is not None and spread <= float(params.get("max_spread_f", 2.0))

    if code == "S5":  # Time Decay — mid lead-time window
        lo = float(params.get("min_hours_to_close", 6))
        hi = float(params.get("max_hours_to_close", 24))
        return hours is not None and lo <= hours <= hi

    if code == "S7":  # Whale Following — driven by whale_watcher; needs a watchlist
        return bool(params.get("watchlist"))

    return True


async def _get_signal(
    db: AsyncSession,
    market: Market,
    strategy: Strategy,
    gm: Any,
    forecast: Any,
    cycle_log: CycleLog,
    llm_counter: "_CycleCounter",
    ob: Any = None,
) -> "tuple[float, Strategy, Any] | None":
    """Run LLM analysis for one strategy. Returns (|edge|, strategy, result) or None."""
    if ob is not None and ob.spread > settings.max_ob_spread:
        logger.debug(
            "Market skipped (wide spread)",
            question=market.question[:80],
            spread=ob.spread,
            max_spread=settings.max_ob_spread,
        )
        return None

    if llm_counter.n >= settings.max_cycle_llm_calls:
        return None
    llm_counter.n += 1
    result = await analyzer.analyze(
        question=market.question,
        yes_price=gm.yes_price,
        no_price=gm.no_price,
        strategy_code=strategy.code,
        strategy_name=strategy.name,
        strategy_description=strategy.description or "",
        strategy_params=strategy.params or {},
        forecast=forecast,
        station=market.weather_station,
    )

    cycle_log.llm_tokens_used = (cycle_log.llm_tokens_used or 0) + result.tokens_used
    cycle_log.llm_cost_usd = float(cycle_log.llm_cost_usd or 0) + result.cost_usd

    if result.signal is None:
        return None

    sig = result.signal

    if sig.direction is None:
        return None
    if sig.confidence != settings.min_confidence:
        logger.debug(
            "Signal dropped (low confidence)",
            question=market.question[:60],
            strategy=strategy.code,
            confidence=sig.confidence,
            required=settings.min_confidence,
        )
        return None
    if abs(sig.edge) < settings.min_edge:
        logger.debug(
            "Signal dropped (insufficient edge)",
            question=market.question[:60],
            strategy=strategy.code,
            edge=round(sig.edge, 4),
            min_edge=settings.min_edge,
        )
        return None
    # Cap claimed edge — anything this large is almost always spurious overconfidence
    # from the uncalibrated ensemble (pct_above saturating at 0/1), not a real opportunity.
    # Live data: |edge|>=0.70 won only 7% (4/57, -$194).
    if abs(sig.edge) >= settings.max_edge:
        logger.debug(
            "Signal dropped (edge too large — likely overconfident)",
            question=market.question[:60],
            strategy=strategy.code,
            edge=round(sig.edge, 4),
            max_edge=settings.max_edge,
        )
        return None

    return (abs(sig.edge), strategy, result)


async def _place_signal_bet(
    db: AsyncSession,
    market: Market,
    strategy: Strategy,
    gm: Any,
    analysis_result: Any,
    forecast: Any,
    cycle_log: CycleLog,
) -> None:
    """Persist signal and place bet for the best-edge signal on this market."""
    from sqlalchemy.exc import IntegrityError

    sig = analysis_result.signal

    signal = Signal(
        market_id=market.id,
        strategy_id=strategy.id,
        our_probability=sig.our_probability,
        market_price=gm.yes_price if sig.direction == "YES" else gm.no_price,
        edge=sig.edge,
        direction=sig.direction,
        confidence=sig.confidence,
        reasoning=sig.reasoning,
        forecast_data=forecast.model_dump(),
        forecast_source=forecast.data_source,
        llm_model=analysis_result.llm_model,
        tokens_used=analysis_result.tokens_used,
        source="live_cycle",
    )
    try:
        async with db.begin_nested():
            db.add(signal)
            await db.flush()
    except IntegrityError as e:
        if "uq_signal_market_strategy_day" in str(e) or "UniqueViolation" in str(e):
            logger.debug("Duplicate signal skipped", market_id=market.id, strategy=strategy.code)
            return
        raise
    cycle_log.signals_found = (cycle_log.signals_found or 0) + 1

    bet = await place_bet(
        db=db,
        signal=signal,
        market=market,
        strategy=strategy,
        yes_price=gm.yes_price,
        no_price=gm.no_price,
    )
    if bet:
        cycle_log.bets_placed = (cycle_log.bets_placed or 0) + 1


async def materialize_daily_stats() -> None:
    """Aggregate yesterday's stats into daily_stats table."""
    from datetime import date, timedelta
    from sqlalchemy import func
    from db.models import DailyStats

    yesterday = date.today() - timedelta(days=1)

    async with db_session() as db:
        wins_r = await db.execute(
            select(func.count(Bet.id)).where(
                func.date(Bet.placed_at) == yesterday, Bet.outcome == "win"
            )
        )
        losses_r = await db.execute(
            select(func.count(Bet.id)).where(
                func.date(Bet.placed_at) == yesterday, Bet.outcome == "loss"
            )
        )
        pnl_r = await db.execute(
            select(func.sum(Bet.pnl)).where(func.date(Bet.placed_at) == yesterday)
        )
        tokens_r = await db.execute(
            select(func.sum(CycleLog.llm_tokens_used)).where(
                func.date(CycleLog.started_at) == yesterday
            )
        )
        cost_r = await db.execute(
            select(func.sum(CycleLog.llm_cost_usd)).where(
                func.date(CycleLog.started_at) == yesterday
            )
        )
        cycles_r = await db.execute(
            select(func.count(CycleLog.id)).where(
                func.date(CycleLog.started_at) == yesterday
            )
        )
        signals_r = await db.execute(
            select(func.coalesce(func.sum(CycleLog.signals_found), 0)).where(
                func.date(CycleLog.started_at) == yesterday
            )
        )

        wins = wins_r.scalar_one() or 0
        losses = losses_r.scalar_one() or 0
        total_bets = wins + losses
        win_rate = (wins / total_bets) if total_bets > 0 else None

        stat = DailyStats(
            stat_date=yesterday,
            total_signals=int(signals_r.scalar_one() or 0),
            total_bets=total_bets,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            pnl_usd=pnl_r.scalar_one() or 0,
            llm_cost_usd=cost_r.scalar_one() or 0,
            cycle_count=cycles_r.scalar_one() or 0,
        )
        db.add(stat)
        logger.info("Daily stats materialized", date=yesterday, win_rate=win_rate)
