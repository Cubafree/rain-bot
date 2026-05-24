"""FastAPI routes for the web dashboard."""
import asyncio
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Integer, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from config import settings
from db.models import Bet, CycleLog, DailyStats, Market, Signal, Strategy
from db.session import get_db
from web.auth import require_auth

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")

_backtest_state: dict = {
    "running": False,
    "started_at": None,
    "params": None,
    "error": None,
    "finished_at": None,
}


async def _run_backtest_task(days: int, strategy_codes: list[str] | None, dry_run: bool, max_llm_calls: int) -> None:
    from backtest.run_backtest import run_backtest
    _backtest_state["error"] = None
    try:
        await run_backtest(days=days, strategy_codes=strategy_codes, dry_run=dry_run, max_llm_calls=max_llm_calls)
    except Exception as exc:
        _backtest_state["error"] = str(exc)
    finally:
        _backtest_state["running"] = False
        _backtest_state["finished_at"] = datetime.now(timezone.utc).isoformat()

Auth = Annotated[str, Depends(require_auth)]
DB = Annotated[AsyncSession, Depends(get_db)]


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, _: Auth, db: DB):
    mode = settings.trading_mode

    balance_result = await db.execute(
        select(func.sum(Bet.pnl)).where(Bet.mode == mode)
    )
    total_pnl = float(balance_result.scalar_one() or 0)

    recent_signals = (
        await db.execute(
            select(Signal, Market, Strategy)
            .join(Market, Signal.market_id == Market.id)
            .join(Strategy, Signal.strategy_id == Strategy.id)
            .order_by(desc(Signal.created_at))
            .limit(10)
        )
    ).all()

    pending_bets = (
        await db.execute(
            select(func.count(Bet.id)).where(Bet.mode == mode, Bet.outcome == "pending")
        )
    ).scalar_one()

    return templates.TemplateResponse(request, "index.html", {
        "mode": mode,
        "total_pnl": total_pnl,
        "starting_balance": settings.paper_starting_balance_usd,
        "recent_signals": recent_signals,
        "pending_bets": pending_bets,
        "now": datetime.now(timezone.utc),
    })


@router.get("/markets", response_class=HTMLResponse)
async def markets(request: Request, _: Auth, db: DB):
    active_markets = (
        await db.execute(
            select(Market)
            .where(Market.resolved.is_(False), Market.parse_failed.is_(False))
            .order_by(Market.close_time)
            .limit(100)
        )
    ).scalars().all()

    latest_signals = (
        await db.execute(
            select(Signal, Market, Strategy)
            .join(Market, Signal.market_id == Market.id)
            .join(Strategy, Signal.strategy_id == Strategy.id)
            .order_by(desc(Signal.created_at))
            .limit(50)
        )
    ).all()

    return templates.TemplateResponse(request, "markets.html", {
        "markets": active_markets,
        "signals": latest_signals,
    })


@router.get("/bets", response_class=HTMLResponse)
async def bets_page(
    request: Request,
    _: Auth,
    db: DB,
    strategy_id: int | None = None,
    outcome: str | None = None,
    limit: int = 100,
):
    q = (
        select(Bet, Market, Strategy)
        .join(Market, Bet.market_id == Market.id)
        .join(Strategy, Bet.strategy_id == Strategy.id)
        .order_by(desc(Bet.placed_at))
    )
    if strategy_id:
        q = q.where(Bet.strategy_id == strategy_id)
    if outcome:
        q = q.where(Bet.outcome == outcome)
    q = q.limit(limit)

    bets_rows = (await db.execute(q)).all()
    strategies = (await db.execute(select(Strategy).order_by(Strategy.code))).scalars().all()

    return templates.TemplateResponse(request, "bets.html", {
        "bets": bets_rows,
        "strategies": strategies,
        "selected_strategy": strategy_id,
        "selected_outcome": outcome,
    })


@router.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request, _: Auth, db: DB):
    mode = settings.trading_mode

    strategy_stats = (
        await db.execute(
            select(
                Strategy.code,
                Strategy.name,
                func.count(Bet.id).label("total"),
                func.sum((Bet.outcome == "win").cast(Integer)).label("wins"),
                func.sum(Bet.pnl).label("total_pnl"),
            )
            .join(Bet, Strategy.id == Bet.strategy_id)
            .where(Bet.mode == mode, Bet.outcome.in_(["win", "loss"]))
            .group_by(Strategy.id, Strategy.code, Strategy.name)
            .order_by(Strategy.code)
        )
    ).all()

    daily_stats = (
        await db.execute(
            select(DailyStats).order_by(DailyStats.stat_date.desc()).limit(30)
        )
    ).scalars().all()

    return templates.TemplateResponse(request, "analytics.html", {
        "strategy_stats": strategy_stats,
        "daily_stats": daily_stats,
        "mode": mode,
    })


@router.post("/api/backtest/run")
async def backtest_run(
    _: Auth,
    days: Annotated[int, Form()] = 90,
    dry_run: Annotated[str | None, Form()] = None,
    max_llm_calls: Annotated[int, Form()] = 200,
    strategies: Annotated[list[str] | None, Form()] = None,
):
    is_dry_run = dry_run is not None
    if _backtest_state["running"]:
        return JSONResponse({"error": "already running"}, status_code=409)
    _backtest_state["running"] = True
    _backtest_state["started_at"] = datetime.now(timezone.utc).isoformat()
    _backtest_state["params"] = {"days": days, "dry_run": is_dry_run, "max_llm_calls": max_llm_calls, "strategies": strategies}
    _backtest_state["finished_at"] = None
    asyncio.create_task(_run_backtest_task(days, strategies or None, is_dry_run, max_llm_calls))
    return JSONResponse({"status": "started"})


@router.get("/api/backtest/status")
async def backtest_status(_: Auth):
    return JSONResponse(_backtest_state)


@router.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request, _: Auth, db: DB):
    strategies = (
        await db.execute(
            select(Strategy).where(Strategy.category == "weather").order_by(Strategy.code)
        )
    ).scalars().all()

    backtest_bets = (
        await db.execute(
            select(Bet, Market, Strategy)
            .join(Market, Bet.market_id == Market.id)
            .join(Strategy, Bet.strategy_id == Strategy.id)
            .where(Bet.mode == "backtest")
            .order_by(desc(Bet.placed_at))
            .limit(200)
            .options(selectinload(Bet.signal))
        )
    ).all()

    return templates.TemplateResponse(request, "backtest.html", {
        "strategies": strategies,
        "bets": backtest_bets,
    })


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, _: Auth, db: DB, limit: int = 50):
    cycle_logs = (
        await db.execute(
            select(CycleLog).order_by(desc(CycleLog.started_at)).limit(limit)
        )
    ).scalars().all()

    return templates.TemplateResponse(request, "logs.html", {
        "logs": cycle_logs,
    })


@router.get("/api/summary")
async def api_summary(_: Auth, db: DB):
    """JSON endpoint for real-time dashboard updates."""
    mode = settings.trading_mode

    pnl_r = await db.execute(
        select(func.sum(Bet.pnl)).where(Bet.mode == mode)
    )
    pending_r = await db.execute(
        select(func.count(Bet.id)).where(Bet.mode == mode, Bet.outcome == "pending")
    )
    last_cycle_r = await db.execute(
        select(CycleLog).order_by(desc(CycleLog.started_at)).limit(1)
    )

    last_cycle = last_cycle_r.scalar_one_or_none()

    return {
        "mode": mode,
        "total_pnl": float(pnl_r.scalar_one() or 0),
        "pending_bets": pending_r.scalar_one() or 0,
        "last_cycle_at": last_cycle.finished_at.isoformat() if last_cycle and last_cycle.finished_at else None,
        "last_cycle_bets": last_cycle.bets_placed if last_cycle else 0,
    }
