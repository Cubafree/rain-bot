"""Resolution cycle: check resolved markets and settle open bets."""
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Bet, Market
from db.session import db_session
from services import polymarket, weather


async def run_resolution_cycle() -> None:
    """Settle pending bets for resolved markets."""
    async with db_session() as db:
        await _settle_resolved_markets(db)
        await _check_unresolved_markets(db)


async def _settle_resolved_markets(db: AsyncSession) -> None:
    """Update bets for markets that have resolved via Polymarket."""
    # Find pending bets on markets we think are unresolved
    pending_bets = (
        await db.execute(
            select(Bet)
            .join(Market)
            .where(Bet.outcome == "pending", Market.resolved.is_(False))
            .order_by(Bet.placed_at)
        )
    ).scalars().all()

    if not pending_bets:
        return

    # Fetch resolved markets from Polymarket
    resolved_gamma = await polymarket.get_resolved_markets(days=7)
    resolved_map = {m.id: m for m in resolved_gamma if m.id}

    settled = 0
    for bet in pending_bets:
        market_result = await db.execute(
            select(Market).where(Market.id == bet.market_id)
        )
        market = market_result.scalar_one_or_none()
        if not market:
            continue

        gamma_market = resolved_map.get(market.polymarket_id)
        if not gamma_market or not gamma_market.closed:
            continue

        # Determine outcome from Polymarket resolution data
        # Gamma API encodes winner via outcomePrices: [1.0, 0.0] = YES won
        yes_price = gamma_market.yes_price
        market_outcome = "YES" if yes_price >= 0.99 else "NO" if yes_price <= 0.01 else None

        if market_outcome is None:
            continue

        # Mark market as resolved
        market.resolved = True
        market.outcome = market_outcome

        # Settle the bet
        bet_won = bet.direction == market_outcome
        bet.outcome = "win" if bet_won else "loss"
        bet.status = "resolved"
        bet.exit_price = 1.0 if bet_won else 0.0
        bet.resolved_at = datetime.now(timezone.utc)

        amount = float(bet.amount_usd)
        entry = float(bet.entry_price)
        if bet_won:
            # Guard against entry_price == 0 (degenerate bets placed at zero price)
            bet.pnl = round(amount * (1.0 / entry - 1.0), 2) if entry > 0 else 0.0
        else:
            bet.pnl = round(-amount, 2)

        settled += 1
        logger.info(
            "Bet settled",
            bet_id=bet.id,
            direction=bet.direction,
            outcome=market_outcome,
            pnl=bet.pnl,
        )

    if settled:
        logger.info("Resolution cycle complete", settled=settled)


async def _check_unresolved_markets(db: AsyncSession) -> None:
    """Void bets for markets that closed without resolution (delisted)."""
    now = datetime.now(timezone.utc)

    # Single query — join Market to avoid N+1
    rows = (
        await db.execute(
            select(Bet, Market)
            .join(Market, Bet.market_id == Market.id)
            .where(
                Bet.outcome == "pending",
                Market.close_time.isnot(None),
                Market.close_time < now,
                Market.resolved.is_(False),
            )
        )
    ).all()

    for bet, market in rows:
        days_past_close = (now - market.close_time).days
        if days_past_close < 3:
            continue

        bet.outcome = "voided"
        bet.status = "voided"
        bet.pnl = 0.0
        bet.resolved_at = now
        logger.warning("Bet voided — market expired unresolved", bet_id=bet.id, market_id=market.id)
