"""Trading logic for paper and live modes."""
import asyncio
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.models import Bet, Market, Signal, Strategy

_position_lock = asyncio.Lock()


async def place_bet(
    db: AsyncSession,
    signal: Signal,
    market: Market,
    strategy: Strategy,
    yes_price: float,
    no_price: float,
) -> Bet | None:
    """Place a paper or live bet. Returns the created Bet or None if rejected."""
    if not await can_open_position(db):
        logger.info("Position limit reached, skipping", market_id=market.id)
        return None

    entry_price = yes_price if signal.direction == "YES" else no_price
    amount = _calculate_bet_size(
        edge=float(signal.edge),
        probability=float(signal.our_probability),
    )
    if amount <= 0:
        return None

    bet = Bet(
        signal_id=signal.id,
        market_id=market.id,
        strategy_id=strategy.id,
        mode=settings.trading_mode,
        direction=signal.direction,
        amount_usd=amount,
        entry_price=entry_price,
        outcome="pending",
        status="pending",
    )

    if settings.trading_mode == "live":
        success = await _execute_live_order(bet, market)
        if not success:
            bet.status = "failed"
            db.add(bet)
            return None
    else:
        logger.info(
            "Paper bet placed",
            market=market.question[:60],
            direction=signal.direction,
            amount=amount,
            price=entry_price,
            strategy=strategy.code,
        )

    db.add(bet)
    return bet


async def validate_live_readiness(db: AsyncSession) -> None:
    """Raise if the bot is not ready to switch to live trading."""
    result = await db.execute(
        select(func.count(Bet.id)).where(
            Bet.mode == "paper",
            Bet.outcome.in_(["win", "loss"]),
        )
    )
    resolved_count = result.scalar_one()

    if resolved_count < settings.min_paper_bets_for_live:
        raise RuntimeError(
            f"Need {settings.min_paper_bets_for_live} resolved paper bets, "
            f"have {resolved_count}"
        )

    wins_result = await db.execute(
        select(func.count(Bet.id)).where(
            Bet.mode == "paper", Bet.outcome == "win"
        )
    )
    wins = wins_result.scalar_one()
    win_rate = wins / resolved_count if resolved_count > 0 else 0.0

    if win_rate < settings.min_live_winrate:
        raise RuntimeError(
            f"Win rate {win_rate:.1%} below minimum {settings.min_live_winrate:.1%}"
        )

    logger.info(
        "Live readiness check passed",
        resolved_bets=resolved_count,
        win_rate=f"{win_rate:.1%}",
    )


async def can_open_position(db: AsyncSession) -> bool:
    result = await db.execute(
        select(func.count(Bet.id)).where(
            Bet.mode == settings.trading_mode,
            Bet.outcome == "pending",
        )
    )
    open_count = result.scalar_one()
    return open_count < settings.max_concurrent_positions


async def get_paper_balance(db: AsyncSession) -> float:
    """Calculate current paper balance = starting - pending - losses + wins."""
    wins_result = await db.execute(
        select(func.sum(Bet.pnl)).where(Bet.mode == "paper", Bet.outcome == "win")
    )
    losses_result = await db.execute(
        select(func.sum(Bet.pnl)).where(Bet.mode == "paper", Bet.outcome == "loss")
    )
    pending_result = await db.execute(
        select(func.sum(Bet.amount_usd)).where(Bet.mode == "paper", Bet.outcome == "pending")
    )

    wins = float(wins_result.scalar_one() or 0)
    losses = float(losses_result.scalar_one() or 0)
    pending = float(pending_result.scalar_one() or 0)

    return settings.paper_starting_balance_usd + wins + losses - pending


def _calculate_bet_size(edge: float, probability: float) -> float:
    """Fractional Kelly criterion bet sizing."""
    if edge <= 0 or probability <= 0 or probability >= 1:
        return 0.0

    denom = 1 - probability
    if denom <= 0:
        return 0.0

    kelly_full = edge / denom
    bet = settings.kelly_fraction * kelly_full * settings.paper_starting_balance_usd
    bet = max(0.5, min(bet, settings.max_bet_usd))
    return round(bet, 2)


async def _execute_live_order(bet: Bet, market: Market) -> bool:
    """Submit order via py-clob-client. Returns True on success."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType

        client = ClobClient(
            host="https://clob.polymarket.com",
            key=settings.polymarket_private_key,
            chain_id=137,
        )

        order_args = OrderArgs(
            token_id=market.polymarket_id,
            price=float(bet.entry_price),
            size=float(bet.amount_usd),
            side="BUY",
            is_yes=bet.direction == "YES",
        )
        result = client.create_order(order_args)
        bet.clob_order_id = result.get("orderID")
        logger.info("Live order placed", order_id=bet.clob_order_id, market=market.question[:60])
        return True

    except ImportError:
        logger.error("py-clob-client not installed")
        return False
    except Exception as e:
        error_str = str(e).lower()
        if "insufficient" in error_str:
            logger.critical("Insufficient funds — disabling live mode")
        elif "closed" in error_str:
            logger.warning("Market closed before order placement")
        else:
            logger.error("CLOB order failed", error=str(e))
        return False
