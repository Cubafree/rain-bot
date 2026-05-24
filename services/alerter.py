"""Discord/Slack webhook alerting for critical bot events."""
from typing import Literal

import httpx
from loguru import logger

from config import settings

AlertLevel = Literal["info", "warn", "critical"]

_LEVEL_EMOJI = {"info": "ℹ️", "warn": "⚠️", "critical": "🚨"}
_LEVEL_COLOR = {"info": 0x3498DB, "warn": 0xF39C12, "critical": 0xE74C3C}


async def send_alert(level: AlertLevel, message: str, **context) -> None:
    """Send alert to configured webhook (Discord/Slack format auto-detected)."""
    if not settings.alert_webhook_url:
        return

    url = settings.alert_webhook_url
    ctx_str = " | ".join(f"{k}={v}" for k, v in context.items()) if context else ""
    full_msg = f"{message} {ctx_str}".strip()

    logger.log(level.upper() if level != "warn" else "WARNING", f"ALERT: {full_msg}")

    payload = _build_discord_payload(level, full_msg)

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Failed to send alert", error=str(e))


def _build_discord_payload(level: AlertLevel, message: str) -> dict:
    emoji = _LEVEL_EMOJI.get(level, "")
    return {
        "embeds": [
            {
                "title": f"{emoji} Polymarket Bot — {level.upper()}",
                "description": message,
                "color": _LEVEL_COLOR.get(level, 0x95A5A6),
            }
        ]
    }


async def alert_live_bet_failed(market_question: str, error: str) -> None:
    await send_alert("critical", f"Live bet FAILED: {error}", market=market_question[:80])


async def alert_llm_budget_warning(daily_cost: float) -> None:
    await send_alert(
        "warn",
        f"LLM daily spend at ${daily_cost:.3f} (limit ${settings.max_daily_llm_cost_usd:.2f})",
    )


async def alert_cycle_no_signals(consecutive: int) -> None:
    if consecutive >= 3:
        await send_alert("critical", f"{consecutive} consecutive cycles with 0 signals")


async def alert_mode_switch(from_mode: str, to_mode: str) -> None:
    await send_alert("info", f"Trading mode switched: {from_mode} → {to_mode}")
