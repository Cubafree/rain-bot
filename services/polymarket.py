"""Polymarket Gamma API client."""
from datetime import datetime
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)
import logging

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

_log = logging.getLogger(__name__)


class GammaMarket(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    question: str
    outcomePrices: list[str] = []
    endDateIso: datetime | None = None
    volume: float = 0.0
    closed: bool = False
    active: bool = True
    tags: list[dict] = []
    conditionId: str | None = None

    @field_validator("outcomePrices", mode="before")
    @classmethod
    def parse_prices(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v or []

    @property
    def yes_price(self) -> float:
        try:
            return float(self.outcomePrices[0])
        except (IndexError, ValueError):
            return 0.5

    @property
    def no_price(self) -> float:
        try:
            return float(self.outcomePrices[1])
        except (IndexError, ValueError):
            return 0.5

    _WEATHER_KEYWORDS = ("temperature", "rain", "snow", "high", "°f", "°c", "heat", "cold", "precipitation", "forecast")

    @property
    def is_weather(self) -> bool:
        if any(t.get("slug") == "weather" for t in self.tags):
            return True
        q = self.question.lower()
        return any(kw in q for kw in self._WEATHER_KEYWORDS)


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
    before_sleep=before_sleep_log(_log, logging.WARNING),
    reraise=True,
)
async def _get(client: httpx.AsyncClient, url: str, **params) -> Any:
    resp = await client.get(url, params=params, timeout=20)
    if resp.status_code == 429:
        logger.warning("Polymarket rate limited, backing off")
        raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
    resp.raise_for_status()
    return resp.json()


async def get_open_markets(category: str = "weather") -> list[GammaMarket]:
    """Fetch open prediction markets filtered by category tag."""
    async with httpx.AsyncClient() as client:
        try:
            data = await _get(
                client,
                f"{GAMMA_BASE}/markets",
                tag=category,
                closed="false",
                active="true",
                limit=200,
            )
        except Exception as e:
            logger.error(f"Failed to fetch open markets: {e}")
            return []

    markets = []
    raw_list = data if isinstance(data, list) else data.get("markets", [])
    for item in raw_list:
        try:
            markets.append(GammaMarket.model_validate(item))
        except Exception as e:
            logger.warning("Failed to parse market", error=str(e), item=item.get("id"))
    return markets


async def get_resolved_markets(category: str = "weather", days: int = 180) -> list[GammaMarket]:
    """Fetch resolved markets for backtesting."""
    from datetime import timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    markets = []
    offset = 0
    limit = 100

    async with httpx.AsyncClient() as client:
        while True:
            try:
                data = await _get(
                    client,
                    f"{GAMMA_BASE}/markets",
                    closed="true",
                    limit=limit,
                    offset=offset,
                )
            except Exception as e:
                logger.error(f"Failed to fetch resolved markets: {e}")
                break

            raw_list = data if isinstance(data, list) else data.get("markets", [])
            if not raw_list:
                break

            stop = False
            for item in raw_list:
                try:
                    m = GammaMarket.model_validate(item)
                except Exception:
                    continue

                # stop paginating once we're past the cutoff window
                if m.endDateIso and m.endDateIso < cutoff:
                    stop = True
                    break

                if m.is_weather:
                    markets.append(m)

            if stop or len(raw_list) < limit:
                break
            offset += limit

    logger.info(f"Fetched {len(markets)} resolved weather markets (last {days}d)")
    return markets


async def get_price_at_time(
    condition_id: str,
    timestamp: datetime,
    interval: str = "1h",
) -> float | None:
    """Fetch historical market price closest to a given timestamp."""
    async with httpx.AsyncClient() as client:
        try:
            data = await _get(
                client,
                f"{CLOB_BASE}/prices-history",
                market=condition_id,
                interval=interval,
                startTs=int(timestamp.timestamp()),
                endTs=int(timestamp.timestamp()) + 3600,
            )
        except Exception as e:
            logger.warning("Failed to fetch price history", error=str(e), condition_id=condition_id)
            return None

    history = data.get("history", [])
    if not history:
        return None

    return float(history[0].get("p", 0.5))
