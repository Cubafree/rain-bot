"""Polymarket Gamma API client."""
from datetime import datetime
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)
import logging

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

_log = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    return False


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
    clobTokenIds: list[str] = []

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
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=5, max=60, jitter=5),
    before_sleep=before_sleep_log(_log, logging.WARNING),
    reraise=True,
)
async def _get(client: httpx.AsyncClient, url: str, **params) -> Any:
    resp = await client.get(url, params=params, timeout=20)
    if resp.status_code == 429:
        logger.warning("Polymarket rate limited, will retry")
        raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
    resp.raise_for_status()
    return resp.json()


async def get_open_markets(category: str = "weather") -> list[GammaMarket]:
    """Fetch open prediction markets filtered by category tag via /events endpoint."""
    import asyncio

    markets: list[GammaMarket] = []
    offset = 0
    limit = 100

    async with httpx.AsyncClient() as client:
        while True:
            try:
                events = await _get(
                    client,
                    f"{GAMMA_BASE}/events",
                    closed="false",
                    active="true",
                    tag_slug=category,
                    order="endDate",
                    ascending="true",
                    limit=limit,
                    offset=offset,
                )
            except Exception as e:
                logger.error(f"Failed to fetch open weather events: {e}")
                break

            if not isinstance(events, list) or not events:
                break

            for event in events:
                tags = event.get("tags", [])
                event_title = event.get("title", "")
                for mkt in event.get("markets", []):
                    item = dict(mkt)
                    item.setdefault("tags", tags)
                    item["endDateIso"] = mkt.get("endDate") or event.get("endDate")
                    mkt_q = mkt.get("question", "")
                    item["question"] = f"{event_title} {mkt_q}".strip() if event_title else mkt_q
                    try:
                        markets.append(GammaMarket.model_validate(item))
                    except Exception:
                        pass

            await asyncio.sleep(0.3)
            if len(events) < limit:
                break
            offset += limit

    logger.info(f"Fetched {len(markets)} open weather markets")
    return markets


async def get_resolved_markets(category: str = "weather", days: int = 180) -> list[GammaMarket]:
    """Fetch resolved weather markets for backtesting via the /events endpoint."""
    import asyncio
    from datetime import timedelta, timezone as tz

    cutoff = datetime.now(tz.utc) - timedelta(days=days)
    markets: list[GammaMarket] = []
    offset = 0
    limit = 100

    async with httpx.AsyncClient() as client:
        while True:
            try:
                events = await _get(
                    client,
                    f"{GAMMA_BASE}/events",
                    closed="true",
                    tag_slug="weather",
                    order="endDate",
                    ascending="false",
                    limit=limit,
                    offset=offset,
                )
            except Exception as e:
                logger.error(f"Failed to fetch resolved weather events: {e}")
                break

            if not isinstance(events, list) or not events:
                break

            stop = False
            for event in events:
                # parse event end date to check cutoff
                end_str = event.get("endDate") or event.get("closedTime", "")
                try:
                    end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=tz.utc)
                    if end_dt < cutoff:
                        stop = True
                        break
                except Exception:
                    pass

                tags = event.get("tags", [])
                event_title = event.get("title", "")
                for mkt in event.get("markets", []):
                    item = dict(mkt)
                    # inject event-level fields the nested market lacks
                    item.setdefault("tags", tags)
                    item["endDateIso"] = mkt.get("endDate") or event.get("endDate")
                    # combine event title (has city+date) with market question (has threshold)
                    mkt_q = mkt.get("question", "")
                    item["question"] = f"{event_title} {mkt_q}".strip() if event_title else mkt_q
                    try:
                        markets.append(GammaMarket.model_validate(item))
                    except Exception:
                        pass

            await asyncio.sleep(0.5)
            if stop or len(events) < limit:
                break
            offset += limit

    logger.info(f"Fetched {len(markets)} resolved weather markets (last {days}d)")
    return markets


async def get_market_token_ids(market_id: str) -> tuple[str | None, list[str]]:
    """Fetch conditionId and clobTokenIds from Gamma /markets/{id}.

    Used as last-resort fallback when the events endpoint doesn't populate
    those fields on nested market objects.
    Returns (condition_id_or_None, clob_token_ids_list).
    """
    import json as _json

    async with httpx.AsyncClient() as client:
        try:
            data = await _get(client, f"{GAMMA_BASE}/markets/{market_id}")
        except Exception as e:
            logger.warning("Gamma market fetch failed", error=str(e), market_id=market_id)
            return None, []

    condition_id = data.get("conditionId") or None
    raw_clob = data.get("clobTokenIds") or []
    if isinstance(raw_clob, str):
        try:
            raw_clob = _json.loads(raw_clob)
        except Exception:
            raw_clob = []

    return condition_id, list(raw_clob)


async def get_yes_token_id(condition_id: str) -> str | None:
    """Fetch the YES clobTokenId for a market given its hex conditionId.

    Falls back gracefully — returns None on any error.
    Used when clobTokenIds is not populated on the GammaMarket object.
    """
    async with httpx.AsyncClient() as client:
        try:
            data = await _get(client, f"{CLOB_BASE}/markets/{condition_id}")
        except Exception as e:
            logger.warning("CLOB market lookup failed", error=str(e), condition_id=condition_id)
            return None

    tokens = data.get("tokens", [])
    for token in tokens:
        if token.get("outcome", "").lower() in ("yes", "true"):
            return str(token.get("token_id", ""))
    # If outcomes aren't labelled, YES is index 0 by convention
    if tokens:
        return str(tokens[0].get("token_id", ""))
    return None


async def get_price_at_time(
    token_id: str,
    timestamp: datetime,
    interval: str = "1h",
) -> float | None:
    """Fetch historical YES price closest to a given timestamp.

    token_id must be the CLOB token ID (numeric string from clobTokenIds[0]),
    NOT the hex conditionId — the CLOB /prices-history endpoint uses token IDs.
    """
    async with httpx.AsyncClient() as client:
        try:
            data = await _get(
                client,
                f"{CLOB_BASE}/prices-history",
                market=token_id,
                interval=interval,
                startTs=int(timestamp.timestamp()),
                endTs=int(timestamp.timestamp()) + 3600,
            )
        except Exception as e:
            logger.warning("Failed to fetch price history", error=str(e), token_id=token_id)
            return None

    history = data.get("history", [])
    if not history:
        return None

    return float(history[0].get("p", 0.5))
