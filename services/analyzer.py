"""LLM-based market analysis via OpenRouter."""
import asyncio
import json
import re
from typing import Literal

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

from config import settings
from services.weather import WeatherSummary

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_log = logging.getLogger(__name__)
# Conservative concurrency — OpenRouter free/starter tiers cap around 20 req/min
_sem = asyncio.Semaphore(3)

WEATHER_ANALYSIS_PROMPT = """You are a prediction markets analyst specializing in weather events.

Market: "{question}"
Current YES price: {yes_price} (market implies {yes_pct:.1f}% probability)
Current NO price: {no_price}
Strategy: {strategy_name} — {strategy_description}

GFS Forecast for station {station} on {date}:
- Mean max temperature: {mean_max_f}°F
- P10/P90 range: {p10_f}°F – {p90_f}°F
- Ensemble members above threshold ({threshold}°F): {pct_above:.1%}
- Ensemble members below threshold: {pct_below:.1%}
- Precipitation forecast: {precip_mm}mm
- Hours until market close: {hours_ahead}h

{strategy_extra}

Return ONLY valid JSON, no markdown, no preamble:
{{
  "our_probability": <float 0.0-1.0>,
  "confidence": "<high|medium|low>",
  "direction": "<YES|NO|null>",
  "edge": <float, difference between our probability and market price for chosen direction>,
  "reasoning": "<brief explanation under 100 words>",
  "data_quality": "<sufficient|insufficient>"
}}

Rules for null direction (no bet):
- confidence != "high"
- |edge| < {min_edge}
- data_quality == "insufficient"
- hours_ahead > 48
- Strategy S2: more than 30 minutes since last GFS update
- Strategy S5: hours_until_close < 6 or > 24
"""


class LLMSignal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    our_probability: float
    confidence: Literal["high", "medium", "low"]
    direction: Literal["YES", "NO"] | None
    edge: float
    reasoning: str
    data_quality: Literal["sufficient", "insufficient"]

    @field_validator("our_probability", "edge", mode="before")
    @classmethod
    def clamp_float(cls, v) -> float:
        return round(float(v), 4)


class AnalysisResult(BaseModel):
    signal: LLMSignal | None
    llm_model: str
    tokens_used: int
    raw_response: str | None = None


def _is_retryable_llm(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable_llm),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=10, max=120, jitter=5),
    before_sleep=before_sleep_log(_log, logging.WARNING),
    reraise=True,
)
async def _call_llm(client: httpx.AsyncClient, prompt: str, model: str) -> dict:
    resp = await client.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/polymarket-bot",
            "X-Title": "Polymarket Weather Bot",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 400,
        },
        timeout=45,
    )
    if resp.status_code == 429:
        logger.warning("OpenRouter rate limited — will retry with backoff")
        raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
    resp.raise_for_status()
    return resp.json()


async def analyze(
    question: str,
    yes_price: float,
    no_price: float,
    strategy_code: str,
    strategy_name: str,
    strategy_description: str,
    strategy_params: dict,
    forecast: WeatherSummary,
    station: str,
) -> AnalysisResult:
    """Run LLM analysis for a single market+strategy pair."""
    strategy_extra = _build_strategy_extra(strategy_code, strategy_params, forecast)

    prompt = WEATHER_ANALYSIS_PROMPT.format(
        question=question,
        yes_price=yes_price,
        yes_pct=yes_price * 100,
        no_price=no_price,
        strategy_name=strategy_name,
        strategy_description=strategy_description,
        station=station,
        date=forecast.target_date,
        mean_max_f=forecast.mean_max_f or "N/A",
        p10_f=forecast.p10_max_f or "N/A",
        p90_f=forecast.p90_max_f or "N/A",
        threshold=forecast.threshold or "N/A",
        pct_above=forecast.pct_above_threshold or 0.0,
        pct_below=forecast.pct_below_threshold or 0.0,
        precip_mm=forecast.precipitation_mm or 0.0,
        hours_ahead=forecast.hours_ahead or 0,
        min_edge=settings.min_edge,
        strategy_extra=strategy_extra,
    )

    model = settings.openrouter_model
    async with _sem:
        async with httpx.AsyncClient() as client:
            try:
                data = await _call_llm(client, prompt, model)
            except Exception as e:
                logger.warning("Primary LLM failed, trying fallback", error=str(e))
                try:
                    data = await _call_llm(client, prompt, settings.openrouter_fallback_model)
                    model = settings.openrouter_fallback_model
                except Exception as e2:
                    logger.error("Both LLM models failed", error=str(e2))
                    return AnalysisResult(signal=None, llm_model=model, tokens_used=0)

    raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    tokens = data.get("usage", {}).get("total_tokens", 0)

    signal = _parse_llm_response(raw, yes_price, no_price)
    if signal:
        _verify_edge(signal, yes_price, no_price)

    return AnalysisResult(signal=signal, llm_model=model, tokens_used=tokens, raw_response=raw)


async def health_check() -> bool:
    """Verify OpenRouter is reachable."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False


def _parse_llm_response(raw: str, yes_price: float, no_price: float) -> LLMSignal | None:
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Extract JSON object
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        logger.warning("No JSON found in LLM response", raw=raw[:200])
        return None

    try:
        parsed = json.loads(m.group())
    except json.JSONDecodeError as e:
        logger.warning("JSON decode failed", error=str(e), raw=raw[:200])
        return None

    if parsed.get("direction") is None:
        return None

    try:
        signal = LLMSignal.model_validate(parsed)
    except Exception as e:
        logger.warning("LLM signal validation failed", error=str(e), parsed=parsed)
        return None

    if not (0.0 <= signal.our_probability <= 1.0):
        logger.warning("LLM returned out-of-range probability", value=signal.our_probability)
        return None

    return signal


def _verify_edge(signal: LLMSignal, yes_price: float, no_price: float) -> None:
    if signal.direction == "YES":
        expected = signal.our_probability - yes_price
    else:
        expected = (1 - signal.our_probability) - no_price

    if abs(signal.edge - expected) > 0.02:
        logger.info("LLM edge mismatch, recomputing", reported=signal.edge, computed=expected)
        signal.edge = round(expected, 4)


def _build_strategy_extra(code: str, params: dict, forecast: WeatherSummary) -> str:
    if code == "S2":
        return "Note (S2): Only signal within 30 minutes of latest GFS model run."
    if code == "S4":
        return "Note (S4): Require agreement between at least 2 forecast models. Only signal if data_quality=sufficient."
    if code == "S5":
        hours = forecast.hours_ahead or 0
        return f"Note (S5): Market closes in {hours:.0f}h. Only signal if 6 <= hours_to_close <= 24."
    if code == "S6":
        threshold_pct = params.get("extreme_event_threshold", 0.8)
        return f"Note (S6): Apply contrarian view if market price exceeds {threshold_pct:.0%}. Check for hype bias."
    return ""
