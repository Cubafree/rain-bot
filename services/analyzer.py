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

GFS/ECMWF Forecast for station {station} on {date}:
- Mean max temperature (GFS/ECMWF blend): {mean_max_f}°F
- ECMWF mean max: {ecmwf_mean_max_f}°F
- GFS/ECMWF agreement: within {model_agreement_delta_f}°F
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


# Approximate cost per 1M total tokens by OpenRouter model slug
_MODEL_COST_PER_M: dict[str, float] = {
    "deepseek/deepseek-chat": 0.27,
    "deepseek/deepseek-r1": 2.50,
    "openai/gpt-4o-mini": 0.30,
}
_DEFAULT_COST_PER_M = 0.50


def _estimate_cost(model: str, tokens: int) -> float:
    rate = _MODEL_COST_PER_M.get(model, _DEFAULT_COST_PER_M)
    return round(tokens * rate / 1_000_000, 6)


class AnalysisResult(BaseModel):
    signal: LLMSignal | None
    llm_model: str
    tokens_used: int
    cost_usd: float = 0.0
    raw_response: str | None = None


def _is_retryable_llm(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 500, 502, 503, 504):
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
            "max_tokens": 2000,
        },
        timeout=90,
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
    model: str | None = None,
) -> AnalysisResult:
    """Run LLM analysis for a single market+strategy pair.

    Pass ``model`` to override the default (e.g. use a faster model for backtesting).
    """
    # Model agreement filter — skip LLM if GFS and ECMWF disagree too much
    delta = forecast.model_agreement_delta_f
    if delta is not None and delta > settings.max_model_agreement_delta_f:
        logger.info("Models disagree, skipping signal", delta_f=delta, question=question[:60])
        return AnalysisResult(signal=None, llm_model=model or settings.openrouter_model, tokens_used=0, raw_response="model_disagreement")

    # LLM bypass for high-certainty ensemble signals
    bypass = _try_bypass_signal(forecast, yes_price, settings.llm_bypass_pct_threshold, station)
    if bypass is not None:
        if bypass.signal:
            _verify_edge(bypass.signal, yes_price, 1 - yes_price)
        return bypass

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
        mean_max_f=forecast.mean_max_f if forecast.mean_max_f is not None else "N/A",
        ecmwf_mean_max_f=forecast.ecmwf_mean_max_f if forecast.ecmwf_mean_max_f is not None else "N/A",
        model_agreement_delta_f=forecast.model_agreement_delta_f if forecast.model_agreement_delta_f is not None else "N/A",
        p10_f=forecast.p10_max_f if forecast.p10_max_f is not None else "N/A",
        p90_f=forecast.p90_max_f if forecast.p90_max_f is not None else "N/A",
        threshold=forecast.threshold if forecast.threshold is not None else "N/A",
        pct_above=forecast.pct_above_threshold or 0.0,
        pct_below=forecast.pct_below_threshold or 0.0,
        precip_mm=forecast.precipitation_mm or 0.0,
        hours_ahead=forecast.hours_ahead or 0,
        min_edge=settings.min_edge,
        strategy_extra=strategy_extra,
    )

    model = model or settings.openrouter_model
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

    raw = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
    tokens = data.get("usage", {}).get("total_tokens", 0)

    signal = _parse_llm_response(raw, yes_price, no_price)
    if signal:
        _verify_edge(signal, yes_price, no_price)

    return AnalysisResult(
        signal=signal,
        llm_model=model,
        tokens_used=tokens,
        cost_usd=_estimate_cost(model, tokens),
        raw_response=raw,
    )


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


def _try_bypass_signal(
    forecast: WeatherSummary,
    yes_price: float,
    threshold: float,
    station: str,
) -> "AnalysisResult | None":
    """Skip LLM when ensemble has overwhelming consensus. Returns None to proceed with LLM."""
    pct = forecast.pct_above_threshold
    if pct is None or forecast.members < 20:
        return None

    if pct >= threshold:
        edge = round(pct - yes_price, 4)
        if abs(edge) < settings.min_edge:
            return None
        signal = LLMSignal(
            our_probability=round(pct, 4),
            confidence="high",
            direction="YES",
            edge=edge,
            reasoning=f"Bypass: {pct:.0%} ensemble above threshold, market at {yes_price:.2f}",
            data_quality="sufficient",
        )
        return AnalysisResult(signal=signal, llm_model="bypass", tokens_used=0)

    elif pct <= (1 - threshold):
        no_prob = round(1 - pct, 4)
        no_price_val = round(1 - yes_price, 4)
        edge = round(no_prob - no_price_val, 4)
        if abs(edge) < settings.min_edge:
            return None
        signal = LLMSignal(
            our_probability=no_prob,
            confidence="high",
            direction="NO",
            edge=edge,
            reasoning=f"Bypass: only {pct:.0%} ensemble above threshold, market NO at {no_price_val:.2f}",
            data_quality="sufficient",
        )
        return AnalysisResult(signal=signal, llm_model="bypass", tokens_used=0)

    return None


def _parse_llm_response(raw: str, yes_price: float, no_price: float) -> LLMSignal | None:
    if not raw:
        return None
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
