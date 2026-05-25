"""Quick smoke test for OpenRouter connection.

Usage:
    OPENROUTER_API_KEY=sk-or-... python backtest/test_llm.py
"""
import asyncio
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = """You are a prediction markets analyst.

Market: "Will the high temperature in New York on May 20, 2025 exceed 75°F?"
Current YES price: 0.55
Current NO price: 0.45
GFS Forecast: mean_max=77F, pct_above=62%, hours_ahead=24

Return ONLY valid JSON:
{"our_probability": 0.62, "confidence": "high", "direction": "YES", "edge": 0.07, "reasoning": "GFS shows majority of ensemble above threshold", "data_quality": "sufficient"}
"""


async def test_openrouter(api_key: str, model: str = "deepseek/deepseek-r1"):
    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"{'='*60}")

    start = time.time()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/polymarket-bot",
                "X-Title": "Polymarket Weather Bot",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            timeout=120,
        )
    elapsed = time.time() - start

    print(f"Status: {resp.status_code} ({elapsed:.1f}s)")
    if resp.status_code != 200:
        print(f"ERROR: {resp.text[:500]}")
        return

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    tokens = data.get("usage", {})
    finish_reason = data.get("choices", [{}])[0].get("finish_reason", "?")

    print(f"Finish reason: {finish_reason}")
    print(f"Tokens: {tokens}")
    print(f"Response ({len(content)} chars):")
    print(content[:600])
    if len(content) > 600:
        print(f"... (truncated, full={len(content)} chars)")


async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY env var")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    print(f"API key: {api_key[:12]}...")

    # Test primary model
    await test_openrouter(api_key, "deepseek/deepseek-r1")

    # Test fallback model
    await test_openrouter(api_key, "meta-llama/llama-3.1-8b-instruct:free")


if __name__ == "__main__":
    asyncio.run(main())
