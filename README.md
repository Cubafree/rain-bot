# rain-trade

Automated Polymarket weather trading bot. Fetches open weather prediction markets, pulls GFS ensemble + ECMWF forecasts, and places bets when the model's probability diverges meaningfully from the market price.

## How it works

1. **Scheduler** fires every 4 hours aligned to GFS publish times (04/10/16/22 UTC)
2. **Parser** extracts city, threshold, and target date from each Polymarket question
3. **Forecast** fetches a 31-member GFS ensemble from Open-Meteo (real `pct_above_threshold`, P10/P90, ensemble mean) blended with ECMWF deterministic for a model-agreement delta
4. **Analyzer** either bypasses the LLM when ensemble consensus is overwhelming (≥85%) or sends a structured prompt to OpenRouter (DeepSeek by default) for a signal with direction, confidence, and edge
5. **Trader** applies quarter-Kelly sizing and places the bet via Polymarket's CLOB API
6. **Resolver** settles bets hourly once Polymarket publishes the outcome

## Stack

| Component | Tech |
|-----------|------|
| Language | Python 3.11+ |
| Async | asyncio + httpx |
| Scheduler | APScheduler v3 |
| DB | PostgreSQL + SQLAlchemy async + Alembic |
| LLM | OpenRouter (deepseek/deepseek-chat, gpt-4o-mini fallback) |
| Weather | Open-Meteo (ensemble API + ECMWF) |
| Markets | Polymarket Gamma API + CLOB API |
| Deploy | Railway (europe-west4) |

## Setup

```bash
pip install -r requirements.txt

# Copy and fill in env vars
cp .env.example .env

# Run migrations
alembic upgrade head

# Seed strategies (run once)
python scripts/seed_strategies.py

# Start
python main.py
```

### Required env vars

```
DATABASE_URL=postgresql+asyncpg://...
OPENROUTER_API_KEY=sk-or-...
TRADING_MODE=paper          # paper or live
DASHBOARD_PASSWORD=...
```

### Live mode additionally requires

```
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_PROXY_ADDRESS=0x...
```

## Safety gates

- `TRADING_MODE=paper` runs without touching real funds
- Live mode requires `MIN_PAPER_BETS_FOR_LIVE=30` resolved paper bets and `MIN_LIVE_WINRATE=0.52`
- Daily LLM spend hard-capped at `MAX_DAILY_LLM_COST_USD` (default $2)
- Markets skipped if: volume < $10k, price at extremes (< 4% or > 96%), target date past or > 16 days out, or model disagreement > 3°F

## Backtest

```bash
python backtest/run_backtest.py --start 2024-01-01 --end 2024-12-31
```

Uses ERA5 archive data with synthetic GFS-like noise for honest out-of-sample testing.

## Dashboard

FastAPI + Jinja2 dashboard at `/` — shows open positions, daily P&L, cycle logs, and signal history. Protected by HTTP basic auth.

## Station bias calibration

GFS systematically over- or under-estimates temperature for specific stations. Run once after collecting enough historical data:

```bash
VISUALCROSSING_API_KEY=... python scripts/compute_bias.py
```

Populates the `station_bias` table; corrections are applied automatically each cycle.
