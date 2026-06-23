# rain-trade — Claude Code Context

## What this is
Polymarket weather prediction-market trading bot. Fetches open weather markets from Polymarket, gets GFS/ECMWF ensemble forecasts from Open-Meteo, runs LLM analysis via OpenRouter, and places bets (paper or live).

## Current state
**Paper trading mode** — running on Railway (europe-west4, Netherlands) to bypass Polymarket geo-blocks. Target: 30+ paper bets, then evaluate for live switch.

Live switch threshold: `min_paper_bets_for_live=30`, `min_live_winrate=0.52`.

## Architecture
```
scheduler.py          — APScheduler, 4 cron jobs
  run_weather_cycle   — main loop, every 4h (04/10/16/22 UTC+15min)
  run_resolution_cycle— hourly at :45
  materialize_daily_stats — 00:30 UTC
  whale_watcher       — (separate)

services/
  weather.py          — Open-Meteo client (GFS025 ensemble + ECMWF det.)
  analyzer.py         — OpenRouter LLM call, signal parsing, bypass logic
  event_router.py     — parse Polymarket question → city/station/threshold/date
  polymarket.py       — Gamma API + CLOB API client
  trader.py           — Kelly sizing, place_bet()
  resolver.py         — settle bets, void expired markets
  alerter.py          — webhook alerts

db/
  models.py           — Strategy, Market, Signal, Bet, CycleLog, StationBias, DailyStats
  session.py          — AsyncSession via db_session()

scripts/
  compute_bias.py     — one-time station bias calibration (Visual Crossing)
backtest/
  run_backtest.py     — offline backtest runner
```

## Key design decisions
- **GFS ensemble**: `ensemble-api.open-meteo.com` with `models=gfs025` → 31 member columns (control + member01..30), hourly → daily max per member
- **Multi-model consensus** (`get_forecast`): GFS ensemble mean blended with a deterministic panel (ECMWF `ecmwf_ifs025`, ICON `icon_seamless`, GEM `gem_global`, JMA `jma_seamless`) in one Open-Meteo call. `pct_above` is parametric: `Normal(consensus_mean, effective_sigma)` with `effective_sigma² = ensemble_sigma² + model_spread²`. Replaces the old empirical member-count `pct_above` (saturated at 0/1 → longshot overconfidence) and the dead `ecmwf_ifs04` 55/45 blend (returned null). `data_source="gfs_ecmwf_consensus"`.
- **Forecast cache key** includes threshold+unit (not just station+date) — markets share station/day at different thresholds.
- **LLM bypass**: skip OpenRouter when `pct_above >= llm_bypass_pct_threshold (0.85)` and `members >= 20`. NOTE: bypass is strategy-agnostic (pure ensemble math) — this is why strategy differentiation must happen via entry gates, not the LLM prompt.
- **Strategy entry gates** (`scheduler._strategy_eligible`): each strategy only competes on markets matching its DB params — S1 tails (`yes<=0.15` or `>=0.85`), S2 short lead (`<=36h`), S3 excludes major cities, S4 tight model agreement (`model_spread<=2F`), S5 6-24h window, S7 needs whale watchlist. Without gates all strategies produced the same bypass signal and S1 always won the dedup tie. Dedup tie-break is now random among top-edge leaders, not first-loaded.
- **Correlated-exposure cap** (`scheduler._city_day_exposure_full`): one strategy holds at most `max_bets_per_city_day` (default 1) pending bets per (city, date) — threshold markets share one forecast. Different strategies are exempt.
- **Session safety**: `db.begin_nested()` SAVEPOINT for signal INSERT to survive `UniqueViolation` without poisoning the outer transaction
- **Forecast coords**: `FORECAST_COORDS` in `event_router.py` overrides airport coords for cities where the national met office reports from city centre (HK, Tokyo, Beijing, Shanghai, Bangkok, São Paulo)

## Bugs fixed (recent)
- PendingRollbackError on duplicate signal → SAVEPOINT fix
- Concurrent `asyncio.gather` on shared AsyncSession → sequential for loop
- Past-date markets accepted → `days_until < 0` filter
- Near-resolved markets (price < 0.04 or > 0.96) accepted → extremity filter
- "Below/minimum" markets sent `mean_max_f` to LLM → now sends `mean_min_f`
- Year-boundary date parse (Dec → "January" = next year) → fixed
- ZeroDivisionError on `entry_price=0` bets → guard added
- N+1 query in `_check_unresolved_markets` → single JOIN
- `datetime.utcnow()` deprecation → `datetime.now(timezone.utc)`
- GFS deterministic (1 member) → real 31-member ensemble
- Kelly zeroed highest-conviction bets (`prob>=1` guard) → clamp prob to [0.02,0.98] (FIX7)
- Longshot/overconfidence losses → `min_entry_price=0.10`, `max_edge=0.70` guardrails, skip null `data_source` (FIX8)
- Single-model overconfidence (pct saturating 0/1) → multi-model parametric consensus (FIX9)
- Only S1 ever bet (strategies undifferentiated, bypass + dedup tie always picked S1) → per-strategy entry gates + random tie-break (FIX10)
- Correlated same-city/day bets (33% of bets were 1 position at N× size) → per-strategy (city,date) exposure cap (FIX11)

## Backlog
- [ ] Resolution via actual weather data — `services/nasa_power.py` client now exists (NASA POWER actuals, no key); wire into resolver
- [x] Station bias — `compute_bias.py` rewritten to calibrate from bot's own forecasts vs NASA POWER actuals (no key); `station_bias` populated. Re-run periodically as samples grow.
- [x] Strategy differentiation — all of S1–S5,S7 now trade via entry gates (FIX10). S6 stays disabled. Watch per-strategy P&L once data accrues.
- [ ] Exposure cap is first-come per (city,date), not strictly best-edge — best-edge selection needs a cycle-level refactor (collect all city/day candidates, then pick max edge).
- [ ] Probability calibration is parametric-Normal (FIX9); revisit with isotonic/Platt once enough resolved bets exist to fit a real calibration curve.
- [ ] Polymarket strategy research for `poly bot 2` (separate project)

## Evaluation playbook (next check ~1 week out)
Query prod DB (`mode='paper'`) and split BEFORE/AFTER each deploy:
- FIX7 ~2026-06-02, FIX8 ~2026-06-17, FIX9/10/11 ~2026-06-23.
- Win-rate + PnL by entry-price band and edge band (was: longshot `price<0.10` 0/34, `edge>=0.70` 7%).
- Per-strategy W/L/PnL (confirm S2–S7 now trade).
- Per-(city,date,strategy) count == 1 (confirm exposure cap holds).
- Calibration: is realized win-rate ≈ `our_probability` per bucket?
Live-switch gate unchanged: `min_paper_bets_for_live=30`, `min_live_winrate=0.52`.

## Config knobs (`.env`)
| Var | Default | Notes |
|-----|---------|-------|
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `MAX_BET_USD` | 5.0 | Per bet cap (binds on every bet → sizing is effectively flat $5; Kelly inert until balance/cap rebalanced) |
| `MIN_EDGE` | 0.08 | Minimum probability edge to bet |
| `MAX_EDGE` | 0.70 | Drop edges above this — spurious overconfidence (FIX8) |
| `MIN_ENTRY_PRICE` | 0.10 | Reject longshot entries below this (FIX8) |
| `MAX_BETS_PER_CITY_DAY` | 1 | Per-strategy correlated-exposure cap (FIX11) |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly sizing; clamps prob to [0.02,0.98] (FIX7) |
| `LLM_BYPASS_PCT_THRESHOLD` | 0.85 | Ensemble consensus bypass |
| `MAX_CYCLE_LLM_CALLS` | 60 | LLM calls per weather cycle |
| `MAX_DAILY_LLM_COST_USD` | 2.0 | Daily OpenRouter budget |
| `OPENROUTER_MODEL` | deepseek/deepseek-chat | Primary LLM |

## Deploy
Railway → `main` branch auto-deploys. europe-west4 region. Entry point: `main.py`.
