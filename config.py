from typing import Literal
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str

    # OpenRouter
    openrouter_api_key: str = Field(..., min_length=10)
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_fallback_model: str = "deepseek/deepseek-chat"
    # Faster model used during backtesting (no chain-of-thought needed)
    openrouter_backtest_model: str = "deepseek/deepseek-chat"

    # Polymarket (live only)
    polymarket_private_key: str | None = None
    polymarket_proxy_address: str | None = None

    # Market filters
    min_market_volume_usd: float = Field(10000.0)
    max_model_agreement_delta_f: float = Field(3.0, ge=0.5, le=10.0)
    llm_bypass_pct_threshold: float = Field(0.85, ge=0.70, le=0.99)
    # Floor on real forecast error (°F) folded into effective_sigma so pct_above
    # is not overconfident. Live calibration showed claimed 0.92-1.0 won only 33%
    # — effective_sigma (~2.3F) badly underestimated true error (station rmse 2-6F).
    default_forecast_rmse_f: float = Field(3.0, ge=0.0, le=15.0)

    # Trading
    trading_mode: Literal["paper", "live"] = "paper"
    max_bet_usd: float = Field(5.0, ge=0.1, le=100.0)
    min_edge: float = Field(0.08, ge=0.01, le=0.50)
    # Max claimed edge — anything above this is almost always spurious overconfidence
    # (uncalibrated ensemble pct_above saturating at 0/1). Live data: |edge|>=0.70 won
    # only 7% (4/57, -$194). Drop them.
    max_edge: float = Field(0.70, ge=0.10, le=1.0)
    # Min entry price — refuse extreme longshots. Live data: entry_price<0.10 won 0/34 (-$170).
    min_entry_price: float = Field(0.10, ge=0.0, le=0.50)
    max_ob_spread: float = Field(0.15, ge=0.0, le=1.0)
    min_confidence: Literal["high", "medium", "low"] = "high"
    paper_starting_balance_usd: float = Field(10000.0, ge=100.0)
    kelly_fraction: float = Field(0.25, ge=0.05, le=1.0)
    max_concurrent_positions: int = Field(20, ge=1, le=50)
    # Correlated-exposure cap: multiple threshold markets for the same city+date
    # share one forecast, so betting several is one position at N× size. Cap how
    # many bets a *single strategy* may hold per (city, date). Different strategies
    # may still each take their own bet on that city/date.
    max_bets_per_city_day: int = Field(1, ge=1, le=10)
    max_strategy_exposure_pct: float = Field(0.20, ge=0.01, le=1.0)
    max_daily_llm_cost_usd: float = Field(2.0, ge=0.0)

    # Live transition safety
    min_paper_bets_for_live: int = Field(30, ge=1)
    min_live_winrate: float = Field(0.52, ge=0.0, le=1.0)

    # Scheduler cycle guard: stop processing after this many LLM calls per cycle
    max_cycle_llm_calls: int = Field(60, ge=10, le=500)

    # GFS timing: nominal runs at 00/06/12/18 UTC, data available ~4h later
    gfs_publish_delay_hours: int = Field(4, ge=1, le=6)

    # App
    port: int = Field(8080, ge=1024, le=65535)
    environment: Literal["development", "production"] = "production"

    # Dashboard
    dashboard_user: str = "admin"
    dashboard_password: str = Field("changeme", min_length=8)

    # Bias calibration (Visual Crossing — free tier, one-time script only)
    visualcrossing_api_key: str | None = None

    # Alerting
    alert_webhook_url: str | None = None
    sentry_dsn: str | None = None

    def __repr__(self) -> str:
        safe = self.model_dump()
        for key in ("openrouter_api_key", "polymarket_private_key", "dashboard_password"):
            if safe.get(key):
                safe[key] = "***SET***"
        return f"Settings({safe})"

    @field_validator("polymarket_private_key", mode="before")
    @classmethod
    def validate_private_key(cls, v: str | None) -> str | None:
        if v and not v.startswith("0x"):
            raise ValueError("POLYMARKET_PRIVATE_KEY must start with 0x")
        return v

    @model_validator(mode="after")
    def live_mode_requires_keys(self) -> "Settings":
        if self.trading_mode == "live":
            if not self.polymarket_private_key:
                raise ValueError("POLYMARKET_PRIVATE_KEY required for live mode")
            if not self.polymarket_proxy_address:
                raise ValueError("POLYMARKET_PROXY_ADDRESS required for live mode")
        return self


settings = Settings()
