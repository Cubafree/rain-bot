"""Initial schema with all tables and indexes

Revision ID: 001
Revises:
Create Date: 2026-05-24
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "strategies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.String(20), unique=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("category", sa.String(50)),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("params", postgresql.JSONB()),
        sa.Column("prompt_extension", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "markets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("polymarket_id", sa.String(100), unique=True, nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("category", sa.String(50)),
        sa.Column("city", sa.String(100)),
        sa.Column("weather_station", sa.String(10)),
        sa.Column("target_date", sa.Date()),
        sa.Column("close_time", sa.DateTime(timezone=True)),
        sa.Column("resolved", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("outcome", sa.String(10)),
        sa.Column("parse_failed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("outcome IN ('YES', 'NO')", name="chk_market_outcome"),
    )
    op.create_index("idx_markets_resolved", "markets", ["resolved"],
                    postgresql_where=sa.text("resolved = false"))

    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("market_id", sa.Integer(), sa.ForeignKey("markets.id"), nullable=False),
        sa.Column("strategy_id", sa.Integer(), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("our_probability", sa.Numeric(5, 4)),
        sa.Column("market_price", sa.Numeric(5, 4)),
        sa.Column("edge", sa.Numeric(5, 4)),
        sa.Column("direction", sa.String(10)),
        sa.Column("confidence", sa.String(20)),
        sa.Column("reasoning", sa.Text()),
        sa.Column("forecast_data", postgresql.JSONB()),
        sa.Column("llm_model", sa.String(100)),
        sa.Column("tokens_used", sa.Integer()),
        sa.Column("source", sa.String(20), nullable=False, server_default="live_cycle"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("market_id", "strategy_id", name="uq_signal_market_strategy_day"),
        sa.CheckConstraint("direction IN ('YES', 'NO')", name="chk_signal_direction"),
        sa.CheckConstraint("confidence IN ('high', 'medium', 'low')", name="chk_signal_confidence"),
        sa.CheckConstraint("source IN ('live_cycle', 'backtest')", name="chk_signal_source"),
    )
    op.create_index("idx_signals_market_strategy", "signals", ["market_id", "strategy_id"])

    op.create_table(
        "bets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("signal_id", sa.Integer(), sa.ForeignKey("signals.id")),
        sa.Column("market_id", sa.Integer(), sa.ForeignKey("markets.id"), nullable=False),
        sa.Column("strategy_id", sa.Integer(), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("mode", sa.String(10), nullable=False),
        sa.Column("direction", sa.String(10)),
        sa.Column("amount_usd", sa.Numeric(10, 2)),
        sa.Column("entry_price", sa.Numeric(5, 4)),
        sa.Column("exit_price", sa.Numeric(5, 4)),
        sa.Column("outcome", sa.String(10), server_default="pending"),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("pnl", sa.Numeric(10, 2)),
        sa.Column("clob_order_id", sa.String(100)),
        sa.Column("placed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("mode IN ('paper', 'live', 'backtest')", name="chk_bet_mode"),
        sa.CheckConstraint("direction IN ('YES', 'NO')", name="chk_bet_direction"),
        sa.CheckConstraint("outcome IN ('win', 'loss', 'voided', 'pending')", name="chk_bet_outcome"),
        sa.CheckConstraint("status IN ('pending', 'resolved', 'voided', 'failed')", name="chk_bet_status"),
    )
    op.create_index("idx_bets_strategy_mode", "bets", ["strategy_id", "mode"])
    op.create_index("idx_bets_outcome_pending", "bets", ["outcome"],
                    postgresql_where=sa.text("outcome = 'pending'"))
    op.create_index("idx_bets_placed_at", "bets", ["placed_at"])

    op.create_table(
        "cycle_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("mode", sa.String(10)),
        sa.Column("markets_scanned", sa.Integer(), server_default="0"),
        sa.Column("signals_found", sa.Integer(), server_default="0"),
        sa.Column("bets_placed", sa.Integer(), server_default="0"),
        sa.Column("llm_tokens_used", sa.Integer(), server_default="0"),
        sa.Column("llm_cost_usd", sa.Numeric(8, 5), server_default="0"),
        sa.Column("errors", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_cycle_logs_started_at", "cycle_logs", ["started_at"])

    op.create_table(
        "daily_stats",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("stat_date", sa.Date(), unique=True, nullable=False),
        sa.Column("total_signals", sa.Integer(), server_default="0"),
        sa.Column("total_bets", sa.Integer(), server_default="0"),
        sa.Column("wins", sa.Integer(), server_default="0"),
        sa.Column("losses", sa.Integer(), server_default="0"),
        sa.Column("win_rate", sa.Numeric(5, 4)),
        sa.Column("pnl_usd", sa.Numeric(10, 2), server_default="0"),
        sa.Column("llm_cost_usd", sa.Numeric(8, 5), server_default="0"),
        sa.Column("cycle_count", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Seed strategies
    op.execute("""
    INSERT INTO strategies (code, name, category, description, params) VALUES
    ('S1', 'Tail Hunter', 'weather',
     'Market overprices unlikely outcomes',
     '{"max_yes_price": 0.15, "min_no_price": 0.85}'),
    ('S2', 'Model Update Sniper', 'weather',
     'Market lags new GFS run by 30+ minutes',
     '{"entry_window_minutes": 30, "gfs_update_hours": [4, 10, 16, 22]}'),
    ('S3', 'Niche Geo', 'weather',
     'Fewer competing bots on unpopular cities',
     '{"excluded_cities": ["New York", "London", "Paris"]}'),
    ('S4', 'Consensus Divergence', 'weather',
     'High agreement across GFS + ECMWF models signals edge',
     '{"sources": ["gfs", "ecmwf"], "min_source_agreement": 2}'),
    ('S5', 'Time Decay', 'weather',
     'Short-horizon forecast is more accurate; market hasnt caught up',
     '{"min_hours_to_close": 6, "max_hours_to_close": 24}'),
    ('S6', 'Contrarian Hype', 'weather',
     'Extreme events are overpriced by public sentiment',
     '{"extreme_event_threshold": 0.8, "contrarian_edge": 0.12}'),
    ('S7', 'Whale Following', 'whale',
     'Copy smart-money wallets with delay',
     '{"min_whale_size_usd": 500, "entry_delay_seconds": 180, "position_pct_of_whale": 0.02, "watchlist": []}')
    """)


def downgrade() -> None:
    op.drop_table("daily_stats")
    op.drop_table("cycle_logs")
    op.drop_table("bets")
    op.drop_table("signals")
    op.drop_table("markets")
    op.drop_table("strategies")
