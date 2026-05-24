from datetime import datetime, date
from sqlalchemy import (
    Boolean, Column, Date, DateTime, ForeignKey, Integer,
    Numeric, String, Text, UniqueConstraint, CheckConstraint, Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    code = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(50))
    is_active = Column(Boolean, default=True, nullable=False)
    params = Column(JSONB)
    prompt_extension = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    signals = relationship("Signal", back_populates="strategy")
    bets = relationship("Bet", back_populates="strategy")


class Market(Base):
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True)
    polymarket_id = Column(String(100), unique=True, nullable=False)
    question = Column(Text, nullable=False)
    category = Column(String(50))
    city = Column(String(100))
    weather_station = Column(String(10))
    target_date = Column(Date)
    close_time = Column(DateTime(timezone=True))
    resolved = Column(Boolean, default=False, nullable=False)
    outcome = Column(String(10))
    parse_failed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("outcome IN ('YES', 'NO')", name="chk_market_outcome"),
        Index("idx_markets_resolved", "resolved", postgresql_where="resolved = false"),
    )

    signals = relationship("Signal", back_populates="market")
    bets = relationship("Bet", back_populates="market")


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    our_probability = Column(Numeric(5, 4))
    market_price = Column(Numeric(5, 4))
    edge = Column(Numeric(5, 4))
    direction = Column(String(10))
    confidence = Column(String(20))
    reasoning = Column(Text)
    forecast_data = Column(JSONB)
    llm_model = Column(String(100))
    tokens_used = Column(Integer)
    source = Column(String(20), default="live_cycle", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("market_id", "strategy_id", name="uq_signal_market_strategy_day"),
        CheckConstraint("direction IN ('YES', 'NO')", name="chk_signal_direction"),
        CheckConstraint("confidence IN ('high', 'medium', 'low')", name="chk_signal_confidence"),
        CheckConstraint("source IN ('live_cycle', 'backtest')", name="chk_signal_source"),
        Index("idx_signals_market_strategy", "market_id", "strategy_id"),
    )

    market = relationship("Market", back_populates="signals")
    strategy = relationship("Strategy", back_populates="signals")
    bets = relationship("Bet", back_populates="signal")


class Bet(Base):
    __tablename__ = "bets"

    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("signals.id"))
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    mode = Column(String(10), nullable=False)
    direction = Column(String(10))
    amount_usd = Column(Numeric(10, 2))
    entry_price = Column(Numeric(5, 4))
    exit_price = Column(Numeric(5, 4))
    outcome = Column(String(10), default="pending")
    status = Column(String(20), default="pending")
    pnl = Column(Numeric(10, 2))
    clob_order_id = Column(String(100))
    placed_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("mode IN ('paper', 'live', 'backtest')", name="chk_bet_mode"),
        CheckConstraint("direction IN ('YES', 'NO')", name="chk_bet_direction"),
        CheckConstraint("outcome IN ('win', 'loss', 'voided', 'pending')", name="chk_bet_outcome"),
        CheckConstraint("status IN ('pending', 'resolved', 'voided', 'failed')", name="chk_bet_status"),
        Index("idx_bets_strategy_mode", "strategy_id", "mode"),
        Index("idx_bets_outcome_pending", "outcome", postgresql_where="outcome = 'pending'"),
        Index("idx_bets_placed_at", "placed_at"),
    )

    signal = relationship("Signal", back_populates="bets")
    market = relationship("Market", back_populates="bets")
    strategy = relationship("Strategy", back_populates="bets")


class CycleLog(Base):
    __tablename__ = "cycle_logs"

    id = Column(Integer, primary_key=True)
    mode = Column(String(10))
    markets_scanned = Column(Integer, default=0)
    signals_found = Column(Integer, default=0)
    bets_placed = Column(Integer, default=0)
    llm_tokens_used = Column(Integer, default=0)
    llm_cost_usd = Column(Numeric(8, 5), default=0)
    errors = Column(JSONB, default=list)
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_cycle_logs_started_at", "started_at"),
    )


class DailyStats(Base):
    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True)
    stat_date = Column(Date, unique=True, nullable=False)
    total_signals = Column(Integer, default=0)
    total_bets = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Numeric(5, 4))
    pnl_usd = Column(Numeric(10, 2), default=0)
    llm_cost_usd = Column(Numeric(8, 5), default=0)
    cycle_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
