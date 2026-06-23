"""Tests for per-strategy eligibility gates."""
from types import SimpleNamespace

from scheduler import _strategy_eligible


def _strat(code, **params):
    return SimpleNamespace(code=code, params=params)


def _ctx(yes=0.5, city="Dallas", hours=20.0, spread=1.0):
    gm = SimpleNamespace(yes_price=yes, no_price=round(1 - yes, 3))
    market = SimpleNamespace(city=city)
    forecast = SimpleNamespace(hours_ahead=hours, model_spread_f=spread)
    return gm, market, forecast


def test_s1_tail_hunter_only_extremes():
    s1 = _strat("S1", max_yes_price=0.15)
    assert _strategy_eligible(s1, *_ctx(yes=0.08))      # cheap longshot
    assert _strategy_eligible(s1, *_ctx(yes=0.93))      # expensive favorite tail
    assert not _strategy_eligible(s1, *_ctx(yes=0.50))  # mid-priced → excluded


def test_s2_short_lead_only():
    s2 = _strat("S2")
    assert _strategy_eligible(s2, *_ctx(hours=24))
    assert not _strategy_eligible(s2, *_ctx(hours=60))


def test_s3_excludes_major_cities():
    s3 = _strat("S3", excluded_cities=["New York", "London", "Paris"])
    assert _strategy_eligible(s3, *_ctx(city="Dallas"))
    assert not _strategy_eligible(s3, *_ctx(city="london"))  # case-insensitive


def test_s4_requires_model_agreement():
    s4 = _strat("S4", max_spread_f=2.0)
    assert _strategy_eligible(s4, *_ctx(spread=1.2))
    assert not _strategy_eligible(s4, *_ctx(spread=4.0))
    assert not _strategy_eligible(s4, *_ctx(spread=None))


def test_s5_time_decay_window():
    s5 = _strat("S5", min_hours_to_close=6, max_hours_to_close=24)
    assert _strategy_eligible(s5, *_ctx(hours=12))
    assert not _strategy_eligible(s5, *_ctx(hours=4))
    assert not _strategy_eligible(s5, *_ctx(hours=40))


def test_s7_needs_watchlist():
    assert not _strategy_eligible(_strat("S7", watchlist=[]), *_ctx())
    assert _strategy_eligible(_strat("S7", watchlist=["0xabc"]), *_ctx())


def test_unknown_strategy_eligible_by_default():
    assert _strategy_eligible(_strat("S9"), *_ctx())
