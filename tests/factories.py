"""Test data factories."""
import factory
from datetime import date, datetime, timezone
from db.models import Market, Strategy, Signal, Bet


class StrategyFactory(factory.Factory):
    class Meta:
        model = Strategy

    id = factory.Sequence(lambda n: n + 1)
    code = factory.Sequence(lambda n: f"S{n + 1}")
    name = factory.Faker("sentence", nb_words=3)
    category = "weather"
    is_active = True
    params = factory.LazyAttribute(lambda _: {"min_edge": 0.08})
    description = "Test strategy"


class MarketFactory(factory.Factory):
    class Meta:
        model = Market

    id = factory.Sequence(lambda n: n + 1)
    polymarket_id = factory.Sequence(lambda n: f"market_{n}")
    question = "Will the high temperature in Dallas exceed 95°F on June 5, 2026?"
    category = "weather"
    city = "Dallas"
    weather_station = "KDAL"
    target_date = date(2026, 6, 5)
    resolved = False
    parse_failed = False


class SignalFactory(factory.Factory):
    class Meta:
        model = Signal

    id = factory.Sequence(lambda n: n + 1)
    market_id = 1
    strategy_id = 1
    our_probability = 0.75
    market_price = 0.60
    edge = 0.15
    direction = "YES"
    confidence = "high"
    reasoning = "GFS ensemble strongly above threshold"
    source = "live_cycle"


class BetFactory(factory.Factory):
    class Meta:
        model = Bet

    id = factory.Sequence(lambda n: n + 1)
    market_id = 1
    strategy_id = 1
    mode = "paper"
    direction = "YES"
    amount_usd = 5.0
    entry_price = 0.60
    outcome = "pending"
    status = "pending"
