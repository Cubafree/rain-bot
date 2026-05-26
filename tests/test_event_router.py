"""Tests for market question parsing."""
import pytest
from datetime import date

from services.event_router import parse_market, ParsedMarket


def test_full_weather_question():
    q = "Will the high temperature in Dallas exceed 95°F on June 5, 2026?"
    result = parse_market(q)

    assert result.city == "Dallas"
    assert result.station == "KDAL"
    assert result.threshold == 95.0
    assert result.unit == "F"
    assert result.condition == "above"
    assert result.target_date == date(2026, 6, 5)
    assert result.parse_confidence >= 0.8


def test_below_condition():
    q = "Will the temperature in Chicago fall below 32°F on January 15, 2026?"
    result = parse_market(q)

    assert result.city == "Chicago"
    assert result.station == "KMDW"
    assert result.condition == "below"
    assert result.threshold == 32.0


def test_unknown_city():
    q = "Will it exceed 80°F in Springfield on May 10, 2026?"
    result = parse_market(q)

    assert result.city is None
    assert result.station is None
    assert result.parse_confidence < 0.8


def test_missing_date():
    q = "Will the high temperature in Miami exceed 90°F?"
    result = parse_market(q)

    assert result.city == "Miami"
    assert result.target_date is None
    assert result.parse_confidence < 1.0


def test_london_maps_to_heathrow():
    q = "Will London exceed 25°C on July 20, 2026?"
    result = parse_market(q)

    assert result.station == "EGLL"


def test_parse_confidence_full():
    q = "Will the high temperature in New York exceed 90°F on August 1, 2026?"
    result = parse_market(q)

    assert result.parse_confidence == 1.0
