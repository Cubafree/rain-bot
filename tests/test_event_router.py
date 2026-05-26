"""Tests for market question parsing."""
import pytest
from datetime import date

from services.event_router import parse_market, ParsedMarket, FORECAST_COORDS, STATION_COORDS


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


def test_hong_kong_uses_city_center_coords():
    # VHHH airport is on Lantau island; HKO observatory (city center) is the
    # official reference used for "Hong Kong temperature" resolution.
    q = "Will the highest temperature in Hong Kong be 33°C or higher on May 25?"
    result = parse_market(q)

    assert result.station == "VHHH"
    city_lat, city_lon = FORECAST_COORDS["VHHH"]
    assert result.latitude == city_lat
    assert result.longitude == city_lon
    # Must differ from raw airport coords
    airport_lat, airport_lon = STATION_COORDS["VHHH"]
    assert result.longitude != airport_lon


def test_tokyo_uses_city_center_coords():
    q = "Will the temperature in Tokyo exceed 35°C on August 10, 2026?"
    result = parse_market(q)

    assert result.station == "RJTT"
    city_lat, city_lon = FORECAST_COORDS["RJTT"]
    assert result.latitude == city_lat
    assert result.longitude == city_lon


def test_dallas_uses_airport_coords():
    # US cities: NWS officially reports from the airport ASOS station
    q = "Will the high temperature in Dallas exceed 100°F on July 15, 2026?"
    result = parse_market(q)

    assert result.station == "KDAL"
    airport_lat, airport_lon = STATION_COORDS["KDAL"]
    assert result.latitude == airport_lat
    assert result.longitude == airport_lon
