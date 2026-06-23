"""Tests for the per-strategy (city, date) correlated-exposure cap."""
from datetime import date
from types import SimpleNamespace

import pytest

from scheduler import _city_day_exposure_full


class _FakeResult:
    def __init__(self, count):
        self._count = count

    def scalar_one(self):
        return self._count


class _FakeDB:
    """Records whether execute() was called and returns a preset pending count."""
    def __init__(self, count):
        self._count = count
        self.executed = False

    async def execute(self, _stmt):
        self.executed = True
        return _FakeResult(self._count)


def _market(city="Seoul", station="RKSS", target=date(2026, 6, 20)):
    return SimpleNamespace(city=city, weather_station=station, target_date=target)


def _strategy(id=1):
    return SimpleNamespace(id=id, code="S1")


@pytest.mark.asyncio
async def test_blocks_when_at_cap():
    db = _FakeDB(count=1)  # default cap is 1
    assert await _city_day_exposure_full(db, _market(), _strategy()) is True


@pytest.mark.asyncio
async def test_allows_when_below_cap():
    db = _FakeDB(count=0)
    assert await _city_day_exposure_full(db, _market(), _strategy()) is False


@pytest.mark.asyncio
async def test_skips_query_when_no_grouping_key():
    # No city AND no station → cannot group → must not block and must not query.
    db = _FakeDB(count=5)
    result = await _city_day_exposure_full(db, _market(city=None, station=None), _strategy())
    assert result is False
    assert db.executed is False


@pytest.mark.asyncio
async def test_skips_query_when_no_target_date():
    db = _FakeDB(count=5)
    result = await _city_day_exposure_full(db, _market(target=None), _strategy())
    assert result is False
    assert db.executed is False


@pytest.mark.asyncio
async def test_falls_back_to_station_when_city_missing():
    db = _FakeDB(count=1)
    # city None but station present → still groups and blocks at cap
    assert await _city_day_exposure_full(db, _market(city=None), _strategy()) is True
    assert db.executed is True
