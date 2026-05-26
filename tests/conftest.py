"""Shared pytest fixtures."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from services import weather


@pytest.fixture(autouse=True)
def clear_weather_cache():
    weather._cache.clear()
    yield
    weather._cache.clear()


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute = AsyncMock()
    db.flush = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db
