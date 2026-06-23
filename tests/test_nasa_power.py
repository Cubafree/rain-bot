"""Tests for the NASA POWER actuals client with mocked HTTP."""
import httpx
import pytest
import respx
from datetime import date

from services import nasa_power
from services.nasa_power import POWER_DAILY_URL, _c_to_f


def test_c_to_f():
    assert _c_to_f(0) == 32.0
    assert _c_to_f(100) == 212.0


MOCK = {
    "properties": {
        "parameter": {
            "T2M_MAX": {
                "20260601": 23.31,   # 73.96F
                "20260602": 21.41,
                "20260603": -999.0,  # fill value → skipped
            }
        }
    }
}


@pytest.mark.asyncio
@respx.mock
async def test_get_actual_max_range_filters_fill_and_converts():
    respx.get(POWER_DAILY_URL).mock(return_value=httpx.Response(200, json=MOCK))
    out = await nasa_power.get_actual_max_range_f(40.71, -74.0, date(2026, 6, 1), date(2026, 6, 3))
    assert out == {date(2026, 6, 1): 74.0, date(2026, 6, 2): 70.5}
    assert date(2026, 6, 3) not in out  # -999 fill dropped


@pytest.mark.asyncio
@respx.mock
async def test_get_actual_max_single_date():
    respx.get(POWER_DAILY_URL).mock(return_value=httpx.Response(200, json=MOCK))
    val = await nasa_power.get_actual_max_f(40.71, -74.0, date(2026, 6, 2))
    assert val == 70.5


@pytest.mark.asyncio
@respx.mock
async def test_get_actual_max_range_handles_error():
    respx.get(POWER_DAILY_URL).mock(return_value=httpx.Response(500))
    out = await nasa_power.get_actual_max_range_f(40.71, -74.0, date(2026, 6, 1), date(2026, 6, 2))
    assert out == {}
