"""Parse free-text Polymarket questions into structured market data."""
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from loguru import logger

CITY_STATIONS: dict[str, str] = {
    "New York": "KLGA",
    "Dallas": "KDAL",
    "Chicago": "KMDW",
    "Los Angeles": "KLAX",
    "Miami": "KMIA",
    "Atlanta": "KATL",
    "London": "EGLL",
    "Paris": "LFPG",
    "Seoul": "RKSS",
    "Tokyo": "RJTT",
    "Sydney": "YSSY",
    "Sao Paulo": "SBGR",
    "Hong Kong": "VHHH",
    "Singapore": "WSSS",
    "Dubai": "OMDB",
    "Houston": "KHOU",
    "Phoenix": "KPHX",
    "Las Vegas": "KLAS",
    "Seattle": "KSEA",
    "Denver": "KDEN",
    "Boston": "KBOS",
    "Minneapolis": "KMSP",
    "Detroit": "KDTW",
    # Polymarket-active cities
    "Kuala Lumpur": "WMKK",
    "Wellington": "NZWN",
    "Bangkok": "VTBS",
    "Jakarta": "WIII",
    "Manila": "RPLL",
    "Mumbai": "VABB",
    "Karachi": "OPKC",
    "Cairo": "HECA",
    "Lagos": "DNMM",
    "Nairobi": "HKNA",
    "Cape Town": "FACT",
    "Buenos Aires": "SAEZ",
    "Mexico City": "MMMX",
    "Toronto": "CYYZ",
    "Vancouver": "CYVR",
    "Amsterdam": "EHAM",
    "Berlin": "EDDB",
    "Madrid": "LEMD",
    "Rome": "LIRF",
    "Istanbul": "LTBA",
    # Polymarket name variants and missing cities
    "New York City": "KLGA",
    "NYC": "KLGA",
    "Munich": "EDDM",
    "Panama City": "MPMM",
    # China (active on Polymarket)
    "Zhengzhou": "ZHCC",
    "Beijing": "ZBAA",
    "Shanghai": "ZSPD",
    "Guangzhou": "ZGGG",
    "Chengdu": "ZUUU",
    "Wuhan": "ZHHH",
    "Shenzhen": "ZGSZ",
    "Xian": "ZLXY",
    "Hangzhou": "ZSHC",
    "Nanjing": "ZSNJ",
    "Chongqing": "ZUCK",
    "Kunming": "ZPPP",
    "Tianjin": "ZBTJ",
    "Harbin": "ZYHB",
    "Jinan": "ZSJN",
    "Qingdao": "ZSQD",
    "Shenyang": "ZYYY",
    "Changsha": "ZGHA",
    "Fuzhou": "ZSFZ",
    "Xiamen": "ZSAM",
    "Hefei": "ZSOF",
    "Nanchang": "ZSCN",
    "Urumqi": "ZWWW",
    # Other Asia
    "Taipei": "RCTP",
    "Osaka": "RJBB",
    # Europe
    "Vienna": "LOWW",
    "Warsaw": "EPWA",
    "Stockholm": "ESSA",
    "Zurich": "LSZH",
    # Middle East / Africa
    "Riyadh": "OERK",
    "Johannesburg": "FAOR",
}

# IATA → (latitude, longitude)
STATION_COORDS: dict[str, tuple[float, float]] = {
    "KLGA": (40.777, -73.873),
    "KDAL": (32.847, -96.852),
    "KMDW": (41.786, -87.752),
    "KLAX": (33.943, -118.408),
    "KMIA": (25.796, -80.287),
    "KATL": (33.641, -84.427),
    "EGLL": (51.477, -0.461),
    "LFPG": (49.013, 2.550),
    "RKSS": (37.558, 126.791),
    "RJTT": (35.552, 139.780),
    "YSSY": (-33.947, 151.177),
    "SBGR": (-23.432, -46.469),
    "VHHH": (22.308, 113.915),
    "WSSS": (1.359, 103.989),
    "OMDB": (25.253, 55.364),
    "KHOU": (29.645, -95.279),
    "KPHX": (33.438, -112.008),
    "KLAS": (36.080, -115.152),
    "KSEA": (47.449, -122.309),
    "KDEN": (39.856, -104.674),
    "KBOS": (42.365, -71.009),
    "KMSP": (44.881, -93.222),
    "KDTW": (42.212, -83.353),
    # Polymarket-active cities
    "WMKK": (2.746, 101.710),
    "NZWN": (-41.327, 174.805),
    "VTBS": (13.681, 100.747),
    "WIII": (-6.125, 106.656),
    "RPLL": (14.508, 121.020),
    "VABB": (19.089, 72.868),
    "OPKC": (24.906, 67.161),
    "HECA": (30.122, 31.406),
    "DNMM": (6.577, 3.321),
    "HKNA": (-1.319, 36.928),
    "FACT": (-33.965, 18.602),
    "SAEZ": (-34.822, -58.536),
    "MMMX": (19.436, -99.072),
    "CYYZ": (43.677, -79.631),
    "CYVR": (49.195, -123.184),
    "EHAM": (52.310, 4.768),
    "EDDB": (52.366, 13.503),
    "LEMD": (40.472, -3.561),
    "LIRF": (41.800, 12.239),
    "LTBA": (40.976, 28.815),
    "EDDM": (48.354, 11.786),
    "MPMM": (9.071, -79.383),
    # China
    "ZHCC": (34.519, 113.841),
    "ZBAA": (40.080, 116.585),
    "ZSPD": (31.143, 121.805),
    "ZGGG": (23.392, 113.300),
    "ZUUU": (30.578, 103.947),
    "ZHHH": (30.783, 114.208),
    "ZGSZ": (22.639, 113.811),
    "ZLXY": (34.447, 108.752),
    "ZSHC": (30.235, 120.434),
    "ZSNJ": (31.742, 118.862),
    "ZUCK": (29.720, 106.640),
    "ZPPP": (24.992, 102.743),
    "ZBTJ": (39.124, 117.346),
    "ZYHB": (45.623, 126.250),
    "ZSJN": (36.857, 117.216),
    "ZSQD": (36.266, 120.374),
    "ZYYY": (41.639, 123.483),
    "ZGHA": (28.189, 113.219),
    "ZSFZ": (25.935, 119.663),
    "ZSAM": (24.544, 118.127),
    "ZSOF": (31.780, 117.298),
    "ZSCN": (28.865, 115.900),
    "ZWWW": (43.907, 87.474),
    # Other Asia
    "RCTP": (25.077, 121.233),
    "RJBB": (34.786, 135.438),
    # Europe
    "LOWW": (48.110, 16.569),
    "EPWA": (52.166, 20.967),
    "ESSA": (59.652, 17.919),
    "LSZH": (47.458, 8.548),
    # Middle East / Africa
    "OERK": (24.958, 46.699),
    "FAOR": (-26.139, 28.246),
}

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

_DATE_PATTERN = re.compile(
    r"(?:on\s+)?(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:,\s*(?P<year>\d{4}))?",
    re.IGNORECASE,
)
_THRESHOLD_PATTERN = re.compile(
    r"(?P<condition>exceed|above|below|under|reach|at least|no more than|less than|be)\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>°?[FC])?",
    re.IGNORECASE,
)
# handles "24°C or below" / "90°F or above" (value before condition)
_THRESHOLD_PATTERN_REV = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>°?[FC])\s+or\s+(?P<condition>above|below|higher|lower)",
    re.IGNORECASE,
)
_CITY_PATTERN = re.compile(
    r"\bin\s+(?P<city>" + "|".join(re.escape(c) for c in sorted(CITY_STATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


@dataclass
class ParsedMarket:
    city: str | None
    station: str | None
    latitude: float | None
    longitude: float | None
    threshold: float | None
    unit: Literal["F", "C"] | None
    condition: Literal["above", "below"] | None
    target_date: date | None
    parse_confidence: float


def parse_market(question: str) -> ParsedMarket:
    """Extract structured fields from a Polymarket weather question.

    Returns a ParsedMarket; fields are None if extraction failed.
    parse_confidence < 0.8 means the market should be skipped.
    """
    city = _extract_city(question)
    station = CITY_STATIONS.get(city) if city else None
    coords = STATION_COORDS.get(station) if station else None

    threshold, unit, condition = _extract_threshold(question)
    target_date = _extract_date(question)

    fields_found = sum(
        x is not None for x in [city, station, threshold, condition, target_date]
    )
    confidence = fields_found / 5.0

    if city and not station:
        logger.warning("Unknown city in question — add to CITY_STATIONS", question=question, city=city)

    return ParsedMarket(
        city=city,
        station=station,
        latitude=coords[0] if coords else None,
        longitude=coords[1] if coords else None,
        threshold=threshold,
        unit=unit,
        condition=condition,
        target_date=target_date,
        parse_confidence=confidence,
    )


def _extract_city(question: str) -> str | None:
    m = _CITY_PATTERN.search(question)
    if not m:
        return None
    raw = m.group("city")
    for city in CITY_STATIONS:
        if city.lower() == raw.lower():
            return city
    return None


def _extract_threshold(question: str) -> tuple[float | None, str | None, str | None]:
    m = _THRESHOLD_PATTERN.search(question)
    if not m:
        m = _THRESHOLD_PATTERN_REV.search(question)
    if not m:
        return None, None, None

    value = float(m.group("value"))
    raw_unit = (m.group("unit") or "").replace("°", "").upper()
    unit: str | None = raw_unit if raw_unit in ("F", "C") else "C"

    raw_cond = m.group("condition").lower()
    if any(w in raw_cond for w in ("exceed", "above", "reach", "at least", "higher", "be")):
        condition = "above"
    else:
        condition = "below"

    return value, unit, condition


def _extract_date(question: str) -> date | None:
    m = _DATE_PATTERN.search(question)
    if not m:
        return None

    month_str = m.group("month").lower()
    month = _MONTH_MAP.get(month_str)
    if not month:
        return None

    day = int(m.group("day"))
    year = int(m.group("year")) if m.group("year") else date.today().year

    try:
        return date(year, month, day)
    except ValueError:
        return None
