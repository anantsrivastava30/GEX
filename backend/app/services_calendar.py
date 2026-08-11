"""Native earnings and US economic-release calendar providers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from backend.app.cache import cache
from backend.app.config import get_settings
from quant_analysis.config import CONFIG

TTL_CALENDAR = 6 * 3600
_HTTP_TIMEOUT = 20
_MARKET_TZ = ZoneInfo("America/New_York")
_NASDAQ_EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings"
_NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GEX-Terminal/0.2)",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}
_CALENDAR_CONFIG = CONFIG.get("calendar", {}) if isinstance(CONFIG, dict) else {}
_DEFAULT_EARNINGS_TICKERS = _CALENDAR_CONFIG.get("earnings_tickers", [])
_FRED_RELEASE_IDS = {
    int(release_id) for release_id in _CALENDAR_CONFIG.get("fred_release_ids", [])
}
_FRED_RELEASE_NAMES = {
    9: "Advance Monthly Sales for Retail and Food Services",
    10: "Consumer Price Index",
    11: "Employment Cost Index",
    13: "Industrial Production and Capacity Utilization",
    27: "New Residential Construction",
    46: "Producer Price Index",
    50: "Employment Situation",
    51: "U.S. International Trade in Goods and Services",
    53: "Gross Domestic Product",
    54: "Personal Income and Outlays",
    97: "New Residential Sales",
    101: "FOMC Press Release",
    180: "Unemployment Insurance Weekly Claims",
    192: "Job Openings and Labor Turnover Survey",
    291: "Existing Home Sales",
}
# FRED lists these releases on every calendar day rather than on their actual
# announcement dates, so they cannot be rendered as calendar rows without
# burying the real releases. FOMC decision dates are not available from FRED.
_DENSE_RELEASE_IDS = {101}

# Headline FRED series behind each curated release, so a calendar row carries a
# number instead of just a date. `mode` is "vintage" for series whose ALFRED
# vintages let us pin the value a given release published (output_type=4 returns
# each period's first-published value, which is what the market traded), and
# "latest" for series with no vintage history, where we can only show the most
# recent reading. FRED publishes no consensus figure, so "favorable" only says
# which direction of a move is generally read as good; there is no beat/miss.
_FRED_LOOKBACK_DAYS = 900
_MAX_RELEASE_SERIES_WORKERS = 6
_RELEASE_SERIES: Dict[int, List[Dict[str, Any]]] = {
    9: [{"id": "RSAFS", "label": "Advance retail sales", "units": "$M", "favorable": "up"}],
    10: [
        {"id": "CPIAUCSL", "label": "CPI", "units": "index", "favorable": "down", "yoy": True},
        {"id": "CPILFESL", "label": "Core CPI", "units": "index", "favorable": "down", "yoy": True},
    ],
    11: [{"id": "ECIALLCIV", "label": "Employment cost index", "units": "index"}],
    13: [
        {"id": "INDPRO", "label": "Industrial production", "units": "index", "favorable": "up"},
        {"id": "TCU", "label": "Capacity utilization", "units": "%", "favorable": "up"},
    ],
    27: [
        {"id": "HOUST", "label": "Housing starts", "units": "K SAAR", "favorable": "up"},
        {"id": "PERMIT", "label": "Building permits", "units": "K SAAR", "favorable": "up"},
    ],
    46: [
        {"id": "PPIFIS", "label": "PPI final demand", "units": "index", "favorable": "down", "yoy": True},
    ],
    50: [
        {"id": "PAYEMS", "label": "Nonfarm payrolls", "units": "K jobs", "favorable": "up"},
        {"id": "UNRATE", "label": "Unemployment rate", "units": "%", "favorable": "down"},
    ],
    51: [{"id": "BOPGSTB", "label": "Trade balance", "units": "$M", "favorable": "up"}],
    53: [
        {"id": "A191RL1Q225SBEA", "label": "Real GDP growth", "units": "% annualized", "favorable": "up"},
    ],
    54: [
        {"id": "PCEPI", "label": "PCE price index", "units": "index", "favorable": "down", "yoy": True},
        {"id": "PCEPILFE", "label": "Core PCE price index", "units": "index", "favorable": "down", "yoy": True},
        {"id": "PI", "label": "Personal income", "units": "$B", "favorable": "up"},
    ],
    97: [{"id": "HSN1F", "label": "New home sales", "units": "K SAAR", "favorable": "up"}],
    101: [
        {
            "id": "DFEDTARU",
            "label": "Fed funds target (upper)",
            "units": "%",
            "mode": "latest",
        },
    ],
    180: [
        {"id": "ICSA", "label": "Initial jobless claims", "units": "claims", "favorable": "down"},
        {"id": "CCSA", "label": "Continuing claims", "units": "claims", "favorable": "down"},
    ],
    192: [{"id": "JTSJOL", "label": "Job openings", "units": "K", "favorable": "up"}],
    291: [
        {
            "id": "EXHOSLUSM495S",
            "label": "Existing home sales",
            "units": "K SAAR",
            "mode": "latest",
            "favorable": "up",
        },
    ],
}


def _number(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _normalise_symbols(symbols: Optional[List[str]]) -> List[str]:
    requested = symbols or []
    return list(
        dict.fromkeys(
            symbol.strip().upper() for symbol in requested if symbol and symbol.strip()
        )
    )[:25]


def _earnings_for_symbol(
    symbol: str, start_date: date, end_date: date
) -> List[Dict[str, Any]]:
    ticker = yf.Ticker(symbol)
    try:
        frame = ticker.get_earnings_dates(limit=12)
    except Exception:
        frame = None

    rows: List[Dict[str, Any]] = []
    seen_dates = set()
    if frame is not None and not frame.empty:
        for raw_timestamp, row in frame.iterrows():
            timestamp = pd.Timestamp(raw_timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(_MARKET_TZ)
            else:
                timestamp = timestamp.tz_convert(_MARKET_TZ)
            if not start_date <= timestamp.date() <= end_date:
                continue
            seen_dates.add(timestamp.date())
            rows.append(
                {
                    "symbol": symbol,
                    "earnings_at": timestamp.isoformat(),
                    "eps_estimate": _number(row.get("EPS Estimate")),
                    "reported_eps": _number(row.get("Reported EPS")),
                    "surprise_pct": _number(row.get("Surprise(%)")),
                    "url": f"https://finance.yahoo.com/quote/{symbol}",
                }
            )

    # Yahoo's detailed earnings history can lag while the quote calendar still
    # has the current estimated next date. Keep it date-only because that source
    # does not guarantee a before/after-market timestamp.
    try:
        calendar = ticker.calendar or {}
    except Exception:
        calendar = {}
    raw_dates = calendar.get("Earnings Date", [])
    if not isinstance(raw_dates, (list, tuple)):
        raw_dates = [raw_dates]
    for raw_date in raw_dates:
        try:
            earnings_date = pd.Timestamp(raw_date).date()
        except (TypeError, ValueError):
            continue
        if (
            earnings_date in seen_dates
            or not start_date <= earnings_date <= end_date
        ):
            continue
        rows.append(
            {
                "symbol": symbol,
                "earnings_at": earnings_date.isoformat(),
                "eps_estimate": _number(calendar.get("Earnings Average")),
                "reported_eps": None,
                "surprise_pct": None,
                "url": f"https://finance.yahoo.com/quote/{symbol}",
            }
        )
    return rows


def _nasdaq_number(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text.upper() in {"N/A", "NA", "--"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("$", "").replace(",", "").replace("%", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def _nasdaq_earnings_for_date(earnings_date: date) -> List[Dict[str, Any]]:
    response = requests.get(
        _NASDAQ_EARNINGS_URL,
        params={"date": earnings_date.isoformat()},
        headers=_NASDAQ_HEADERS,
        timeout=_HTTP_TIMEOUT,
    )
    response.raise_for_status()
    records = ((response.json().get("data") or {}).get("rows") or [])
    rows = []
    for record in records:
        symbol = str(record.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        raw_session = str(record.get("time") or "").strip().lower()
        session = {
            "time-pre-market": "pre-market",
            "time-after-hours": "after-hours",
            "time-not-supplied": "time TBD",
        }.get(raw_session, raw_session.replace("time-", " ").strip() or "time TBD")
        estimate_count = _nasdaq_number(record.get("noOfEsts"))
        rows.append(
            {
                "symbol": symbol,
                "company_name": str(record.get("name") or "").strip() or None,
                "earnings_at": earnings_date.isoformat(),
                "session": session,
                "eps_estimate": _nasdaq_number(record.get("epsForecast")),
                "reported_eps": _nasdaq_number(record.get("eps")),
                "surprise_pct": _nasdaq_number(record.get("surprise")),
                "market_cap": _nasdaq_number(record.get("marketCap")),
                "fiscal_quarter": str(record.get("fiscalQuarterEnding") or "").strip()
                or None,
                "estimate_count": int(estimate_count)
                if estimate_count is not None
                else None,
                "url": (
                    "https://www.nasdaq.com/market-activity/stocks/"
                    f"{symbol.lower()}/earnings"
                ),
            }
        )
    return rows


def _load_market_earnings(
    start_date: date, end_date: date, symbols: List[str]
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    days = [
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    ]
    rows: List[Dict[str, Any]] = []
    unavailable: List[str] = []
    successful = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_nasdaq_earnings_for_date, day): day for day in days
        }
        for future in as_completed(futures):
            day = futures[future]
            try:
                rows.extend(future.result())
                successful += 1
            except Exception:
                unavailable.append(day.isoformat())

    if symbols:
        selected = set(symbols)
        rows = [row for row in rows if row["symbol"] in selected]
    rows.sort(
        key=lambda row: (
            row["earnings_at"],
            -(row.get("market_cap") or 0),
            row["symbol"],
        )
    )
    return rows, {
        "configured": True,
        "available": successful > 0,
        "partial": bool(unavailable) and successful > 0,
        "unavailable": unavailable,
        "message": (
            "Some Nasdaq earnings dates were unavailable."
            if unavailable and successful
            else "The market-wide Nasdaq earnings calendar is unavailable."
            if not successful
            else None
        ),
    }


def _load_yahoo_earnings(
    start_date: date, end_date: date, symbols: List[str]
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    unavailable: List[str] = []
    successful = 0
    for symbol in symbols:
        try:
            rows.extend(_earnings_for_symbol(symbol, start_date, end_date))
            successful += 1
        except Exception:
            unavailable.append(symbol)

    rows.sort(key=lambda row: (row["earnings_at"], row["symbol"]))
    return rows, {
        "configured": bool(symbols),
        "available": successful > 0,
        "partial": bool(unavailable) and successful > 0,
        "unavailable": unavailable,
        "message": (
            "Some ticker calendars were unavailable."
            if unavailable and successful
            else "Yahoo earnings calendars are unavailable."
            if unavailable
            else None
        ),
    }


def _load_fred_releases(
    start_date: date, end_date: date, today: date
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    settings = get_settings()
    if not settings.fred_api_key:
        return [], {
            "configured": False,
            "available": False,
            "partial": False,
            "unavailable": [],
            "message": "FRED_API_KEY is not configured.",
        }

    def fetch_release(release_id: int) -> List[Dict[str, Any]]:
        response = requests.get(
            f"{settings.fred_api_url.rstrip('/')}/release/dates",
            params={
                "release_id": release_id,
                "api_key": settings.fred_api_key,
                "file_type": "json",
                "include_release_dates_with_no_data": "true",
                "sort_order": "desc",
                # FRED publishes each release's schedule to the end of the
                # calendar year, so the newest 20 dates are months in the
                # future and a window near today matches nothing. 400 spans
                # from FRED's furthest scheduled date back well past a year
                # even for the daily and weekly releases.
                "limit": 400,
            },
            timeout=_HTTP_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("release_dates", [])

    records: List[Dict[str, Any]] = []
    unavailable: List[str] = []
    release_ids = sorted(
        set(_FRED_RELEASE_IDS or _FRED_RELEASE_NAMES) - _DENSE_RELEASE_IDS
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_release, release_id): release_id
            for release_id in release_ids
        }
        for future in as_completed(futures):
            release_id = futures[future]
            try:
                records.extend(future.result())
            except Exception:
                unavailable.append(str(release_id))

    if not records:
        return [], {
            "configured": True,
            "available": False,
            "partial": False,
            "unavailable": ["fred"],
            "message": "FRED release dates are temporarily unavailable.",
        }

    rows = []
    seen = set()
    for record in records:
        try:
            release_id = int(record.get("release_id"))
            release_date = date.fromisoformat(str(record.get("date")))
        except (TypeError, ValueError):
            continue
        if release_id in _DENSE_RELEASE_IDS:
            continue
        if _FRED_RELEASE_IDS and release_id not in _FRED_RELEASE_IDS:
            continue
        if not start_date <= release_date <= end_date:
            continue
        key = (release_id, release_date)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "release_id": release_id,
                "release_name": _FRED_RELEASE_NAMES.get(
                    release_id, f"FRED release {release_id}"
                ),
                "release_date": release_date.isoformat(),
                "url": f"https://fred.stlouisfed.org/release?rid={release_id}",
                "status": "released" if release_date < today else "scheduled",
                "series": [],
            }
        )
    rows.sort(key=lambda row: (row["release_date"], row["release_name"]))
    return rows, {
        "configured": True,
        "available": True,
        "partial": bool(unavailable),
        "unavailable": unavailable,
        "message": (
            "Some curated FRED release calendars were unavailable."
            if unavailable
            else None
        ),
    }


def _fred_value(raw: Any) -> Optional[float]:
    # FRED writes "." for a missing observation.
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fred_observations(spec: Dict[str, Any], today: date) -> List[Dict[str, Any]]:
    """Ascending [{period, value, published}] for one headline series."""

    settings = get_settings()
    window_start = (today - timedelta(days=_FRED_LOOKBACK_DAYS)).isoformat()
    params: Dict[str, Any] = {
        "series_id": spec["id"],
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "observation_start": window_start,
        "sort_order": "asc",
    }
    if spec.get("mode", "vintage") == "vintage":
        # output_type=4 is each period's first published value, stamped with the
        # date it was published - the join key back to the release calendar.
        params["output_type"] = 4
        params["realtime_start"] = window_start
        params["realtime_end"] = today.isoformat()

    response = requests.get(
        f"{settings.fred_api_url.rstrip('/')}/series/observations",
        params=params,
        timeout=_HTTP_TIMEOUT,
    )
    response.raise_for_status()
    rows: List[Dict[str, Any]] = []
    for record in response.json().get("observations", []):
        value = _fred_value(record.get("value"))
        if value is None:
            continue
        rows.append(
            {
                "period": str(record.get("date") or ""),
                "value": value,
                "published": str(record.get("realtime_start") or ""),
            }
        )
    return rows


def _series_entry(
    spec: Dict[str, Any], observations: List[Dict[str, Any]], release_date: str
) -> Optional[Dict[str, Any]]:
    if not observations:
        return None

    index: Optional[int] = None
    if spec.get("mode", "vintage") == "vintage":
        for position, record in enumerate(observations):
            if record["published"] == release_date:
                index = position
    matched = index is not None
    if index is None:
        index = len(observations) - 1

    actual = observations[index]["value"]
    prior = observations[index - 1]["value"] if index > 0 else None
    change = None if prior is None else actual - prior
    change_pct = None if not prior else (actual - prior) / abs(prior) * 100.0

    change_yoy_pct = None
    if spec.get("yoy") and index >= 12:
        year_ago = observations[index - 12]["value"]
        if year_ago:
            change_yoy_pct = (actual - year_ago) / abs(year_ago) * 100.0

    return {
        "series_id": spec["id"],
        "label": spec["label"],
        "units": spec["units"],
        "period": observations[index]["period"],
        "actual": actual,
        "prior": prior,
        "change": change,
        "change_pct": change_pct,
        "change_yoy_pct": change_yoy_pct,
        "matched": matched,
        "favorable": spec.get("favorable"),
    }


def _attach_release_series(rows: List[Dict[str, Any]], today: date) -> None:
    """Fill each release row with its headline readings, in place."""

    if not rows or not get_settings().fred_api_key:
        return
    specs: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for spec in _RELEASE_SERIES.get(row["release_id"], []):
            specs.setdefault(spec["id"], spec)
    if not specs:
        return

    history: Dict[str, List[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_RELEASE_SERIES_WORKERS) as executor:
        futures = {
            executor.submit(_fred_observations, spec, today): series_id
            for series_id, spec in specs.items()
        }
        for future in as_completed(futures):
            series_id = futures[future]
            try:
                history[series_id] = future.result()
            except Exception:
                history[series_id] = []

    for row in rows:
        entries = []
        for spec in _RELEASE_SERIES.get(row["release_id"], []):
            entry = _series_entry(
                spec, history.get(spec["id"], []), row["release_date"]
            )
            if entry:
                entries.append(entry)
        row["series"] = entries


def get_calendar(
    start_date: date, end_date: date, symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Return independently degradable earnings and economic calendars."""

    selected = _normalise_symbols(symbols)
    # Released-vs-scheduled is relative to today, so today is part of the key.
    key = (
        f"calendar:{datetime.now(_MARKET_TZ).date()}"
        f":{start_date}:{end_date}:{','.join(selected)}"
    )

    def load() -> Dict[str, Any]:
        today = datetime.now(_MARKET_TZ).date()
        earnings, nasdaq_status = _load_market_earnings(
            start_date, end_date, selected
        )
        sources = {"nasdaq": nasdaq_status}
        if not nasdaq_status["available"]:
            fallback_symbols = selected or list(_DEFAULT_EARNINGS_TICKERS)
            earnings, yahoo_status = _load_yahoo_earnings(
                start_date, end_date, fallback_symbols
            )
            sources["yfinance"] = yahoo_status
        releases, fred_status = _load_fred_releases(start_date, end_date, today)
        _attach_release_series(releases, today)
        sources["fred"] = fred_status
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "earnings": earnings,
            "economic_releases": releases,
            "sources": sources,
        }

    return cache.get_or_compute(key, TTL_CALENDAR, load)
