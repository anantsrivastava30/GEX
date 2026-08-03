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
    start_date: date, end_date: date
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
                "limit": 20,
            },
            timeout=_HTTP_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("release_dates", [])

    records: List[Dict[str, Any]] = []
    unavailable: List[str] = []
    release_ids = sorted(_FRED_RELEASE_IDS or _FRED_RELEASE_NAMES)
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


def get_calendar(
    start_date: date, end_date: date, symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Return independently degradable earnings and economic calendars."""

    selected = _normalise_symbols(symbols)
    key = f"calendar:{start_date}:{end_date}:{','.join(selected)}"

    def load() -> Dict[str, Any]:
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
        releases, fred_status = _load_fred_releases(start_date, end_date)
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
