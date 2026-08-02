"""Native earnings and US economic-release calendar providers."""

from __future__ import annotations

from datetime import date, datetime, timezone
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
_CALENDAR_CONFIG = CONFIG.get("calendar", {}) if isinstance(CONFIG, dict) else {}
_DEFAULT_EARNINGS_TICKERS = _CALENDAR_CONFIG.get("earnings_tickers", [])
_FRED_RELEASE_IDS = {
    int(release_id) for release_id in _CALENDAR_CONFIG.get("fred_release_ids", [])
}


def _number(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _normalise_symbols(symbols: Optional[List[str]]) -> List[str]:
    requested = symbols or _DEFAULT_EARNINGS_TICKERS
    return list(
        dict.fromkeys(
            symbol.strip().upper() for symbol in requested if symbol and symbol.strip()
        )
    )[:25]


def _earnings_for_symbol(
    symbol: str, start_date: date, end_date: date
) -> List[Dict[str, Any]]:
    frame = yf.Ticker(symbol).get_earnings_dates(limit=12)
    if frame is None or frame.empty:
        return []

    rows: List[Dict[str, Any]] = []
    for raw_timestamp, row in frame.iterrows():
        timestamp = pd.Timestamp(raw_timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(_MARKET_TZ)
        else:
            timestamp = timestamp.tz_convert(_MARKET_TZ)
        if not start_date <= timestamp.date() <= end_date:
            continue
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
    return rows


def _load_earnings(
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

    try:
        response = requests.get(
            f"{settings.fred_api_url.rstrip('/')}/releases/dates",
            params={
                "api_key": settings.fred_api_key,
                "file_type": "json",
                "realtime_start": start_date.isoformat(),
                "realtime_end": end_date.isoformat(),
                "include_release_dates_with_no_data": "true",
                "sort_order": "asc",
                "limit": 1000,
            },
            timeout=_HTTP_TIMEOUT,
        )
        response.raise_for_status()
        records = response.json().get("release_dates", [])
    except Exception:
        return [], {
            "configured": True,
            "available": False,
            "partial": False,
            "unavailable": ["fred"],
            "message": "FRED release dates are temporarily unavailable.",
        }

    rows = []
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
        rows.append(
            {
                "release_id": release_id,
                "release_name": str(record.get("release_name") or "FRED release"),
                "release_date": release_date.isoformat(),
                "url": f"https://fred.stlouisfed.org/release?rid={release_id}",
            }
        )
    return rows, {
        "configured": True,
        "available": True,
        "partial": False,
        "unavailable": [],
        "message": None,
    }


def get_calendar(
    start_date: date, end_date: date, symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Return independently degradable earnings and economic calendars."""

    selected = _normalise_symbols(symbols)
    key = f"calendar:{start_date}:{end_date}:{','.join(selected)}"

    def load() -> Dict[str, Any]:
        earnings, yahoo_status = _load_earnings(start_date, end_date, selected)
        releases, fred_status = _load_fred_releases(start_date, end_date)
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "earnings": earnings,
            "economic_releases": releases,
            "sources": {"yfinance": yahoo_status, "fred": fred_status},
        }

    return cache.get_or_compute(key, TTL_CALENDAR, load)
