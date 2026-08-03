"""Persisted daily-close history and intraday price/SMA crossing signals."""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, Set, Tuple
from zoneinfo import ZoneInfo

from backend.app import storage
from backend.app.deps import get_tradier, tradier_call
from backend.app.services_watchlists import snapshot_symbols
from quant_analysis.analytics.technicals import price_sma_cross
from quant_analysis.storage.snapshots import is_market_session, previous_market_session

logger = logging.getLogger(__name__)
MARKET_TZ = ZoneInfo("America/New_York")
_HISTORY_CALENDAR_DAYS = 450


def _history_target(day: date) -> date:
    return previous_market_session(day)


def _normalise_history(rows: Iterable[Dict[str, Any]], through: date) -> list[dict]:
    by_date: dict[str, dict] = {}
    for row in rows:
        try:
            session_date = date.fromisoformat(str(row["date"]))
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if session_date > through or not math.isfinite(close) or close <= 0:
            continue
        by_date[session_date.isoformat()] = {
            "date": session_date.isoformat(),
            "close": close,
        }
    return [by_date[key] for key in sorted(by_date)]


def refresh_daily_price_history(now: datetime | None = None) -> dict:
    """Backfill prior-session closes once per scheduled symbol and market day."""

    now = now or datetime.now(tz=MARKET_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TZ)
    else:
        now = now.astimezone(MARKET_TZ)
    if not is_market_session(now.date()):
        return {"target": None, "refreshed": [], "current": [], "failed": []}

    target = _history_target(now.date())
    api = get_tradier()
    refreshed: list[str] = []
    current: list[str] = []
    failed: list[str] = []
    for symbol in snapshot_symbols():
        if storage.latest_daily_price_date(symbol) == target.isoformat():
            current.append(symbol)
            continue
        try:
            rows = tradier_call(
                api.history,
                symbol,
                "daily",
                (target - timedelta(days=_HISTORY_CALENDAR_DAYS)).isoformat(),
                target.isoformat(),
            )
            normalised = _normalise_history(rows or [], target)
            if not normalised or normalised[-1]["date"] != target.isoformat():
                failed.append(symbol)
                continue
            storage.upsert_daily_price_bars(symbol, normalised)
            refreshed.append(symbol)
        except Exception:
            logger.exception("Daily price-history refresh failed for %s", symbol)
            failed.append(symbol)
    logger.info(
        "Daily price history: %d refreshed, %d current, %d failed for %s",
        len(refreshed),
        len(current),
        len(failed),
        target,
    )
    return {
        "target": target.isoformat(),
        "refreshed": refreshed,
        "current": current,
        "failed": failed,
    }


def get_sma_metrics(
    symbol: str, spot: float, now: datetime | None = None
) -> Tuple[Dict[str, Any], Set[int]]:
    """Compute current price/SMA state from completed closes and snapshot spot."""

    now = now or datetime.now(tz=MARKET_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TZ)
    else:
        now = now.astimezone(MARKET_TZ)
    target = _history_target(now.date())
    try:
        current_price = float(spot)
    except (TypeError, ValueError):
        return {}, set()
    rows = storage.load_daily_closes(symbol, limit=200)
    if not rows or rows[-1]["date"] != target.isoformat():
        return {}, set()

    closes = [float(row["close"]) for row in rows]
    metrics: Dict[str, Any] = {"sma_history_sessions": len(closes)}
    ready: Set[int] = set()
    for window in (50, 200):
        result = price_sma_cross(closes, current_price, window)
        if result is None:
            continue
        ready.add(window)
        metrics.update(
            {
                f"sma_{window}": result["sma"],
                f"price_vs_sma_{window}_pct": result["distance_pct"],
                f"sma_{window}_cross": result["cross"],
            }
        )
    return metrics, ready
