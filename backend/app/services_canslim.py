"""Per-stock CAN SLIM scan: which stocks are flagged to buy, and why.

Scans the snapshot/watchlist stock universe (index ETFs excluded), scores
each name on the seven letters, gates buys on the market-direction state,
and persists fresh heavy-volume breakouts as ``stock_breakout`` signals in
the shared direction signal feed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from quant_analysis.analytics.canslim import (
    build_stock_narrative,
    describe_breakout_signal,
    detect_breakout,
    evaluate_stock_canslim,
    merged_canslim_config,
    weighted_rs_score,
)
from quant_analysis.analytics.market_direction import STATE_LABELS, classify_market_state
from quant_analysis.config import CONFIG

from backend.app.cache import cache
from backend.app.services_direction import (
    _completed_bars,
    _fetch_bars,
    _market_open_now,
    direction_config,
)

logger = logging.getLogger(__name__)

TTL_FUNDAMENTALS = 24 * 3600
TTL_LEADERS = 900

_GATE_MESSAGES = {
    "confirmed_uptrend": "Confirmed uptrend - breakout buys are allowed under O'Neil's rules.",
    "uptrend_under_pressure": "Uptrend under pressure - distribution is building; size new buys with caution.",
    "rally_attempt": "Rally attempt in progress - wait for a follow-through day before new buys.",
    "correction": "Market in correction - O'Neil's rule is no new buys; build the watchlist instead.",
    "unavailable": "Broad market state unavailable.",
}

_READINESS_ORDER = {
    "buy_candidate": 0,
    "extended": 1,
    "near_pivot": 2,
    "wait_market": 3,
    "not_ready": 4,
    "insufficient_data": 5,
}


def canslim_config() -> Dict[str, Any]:
    raw = CONFIG.get("canslim", {}) or {}
    cfg = merged_canslim_config(raw)
    cfg["exclude"] = [str(s).upper() for s in raw.get("exclude", []) or []]
    cfg["max_symbols"] = int(raw.get("max_symbols", 30))
    cfg["fundamentals_pause_seconds"] = float(
        raw.get("fundamentals_pause_seconds", 0.4)
    )
    return cfg


def _stock_universe(cfg: Dict[str, Any]) -> List[str]:
    from backend.app.services_watchlists import snapshot_symbols

    exclude = set(cfg["exclude"])
    symbols = [s for s in snapshot_symbols() if s.upper() not in exclude]
    return symbols[: cfg["max_symbols"]]


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    def compute() -> Dict[str, Any]:
        from quant_analysis.integrations.fundamentals import fetch_stock_fundamentals

        return fetch_stock_fundamentals(symbol)

    return cache.get_or_compute(
        f"canslim:fundamentals:{symbol}", TTL_FUNDAMENTALS, compute
    )


def _market_state() -> str:
    dcfg = direction_config()
    bars = _fetch_bars(dcfg["benchmark"], dcfg["history_days"])
    if not bars:
        return "unavailable"
    return classify_market_state(bars, dcfg)["state"]


def _leader_item(
    symbol: str,
    bars: List[Dict[str, Any]],
    fundamentals: Dict[str, Any],
    rs_percentile: Optional[float],
    universe_size: int,
    market_state: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    result = evaluate_stock_canslim(
        bars, fundamentals, rs_percentile, universe_size, market_state, cfg
    )
    closes = [float(b["close"]) for b in bars]
    change_pct = (
        (closes[-1] / closes[-2] - 1.0) * 100.0
        if len(closes) > 1 and closes[-2]
        else None
    )
    label = fundamentals.get("name") or symbol
    return {
        "symbol": symbol,
        "name": fundamentals.get("name"),
        "readiness": result["readiness"],
        "readiness_label": result["readiness_label"],
        "score": result["score"],
        "scorecard": result["rows"],
        "rs_percentile": result["rs_percentile"],
        "off_high_pct": result["off_high_pct"],
        "quarterly_eps_growth_pct": _round(fundamentals.get("quarterly_eps_growth_pct")),
        "quarterly_revenue_growth_pct": _round(
            fundamentals.get("quarterly_revenue_growth_pct")
        ),
        "annual_eps_growth_pct": _round(fundamentals.get("annual_eps_growth_pct")),
        "roe_pct": _round(fundamentals.get("roe_pct")),
        "institutional_pct": _round(fundamentals.get("institutional_pct")),
        "breakout": result["breakout"],
        "entry": result.get("entry"),
        "last_close": round(closes[-1], 2) if closes else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "narrative": build_stock_narrative(label, result, market_state),
        "missing": list(fundamentals.get("missing", [])),
        "_result": result,
    }


def _round(value: Any) -> Optional[float]:
    return round(float(value), 1) if isinstance(value, (int, float)) else None


def _scan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    symbols = _stock_universe(cfg)
    dcfg = direction_config()
    market_state = _market_state()

    bars_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    fundamentals_by_symbol: Dict[str, Dict[str, Any]] = {}
    unavailable: List[str] = []
    for i, symbol in enumerate(symbols):
        bars = _fetch_bars(symbol, dcfg["history_days"])
        if len(bars) < 120:
            unavailable.append(symbol)
            continue
        bars_by_symbol[symbol] = bars
        started = time.monotonic()
        fundamentals_by_symbol[symbol] = get_fundamentals(symbol)
        fetched_live = time.monotonic() - started > 0.05
        if fetched_live and i < len(symbols) - 1 and cfg["fundamentals_pause_seconds"]:
            time.sleep(cfg["fundamentals_pause_seconds"])

    rs_raw = {
        symbol: weighted_rs_score([float(b["close"]) for b in bars])
        for symbol, bars in bars_by_symbol.items()
    }
    ranked = sorted(
        (s for s, v in rs_raw.items() if v is not None), key=lambda s: rs_raw[s]
    )
    n_ranked = len(ranked)
    percentiles = {
        symbol: (idx / (n_ranked - 1) * 100.0 if n_ranked > 1 else 50.0)
        for idx, symbol in enumerate(ranked)
    }

    items = [
        _leader_item(
            symbol,
            bars,
            fundamentals_by_symbol[symbol],
            percentiles.get(symbol),
            n_ranked,
            market_state,
            cfg,
        )
        for symbol, bars in bars_by_symbol.items()
    ]
    items.sort(
        key=lambda i: (
            _READINESS_ORDER.get(i["readiness"], 9),
            -(i["rs_percentile"] or 0),
        )
    )

    as_of = None
    if bars_by_symbol:
        as_of = max(bars[-1]["date"] for bars in bars_by_symbol.values())

    return {
        "as_of": as_of,
        "provisional": _market_open_now(),
        "market_state": market_state,
        "market_state_label": STATE_LABELS.get(market_state, "Unavailable"),
        "gate_open": market_state in ("confirmed_uptrend", "uptrend_under_pressure"),
        "gate_message": _GATE_MESSAGES.get(market_state, _GATE_MESSAGES["unavailable"]),
        "items": items,
        "universe": symbols,
        "excluded": list(cfg["exclude"]),
        "unavailable": unavailable,
        "stop_loss_pct": cfg["stop_loss_pct"],
        "methodology": (
            "Seven-letter CAN SLIM scan over the snapshot/watchlist stock universe. "
            f"C: quarterly EPS {cfg['quarterly_growth_met_pct']:.0f}%+ YoY. A: annual EPS "
            f"{cfg['annual_growth_met_pct']:.0f}%+/yr and ROE {cfg['roe_met_pct']:.0f}%+. N: 52-week-high "
            f"breakout on {cfg['breakout_volume_ratio']}x average volume. S: up/down volume above 1.00. "
            f"L: relative strength {cfg['rs_met_percentile']:.0f}th percentile+ (12-month, recent quarter "
            "double-weighted). I: institutional ownership with buying power left. M: the follow-through-day "
            "market state; no buy flags while the market is in a correction. Fundamentals are best-effort "
            "Yahoo data - unavailable letters are labeled, never guessed."
        ),
    }


def get_leaders() -> Dict[str, Any]:
    cfg = canslim_config()
    payload = cache.get_or_compute("canslim:leaders", TTL_LEADERS, lambda: _scan(cfg))
    try:
        _maybe_evaluate_stock_signals()
    except Exception:
        logger.exception("Lazy stock-signal evaluation failed")
    return {
        **payload,
        "items": [{k: v for k, v in i.items() if k != "_result"} for i in payload["items"]],
    }


def get_stock_detail(symbol: str) -> Optional[Dict[str, Any]]:
    cfg = canslim_config()
    symbol = symbol.upper()
    if symbol in cfg["exclude"]:
        return None
    dcfg = direction_config()
    bars = _fetch_bars(symbol, dcfg["history_days"])
    if len(bars) < 120:
        return None
    market_state = _market_state()
    fundamentals = get_fundamentals(symbol)
    item = _leader_item(symbol, bars, fundamentals, None, None, market_state, cfg)
    item.pop("_result", None)
    return item


_STOCK_EVAL_LOCK = None
_LAST_STOCK_EVAL_SESSION: Optional[str] = None


def _maybe_evaluate_stock_signals() -> None:
    """Evaluate once per completed session from whichever caller gets here first."""

    global _STOCK_EVAL_LOCK, _LAST_STOCK_EVAL_SESSION
    from threading import Lock

    if _STOCK_EVAL_LOCK is None:
        _STOCK_EVAL_LOCK = Lock()
    dcfg = direction_config()
    benchmark_bars = _completed_bars(_fetch_bars(dcfg["benchmark"], dcfg["history_days"]))
    if not benchmark_bars:
        return
    session = benchmark_bars[-1]["date"]
    with _STOCK_EVAL_LOCK:
        if _LAST_STOCK_EVAL_SESSION == session:
            return
        _LAST_STOCK_EVAL_SESSION = session
    evaluate_stock_signals()


def evaluate_stock_signals() -> Dict[str, Any]:
    """Persist fresh buy-candidate breakouts for the last completed session.

    Idempotent via the direction_signals unique constraint; delivery reuses
    the direction channels and only fires for newly inserted rows.
    """

    from backend.app import storage
    from backend.app.config import get_settings
    from backend.app.services_direction import _deliver_signals

    cfg = canslim_config()
    payload = _scan(cfg)
    inserted: List[Dict[str, Any]] = []
    for item in payload["items"]:
        result = item.get("_result")
        if not result or item["readiness"] != "buy_candidate":
            continue
        bars = _completed_bars(
            _fetch_bars(item["symbol"], direction_config()["history_days"])
        )
        breakout = detect_breakout(bars, result["config"], within=1)
        # Only the session that started the advance is signal-worthy; a
        # stock mid-run resolves to an older pivot and must not re-alert.
        if not breakout or breakout.get("sessions_ago", 0) != 0:
            continue
        label = item.get("name") or item["symbol"]
        completed_result = {**result, "breakout": breakout}
        text = describe_breakout_signal(label, completed_result)
        if not text:
            continue
        row = {
            "symbol": item["symbol"],
            "label": label,
            "signal_type": "stock_breakout",
            "signal_date": breakout["date"],
            "state": payload["market_state"],
            "title": text["title"],
            "message": text["message"],
            "payload": breakout,
        }
        try:
            if storage.insert_direction_signal(row):
                inserted.append(row)
        except Exception:
            logger.exception("Failed to persist breakout signal for %s", item["symbol"])

    if inserted:
        _deliver_signals(inserted, get_settings())
    return {
        "evaluated": len(payload["items"]),
        "inserted": len(inserted),
        "as_of": payload["as_of"],
    }
