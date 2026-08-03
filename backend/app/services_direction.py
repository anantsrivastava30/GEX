"""O'Neil market direction service.

Tracks a configured universe of market and sector index ETFs through the
rally-attempt / follow-through-day state machine, the index-adapted CAN SLIM
scorecard, RSI timing, and 20/50/200-day EMA touch detection. Newly detected
signals for completed sessions are persisted to SQLite and optionally
delivered through the server-owned Discord/email channels.

Data path: Tradier daily history when a token is configured, with a yfinance
fallback so the feature degrades to free data instead of an empty page.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Dict, List, Optional

from quant_analysis.analytics.market_direction import (
    STATE_LABELS,
    build_state_narrative,
    classify_market_state,
    describe_signal,
    ema_series,
    ema_statuses,
    entry_assessment,
    evaluate_index_scorecard,
    merged_config,
    pct_return,
    scorecard_summary,
    timing_assessment,
)
from quant_analysis.config import CONFIG

from backend.app.cache import cache
from backend.app.config import get_settings

logger = logging.getLogger(__name__)

TTL_DIRECTION_BARS = 1800
_DETAIL_SESSIONS = 160
_SIGNAL_EVENT_TYPES = {
    "follow_through_day",
    "rally_day1",
    "rally_failed",
    "correction_entered",
    "under_pressure",
}

_EVAL_LOCK = Lock()
_LAST_EVALUATED_SESSION: Optional[str] = None


def direction_config() -> Dict[str, Any]:
    raw = CONFIG.get("market_direction", {}) or {}
    cfg = merged_config(raw)
    cfg["benchmark"] = str(raw.get("benchmark", "SPY")).upper()
    cfg["history_days"] = int(raw.get("history_days", 730))
    indices = []
    for item in raw.get("indices", []) or []:
        symbol = str(item.get("symbol", "")).upper()
        if not symbol:
            continue
        indices.append(
            {
                "symbol": symbol,
                "label": str(item.get("label", symbol)),
                "group": str(item.get("group", "Index")),
                "domain": str(item["domain"]) if item.get("domain") else None,
            }
        )
    if not indices:
        indices = [{"symbol": "SPY", "label": "S&P 500", "group": "Broad market"}]
    if all(i["symbol"] != cfg["benchmark"] for i in indices):
        indices.insert(
            0,
            {"symbol": cfg["benchmark"], "label": cfg["benchmark"], "group": "Broad market"},
        )
    cfg["indices"] = indices
    return cfg


def _yf_bars(symbol: str, days: int) -> List[Dict[str, Any]]:
    import yfinance as yf

    period = "2y" if days > 365 else "1y"
    hist = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
    out: List[Dict[str, Any]] = []
    for ts, row in hist.iterrows():
        try:
            out.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row.get("Volume") or 0),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _fetch_bars(symbol: str, days: int) -> List[Dict[str, Any]]:
    """Daily OHLCV bars, Tradier first, yfinance fallback, cached."""

    def compute() -> List[Dict[str, Any]]:
        settings = get_settings()
        if settings.tradier_token:
            try:
                from backend.app.services import get_candles

                bars = get_candles(symbol, days)
                if len(bars) >= 250:
                    return bars
            except Exception:
                logger.warning("Tradier history failed for %s; trying yfinance", symbol)
        try:
            return _yf_bars(symbol, days)
        except Exception:
            logger.warning("yfinance history failed for %s", symbol)
            return []

    return cache.get_or_compute(
        f"direction:bars:{symbol}:{days}", TTL_DIRECTION_BARS, compute
    )


def _market_open_now() -> bool:
    from backend.app.jobs import market_is_open

    try:
        return market_is_open()
    except Exception:
        return False


def _completed_bars(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop today's partial bar while the session is still trading."""

    if bars and _market_open_now():
        return bars[:-1]
    return bars


def _analyze(
    entry: Dict[str, Any],
    bars: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    benchmark_state: str,
    benchmark_r63: Optional[float],
    rank: Optional[int],
    universe_size: Optional[int],
) -> Dict[str, Any]:
    state = classify_market_state(bars, cfg)
    timing = timing_assessment(bars, cfg) if bars else {"rsi": None, "zone": "unavailable", "label": "RSI unavailable"}
    emas = (
        ema_statuses(bars, list(cfg["ema_periods"]), cfg["ema_touch_band_pct"])
        if bars
        else []
    )
    scorecard = (
        evaluate_index_scorecard(
            bars, benchmark_r63, rank, universe_size, benchmark_state, cfg
        )
        if bars
        else []
    )
    closes = [float(b["close"]) for b in bars]
    change_pct = (
        (closes[-1] / closes[-2] - 1.0) * 100.0 if len(closes) > 1 and closes[-2] else None
    )
    entry_read = (
        entry_assessment(state, emas, closes[-1], cfg) if closes else None
    )
    return {
        "entry": entry_read,
        "symbol": entry["symbol"],
        "label": entry["label"],
        "group": entry["group"],
        "domain": entry.get("domain"),
        "state": state["state"],
        "state_label": state["state_label"],
        "state_since": state.get("mode_since"),
        "as_of": state.get("as_of"),
        "last_close": round(closes[-1], 2) if closes else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "return_3m": (
            round(pct_return(closes, 63), 2)
            if closes and pct_return(closes, 63) is not None
            else None
        ),
        "drawdown_pct": state.get("drawdown_pct"),
        "rally_day": state.get("rally_day"),
        "rally_day1_low": state.get("rally_day1_low"),
        "rally_day1_date": state.get("rally_day1_date"),
        "distribution_count": state.get("distribution_count"),
        "last_ftd": state.get("last_ftd"),
        "durability": state.get("durability"),
        "rsi": timing.get("rsi"),
        "rsi_zone": timing.get("zone"),
        "timing_label": timing.get("label"),
        "emas": [e for e in emas if e.get("value") is not None],
        "scorecard": scorecard,
        "score": scorecard_summary(scorecard) if scorecard else None,
        "narrative": build_state_narrative(entry["label"], state, timing, emas),
        "_state_raw": state,
    }


def _load_universe(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    bars_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for entry in cfg["indices"]:
        bars = _fetch_bars(entry["symbol"], cfg["history_days"])
        if bars:
            bars_by_symbol[entry["symbol"]] = bars
    return bars_by_symbol


def _overview_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bars_by_symbol = _load_universe(cfg)
    benchmark_symbol = cfg["benchmark"]
    benchmark_bars = bars_by_symbol.get(benchmark_symbol, [])
    benchmark_state_raw = (
        classify_market_state(benchmark_bars, cfg) if benchmark_bars else None
    )
    benchmark_state = (
        benchmark_state_raw["state"] if benchmark_state_raw else "unavailable"
    )
    benchmark_r63 = (
        pct_return([float(b["close"]) for b in benchmark_bars], 63)
        if benchmark_bars
        else None
    )

    returns: Dict[str, Optional[float]] = {}
    for symbol, bars in bars_by_symbol.items():
        returns[symbol] = pct_return([float(b["close"]) for b in bars], 63)
    ranked = sorted(
        (s for s, r in returns.items() if r is not None),
        key=lambda s: returns[s],
        reverse=True,
    )
    ranks = {symbol: i + 1 for i, symbol in enumerate(ranked)}

    indices = []
    unavailable = []
    for entry in cfg["indices"]:
        bars = bars_by_symbol.get(entry["symbol"])
        if not bars:
            unavailable.append(entry["symbol"])
            continue
        indices.append(
            _analyze(
                entry,
                bars,
                cfg,
                benchmark_state,
                benchmark_r63,
                ranks.get(entry["symbol"]),
                len(ranked),
            )
        )

    uptrend = sum(
        1
        for i in indices
        if i["state"] in ("confirmed_uptrend", "uptrend_under_pressure")
    )
    rallying = sum(1 for i in indices if i["state"] == "rally_attempt")
    above_50 = sum(
        1
        for i in indices
        if any(e["period"] == 50 and e["above"] for e in i["emas"])
    )
    total = len(indices)
    if total:
        if uptrend / total >= 0.6:
            reading = "Broad participation: most tracked indices are in uptrends."
        elif (uptrend + rallying) / total >= 0.5:
            reading = "Mixed breadth: participation is split between uptrends and repair attempts."
        else:
            reading = "Narrow breadth: most tracked indices are still correcting."
    else:
        reading = "No index data available."

    benchmark_index = next(
        (i for i in indices if i["symbol"] == benchmark_symbol), None
    )
    durability = None
    if benchmark_index and benchmark_index.get("durability"):
        d = dict(benchmark_index["durability"])
        checks = [
            {
                "name": "Holds the follow-through gains",
                "passed": bool(d.get("gains_held")),
                "detail": "The close is still at or above the follow-through day's close."
                if d.get("gains_held")
                else "The close has slipped below the follow-through day's close.",
            },
            {
                "name": "Distribution stays low",
                "passed": int(d.get("distribution_since") or 0) <= 2,
                "detail": f"{d.get('distribution_since')} distribution day(s) since the follow-through.",
            },
            {
                "name": "Breadth improves",
                "passed": bool(total) and (uptrend + rallying) / total >= 0.5,
                "detail": f"{uptrend} of {total} tracked indices are in uptrends, {rallying} attempting rallies.",
            },
        ]
        passed = sum(1 for c in checks if c["passed"])
        if passed == len(checks):
            assessment = "intact"
        elif passed >= 1:
            assessment = "mixed"
        else:
            assessment = "failing"
        durability = {**d, "checks": checks, "assessment": assessment}

    for i in indices:
        i.pop("_state_raw", None)

    return {
        "as_of": benchmark_index["as_of"] if benchmark_index else None,
        "provisional": _market_open_now(),
        "benchmark": benchmark_symbol,
        "breadth": {
            "uptrend_count": uptrend,
            "rally_count": rallying,
            "total": total,
            "above_ema50": above_50,
            "reading": reading,
        },
        "durability": durability,
        "indices": indices,
        "unavailable": unavailable,
        "thresholds": {
            "ftd_gain_pct": cfg["ftd_gain_pct"],
            "ftd_min_day": cfg["ftd_min_day"],
            "ftd_ideal_max_day": cfg["ftd_ideal_max_day"],
            "distribution_decline_pct": cfg["distribution_decline_pct"],
            "distribution_lookback": cfg["distribution_lookback"],
            "distribution_pressure_count": cfg["distribution_pressure_count"],
            "correction_drawdown_pct": cfg["correction_drawdown_pct"],
            "ema_periods": list(cfg["ema_periods"]),
            "ema_touch_band_pct": cfg["ema_touch_band_pct"],
            "rsi_period": cfg["rsi_period"],
        },
    }


def get_direction_overview() -> Dict[str, Any]:
    cfg = direction_config()
    payload = _overview_payload(cfg)
    try:
        _maybe_evaluate_signals(cfg)
    except Exception:
        logger.exception("Lazy direction-signal evaluation failed")
    return payload


def get_direction_detail(symbol: str) -> Optional[Dict[str, Any]]:
    cfg = direction_config()
    symbol = symbol.upper()
    entry = next((i for i in cfg["indices"] if i["symbol"] == symbol), None)
    if entry is None:
        return None
    bars = _fetch_bars(symbol, cfg["history_days"])
    if not bars:
        return None

    benchmark_bars = (
        bars
        if symbol == cfg["benchmark"]
        else _fetch_bars(cfg["benchmark"], cfg["history_days"])
    )
    benchmark_state = (
        classify_market_state(benchmark_bars, cfg)["state"]
        if benchmark_bars
        else "unavailable"
    )
    benchmark_r63 = (
        pct_return([float(b["close"]) for b in benchmark_bars], 63)
        if benchmark_bars
        else None
    )

    analysis = _analyze(entry, bars, cfg, benchmark_state, benchmark_r63, None, None)
    state = analysis.pop("_state_raw")

    window = bars[-_DETAIL_SESSIONS:]
    offset = len(bars) - len(window)
    series = ema_series(bars, list(cfg["ema_periods"]))
    trimmed_series = [
        {"period": s["period"], "values": s["values"][offset:]} for s in series
    ]
    window_dates = {b["date"] for b in window}
    markers = [
        {
            "date": e["date"],
            "kind": e["type"],
            "label": describe_signal(entry["label"], e, state, {})["title"],
        }
        for e in state.get("events", [])
        if e["date"] in window_dates
    ]

    return {
        "index": analysis,
        "candles": window,
        "ema_series": trimmed_series,
        "markers": markers,
        "provisional": _market_open_now(),
    }


def _maybe_evaluate_signals(cfg: Dict[str, Any]) -> None:
    """Evaluate once per completed session, from whichever caller gets here first."""

    global _LAST_EVALUATED_SESSION
    benchmark_bars = _completed_bars(
        _fetch_bars(cfg["benchmark"], cfg["history_days"])
    )
    if not benchmark_bars:
        return
    session = benchmark_bars[-1]["date"]
    with _EVAL_LOCK:
        if _LAST_EVALUATED_SESSION == session:
            return
        _LAST_EVALUATED_SESSION = session
    evaluate_direction_signals()


def evaluate_direction_signals() -> Dict[str, Any]:
    """Detect and persist signals for the latest completed session.

    Idempotent: the unique (symbol, signal_type, signal_date) constraint
    makes reruns free, and delivery happens only for newly inserted rows.
    """

    from backend.app import storage

    cfg = direction_config()
    settings = get_settings()
    inserted: List[Dict[str, Any]] = []
    evaluated = 0
    as_of = None

    for entry in cfg["indices"]:
        bars = _completed_bars(_fetch_bars(entry["symbol"], cfg["history_days"]))
        if len(bars) < cfg["min_history"]:
            continue
        evaluated += 1
        session = bars[-1]["date"]
        as_of = max(as_of, session) if as_of else session
        state = classify_market_state(bars, cfg)
        timing = timing_assessment(bars, cfg)

        events = [
            e
            for e in state.get("events", [])
            if e["date"] == session and e["type"] in _SIGNAL_EVENT_TYPES
        ]
        for ema in ema_statuses(bars, list(cfg["ema_periods"]), cfg["ema_touch_band_pct"]):
            if ema.get("value") is not None and ema.get("new_touch"):
                events.append(
                    {
                        "date": session,
                        "type": f"ema_touch_{ema['period']}",
                        "detail": {
                            "period": ema["period"],
                            "value": ema["value"],
                            "above": ema["above"],
                            "distance_pct": ema["distance_pct"],
                        },
                    }
                )

        for event in events:
            text = describe_signal(entry["label"], event, state, timing)
            row = {
                "symbol": entry["symbol"],
                "label": entry["label"],
                "signal_type": event["type"],
                "signal_date": event["date"],
                "state": state["state"],
                "title": text["title"],
                "message": text["message"],
                "payload": event.get("detail", {}),
            }
            try:
                row_id = storage.insert_direction_signal(row)
            except Exception:
                logger.exception(
                    "Failed to persist direction signal for %s", entry["symbol"]
                )
                continue
            if row_id:
                inserted.append(row)

    if inserted:
        _deliver_signals(inserted, settings)
    return {"evaluated": evaluated, "inserted": len(inserted), "as_of": as_of}


def _deliver_signals(rows: List[Dict[str, Any]], settings: Any) -> None:
    from backend.app.services_alerts import post_discord_message, send_email_message

    lines = [f"**Market direction** - {len(rows)} new signal(s)"]
    lines.extend(f"- {row['title']}: {row['message']}" for row in rows[:10])
    body = "\n".join(lines)
    if settings.direction_alert_discord and settings.alert_discord_webhook_url:
        try:
            post_discord_message(body)
        except Exception:
            logger.warning("Direction signal Discord delivery failed")
    if settings.direction_alert_email and settings.smtp_host and settings.alert_email_to:
        try:
            send_email_message(
                f"GEX market direction: {len(rows)} new signal(s)",
                body.replace("**", ""),
            )
        except Exception:
            logger.warning("Direction signal email delivery failed")


def list_direction_signals(limit: int, symbol: Optional[str] = None) -> Dict[str, Any]:
    from backend.app import storage

    items = storage.list_direction_signals(limit, symbol.upper() if symbol else None)
    return {"items": items, "count": len(items)}


_SCORABLE_TYPES = {"follow_through_day", "stock_breakout"}


def get_signal_outcomes(
    signal_type: Optional[str], horizon: int, limit: int
) -> Dict[str, Any]:
    """Realized outcomes for logged follow-through and breakout signals.

    Each signal carries its own entry and invalidation level, so the score
    is against the rule that produced it: a follow-through fails when the
    index closes back below the rally-attempt low, a breakout fails when
    the stock closes below its stop. Excursion stats answer the practical
    question - buying the signal, how deep was the drawdown before it
    worked, and would a mechanical 7-8% stop have fired?
    """

    from backend.app import storage
    from quant_analysis.analytics.track_record import (
        evaluate_direction_signal_outcome,
        summarize_direction_outcomes,
    )

    cfg = direction_config()
    stop_pct = float(
        (CONFIG.get("canslim", {}) or {}).get("stop_loss_pct", 8.0)
    )
    wanted = {signal_type} if signal_type else _SCORABLE_TYPES
    raw = storage.list_direction_signals(limit * 4, None)
    signals = [s for s in raw if s["signal_type"] in wanted][:limit]

    rows: List[Dict[str, Any]] = []
    for signal in signals:
        payload = signal.get("payload") or {}
        entry_price = payload.get("close") or payload.get("price")
        invalidation = payload.get("anchor_low") or payload.get("stop_price")
        if entry_price is None or invalidation is None:
            continue
        bars = _completed_bars(_fetch_bars(signal["symbol"], cfg["history_days"]))
        if not bars:
            continue
        outcome = evaluate_direction_signal_outcome(
            signal["signal_date"],
            float(entry_price),
            float(invalidation),
            bars,
            horizon_sessions=horizon,
            stop_pct=stop_pct,
        )
        rows.append(
            {
                "id": signal["id"],
                "symbol": signal["symbol"],
                "label": signal["label"],
                "signal_type": signal["signal_type"],
                "signal_date": signal["signal_date"],
                "entry_price": round(float(entry_price), 2),
                "invalidation_level": round(float(invalidation), 2),
                **outcome,
            }
        )

    return {
        "horizon_sessions": horizon,
        "stop_pct": stop_pct,
        "summary": summarize_direction_outcomes(rows),
        "rows": rows,
    }


def state_labels() -> Dict[str, str]:
    return dict(STATE_LABELS)
