"""O'Neil-style market direction analytics.

Pure functions over daily OHLCV bars: EMA and RSI math, the rally-attempt /
follow-through-day / distribution-day state machine, EMA touch detection, and
an index-adapted CAN SLIM scorecard. No I/O and no provider calls; callers
supply bars as dicts with date, open, high, low, close, volume (oldest first).

The state machine implements the published O'Neil rules:
- Correction: the index has fallen from its trailing high past a threshold,
  undercut the low that anchored the last follow-through day, or accumulated
  a cluster of distribution days.
- Rally attempt: the first up close after a correction low starts day 1; the
  attempt survives while the day-1 low holds and fails on any undercut.
- Follow-through day: on day 4 or later of the attempt, a close up at least
  the configured percent on volume above the prior day confirms a new
  uptrend. Days 1-3 never confirm. Day 4-7 is the classic window; later
  follow-throughs still count but are labeled late.
- Distribution days: a decline of at least the configured percent on rising
  volume while in an uptrend. Each expires after a lookback window or once
  the index recovers the configured percent above that day's close. A
  growing cluster degrades the uptrend to under pressure and eventually to
  correction.

Every threshold is configurable and surfaced to the caller so the UI can
show, not hide, the parameters in use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

STATE_CONFIRMED = "confirmed_uptrend"
STATE_PRESSURE = "uptrend_under_pressure"
STATE_RALLY = "rally_attempt"
STATE_CORRECTION = "correction"
STATE_UNAVAILABLE = "unavailable"

STATE_LABELS = {
    STATE_CONFIRMED: "Confirmed uptrend",
    STATE_PRESSURE: "Uptrend under pressure",
    STATE_RALLY: "Rally attempt",
    STATE_CORRECTION: "Correction",
    STATE_UNAVAILABLE: "Insufficient history",
}

DEFAULTS: Dict[str, Any] = {
    "warmup_sessions": 50,
    "min_history": 120,
    "ftd_gain_pct": 1.25,
    "ftd_min_day": 4,
    "ftd_ideal_max_day": 7,
    "distribution_decline_pct": 0.2,
    "distribution_lookback": 25,
    "distribution_recovery_pct": 5.0,
    "distribution_pressure_count": 4,
    "distribution_correction_count": 6,
    "correction_drawdown_pct": 8.0,
    "ema_periods": [20, 50, 200],
    "ema_touch_band_pct": 0.4,
    "rsi_period": 14,
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    # How far above the follow-through close an index still counts as
    # buyable - O'Neil's never-chase-more-than-5%-past-the-pivot rule
    # applied at the index level.
    "entry_extended_pct": 5.0,
    "ema_pullback_band_pct": 1.5,
}


def merged_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    for key, value in (config or {}).items():
        if key in cfg and value is not None:
            cfg[key] = value
    return cfg


# ---------------------------------------------------------------------------
# Indicator math
# ---------------------------------------------------------------------------


def compute_ema(values: List[float], period: int) -> List[Optional[float]]:
    """EMA seeded with the SMA of the first ``period`` values."""

    if period <= 0 or len(values) < period:
        return [None] * len(values)
    out: List[Optional[float]] = [None] * (period - 1)
    prev = sum(values[:period]) / period
    out.append(prev)
    k = 2.0 / (period + 1)
    for value in values[period:]:
        prev = value * k + prev * (1.0 - k)
        out.append(prev)
    return out


def compute_rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """Wilder RSI; None until enough history has accrued."""

    if period <= 0 or len(closes) < period + 1:
        return [None] * len(closes)
    out: List[Optional[float]] = [None] * period
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, period + 1)]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, period + 1)]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def rsi_value(g: float, l: float) -> float:
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - 100.0 / (1.0 + rs)

    out.append(rsi_value(avg_gain, avg_loss))
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out.append(rsi_value(avg_gain, avg_loss))
    return out


def pct_return(closes: List[float], sessions: int) -> Optional[float]:
    if len(closes) <= sessions or closes[-1 - sessions] == 0:
        return None
    return (closes[-1] / closes[-1 - sessions] - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Market state machine
# ---------------------------------------------------------------------------


def classify_market_state(
    bars: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run the rally-attempt / follow-through / distribution state machine.

    Returns the current state plus the full dated event log so callers can
    annotate charts and persist newly detected signals.
    """

    cfg = merged_config(config)
    n = len(bars)
    if n < cfg["min_history"]:
        return {
            "state": STATE_UNAVAILABLE,
            "state_label": STATE_LABELS[STATE_UNAVAILABLE],
            "as_of": bars[-1]["date"] if bars else None,
            "events": [],
            "config": cfg,
        }

    closes = [float(b["close"]) for b in bars]
    warmup = min(int(cfg["warmup_sessions"]), n - 2)
    events: List[Dict[str, Any]] = []

    running_high = max(closes[: warmup + 1])
    seed_drawdown = (closes[warmup] / running_high - 1.0) * 100.0
    mode = "uptrend" if seed_drawdown > -cfg["correction_drawdown_pct"] else "correction"
    mode_since = bars[warmup]["date"]

    trailing_high = running_high
    correction_high = running_high
    correction_low = min(float(b["low"]) for b in bars[: warmup + 1])
    rally_day1_low: Optional[float] = None
    rally_day1_date: Optional[str] = None
    rally_days = 0
    last_ftd: Optional[Dict[str, Any]] = None
    dist_days: List[Dict[str, Any]] = []
    was_pressured = False

    def add_event(date: str, etype: str, detail: Dict[str, Any]) -> None:
        events.append({"date": date, "type": etype, "detail": detail})

    for i in range(warmup + 1, n):
        cur = bars[i]
        prev = bars[i - 1]
        close = float(cur["close"])
        low = float(cur["low"])
        prev_close = float(prev["close"])
        change_pct = (close / prev_close - 1.0) * 100.0 if prev_close else 0.0
        cur_vol = float(cur.get("volume") or 0)
        prev_vol = float(prev.get("volume") or 0)
        vol_up = cur_vol > prev_vol > 0

        if mode == "uptrend":
            trailing_high = max(trailing_high, close)
            dist_days = [
                d
                for d in dist_days
                if i - d["i"] <= cfg["distribution_lookback"]
                and close < d["close"] * (1.0 + cfg["distribution_recovery_pct"] / 100.0)
            ]
            if change_pct <= -cfg["distribution_decline_pct"] and vol_up:
                dist_days.append({"i": i, "date": cur["date"], "close": close})
                add_event(
                    cur["date"],
                    "distribution_day",
                    {"change_pct": round(change_pct, 2), "count": len(dist_days)},
                )
                if (
                    len(dist_days) >= cfg["distribution_pressure_count"]
                    and not was_pressured
                ):
                    was_pressured = True
                    add_event(
                        cur["date"], "under_pressure", {"count": len(dist_days)}
                    )
            if len(dist_days) < cfg["distribution_pressure_count"]:
                was_pressured = False

            drawdown = (close / trailing_high - 1.0) * 100.0
            undercut_ftd = bool(last_ftd) and close < float(last_ftd["anchor_low"])
            reason = None
            if undercut_ftd:
                reason = "undercut the rally low that anchored the follow-through day"
            elif len(dist_days) >= cfg["distribution_correction_count"]:
                reason = f"distribution cluster reached {len(dist_days)} days"
            elif drawdown <= -cfg["correction_drawdown_pct"]:
                reason = (
                    f"fell {abs(round(drawdown, 1))}% from the trailing high"
                )
            if reason:
                mode = "correction"
                mode_since = cur["date"]
                correction_high = trailing_high
                correction_low = low
                dist_days = []
                was_pressured = False
                add_event(cur["date"], "correction_entered", {"reason": reason})

        elif mode == "correction":
            correction_low = min(correction_low, low)
            if close > prev_close:
                mode = "rally"
                mode_since = cur["date"]
                rally_day1_low = low
                rally_day1_date = cur["date"]
                rally_days = 1
                add_event(
                    cur["date"],
                    "rally_day1",
                    {"day1_low": round(rally_day1_low, 2)},
                )

        elif mode == "rally":
            if rally_day1_low is not None and low < rally_day1_low:
                mode = "correction"
                mode_since = cur["date"]
                correction_low = min(correction_low, low)
                add_event(
                    cur["date"],
                    "rally_failed",
                    {"day1_low": round(rally_day1_low, 2), "days": rally_days},
                )
                rally_days = 0
                rally_day1_low = None
                rally_day1_date = None
            else:
                rally_days += 1
                if (
                    rally_days >= cfg["ftd_min_day"]
                    and change_pct >= cfg["ftd_gain_pct"]
                    and vol_up
                ):
                    quality = (
                        "ideal" if rally_days <= cfg["ftd_ideal_max_day"] else "late"
                    )
                    last_ftd = {
                        "date": cur["date"],
                        "i": i,
                        "gain_pct": round(change_pct, 2),
                        "volume_ratio": round(cur_vol / prev_vol, 2)
                        if prev_vol
                        else None,
                        "day_number": rally_days,
                        "quality": quality,
                        "anchor_low": rally_day1_low,
                        "close": close,
                    }
                    mode = "uptrend"
                    mode_since = cur["date"]
                    trailing_high = close
                    dist_days = []
                    was_pressured = False
                    add_event(
                        cur["date"],
                        "follow_through_day",
                        {
                            "gain_pct": last_ftd["gain_pct"],
                            "volume_ratio": last_ftd["volume_ratio"],
                            "day_number": rally_days,
                            "quality": quality,
                            # Entry and invalidation levels ride along so
                            # persisted signals can be outcome-scored later.
                            "close": round(close, 2),
                            "anchor_low": round(rally_day1_low, 2)
                            if rally_day1_low is not None
                            else None,
                        },
                    )
                    rally_days = 0
                    rally_day1_date = None

    if mode == "uptrend":
        state = (
            STATE_PRESSURE
            if len(dist_days) >= cfg["distribution_pressure_count"]
            else STATE_CONFIRMED
        )
        drawdown_pct = (closes[-1] / trailing_high - 1.0) * 100.0
    elif mode == "rally":
        state = STATE_RALLY
        drawdown_pct = (closes[-1] / correction_high - 1.0) * 100.0
    else:
        state = STATE_CORRECTION
        drawdown_pct = (closes[-1] / correction_high - 1.0) * 100.0

    durability = None
    if mode == "uptrend" and last_ftd:
        dist_since = sum(
            1
            for e in events
            if e["type"] == "distribution_day" and e["date"] > last_ftd["date"]
        )
        durability = {
            "ftd_date": last_ftd["date"],
            "sessions_since": (n - 1) - int(last_ftd["i"]),
            "distribution_since": dist_since,
            "gains_held": closes[-1] >= float(last_ftd["close"]),
        }

    return {
        "state": state,
        "state_label": STATE_LABELS[state],
        "as_of": bars[-1]["date"],
        "mode_since": mode_since,
        "rally_day": rally_days if mode == "rally" else None,
        "rally_day1_low": round(rally_day1_low, 2)
        if mode == "rally" and rally_day1_low is not None
        else None,
        "rally_day1_date": rally_day1_date if mode == "rally" else None,
        "correction_low": round(correction_low, 2),
        "drawdown_pct": round(drawdown_pct, 2),
        "distribution_count": len(dist_days) if mode == "uptrend" else None,
        "distribution_dates": [d["date"] for d in dist_days]
        if mode == "uptrend"
        else [],
        "last_ftd": {k: v for k, v in last_ftd.items() if k != "i"}
        if last_ftd
        else None,
        "durability": durability,
        "events": events,
        "config": cfg,
    }


# ---------------------------------------------------------------------------
# EMA status and touch detection
# ---------------------------------------------------------------------------


def ema_statuses(
    bars: List[Dict[str, Any]],
    periods: List[int],
    band_pct: float,
) -> List[Dict[str, Any]]:
    """Current price position vs each EMA, with touch edge detection.

    A touch means the day's range crossed the EMA or the close finished
    within ``band_pct`` percent of it. ``new_touch`` is true only when the
    prior session was not already touching, so repeated alerts are not
    emitted while price rides along the average.
    """

    closes = [float(b["close"]) for b in bars]
    out: List[Dict[str, Any]] = []
    for period in periods:
        series = compute_ema(closes, period)
        value = series[-1] if series else None
        if value is None:
            out.append({"period": period, "value": None})
            continue

        def touched_at(idx: int) -> bool:
            ema_v = series[idx]
            if ema_v is None:
                return False
            bar = bars[idx]
            in_range = float(bar["low"]) <= ema_v <= float(bar["high"])
            near = abs(float(bar["close"]) - ema_v) / ema_v * 100.0 <= band_pct
            return in_range or near

        touched = touched_at(len(bars) - 1)
        prev_touched = touched_at(len(bars) - 2) if len(bars) > 1 else False
        prev_value = series[-2] if len(series) > 1 else None
        prev_close = closes[-2] if len(closes) > 1 else None
        crossed_down = (
            prev_value is not None
            and prev_close is not None
            and prev_close >= prev_value
            and closes[-1] < value
        )
        crossed_up = (
            prev_value is not None
            and prev_close is not None
            and prev_close < prev_value
            and closes[-1] >= value
        )
        out.append(
            {
                "period": period,
                "value": round(value, 2),
                "distance_pct": round((closes[-1] / value - 1.0) * 100.0, 2),
                "above": closes[-1] >= value,
                "touched": touched,
                "new_touch": touched and not prev_touched,
                "crossed_up": crossed_up,
                "crossed_down": crossed_down,
            }
        )
    return out


def ema_series(
    bars: List[Dict[str, Any]], periods: List[int]
) -> List[Dict[str, Any]]:
    """Full EMA series per period for charting, aligned with ``bars``."""

    closes = [float(b["close"]) for b in bars]
    return [
        {
            "period": period,
            "values": [
                round(v, 2) if v is not None else None
                for v in compute_ema(closes, period)
            ],
        }
        for period in periods
    ]


# ---------------------------------------------------------------------------
# RSI timing
# ---------------------------------------------------------------------------


def timing_assessment(
    bars: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    closes = [float(b["close"]) for b in bars]
    series = compute_rsi(closes, int(cfg["rsi_period"]))
    value = series[-1] if series else None
    if value is None:
        return {"rsi": None, "zone": "unavailable", "label": "RSI unavailable"}
    if value <= cfg["rsi_oversold"]:
        zone, label = "oversold", "Buy zone - oversold"
    elif value >= cfg["rsi_overbought"]:
        zone, label = "overbought", "Overbought - caution"
    else:
        zone, label = "neutral", "Neutral - wait"
    return {"rsi": round(value, 1), "zone": zone, "label": label}


# ---------------------------------------------------------------------------
# Entry assessment: is this index buyable here, or extended?
# ---------------------------------------------------------------------------


def entry_assessment(
    state: Dict[str, Any],
    emas: List[Dict[str, Any]],
    last_close: float,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, str]]:
    """Separate the regime call from the entry price.

    A confirmed uptrend is permission, not an entry: the buyable window is
    within ``entry_extended_pct`` of the follow-through close, and the
    second-chance entry is the first pullback to a rising 20/50-day EMA.
    Everything past that is extended - the exact "bought high, got stopped"
    trap this label exists to prevent.
    """

    cfg = merged_config(cfg)
    s = state.get("state")
    if s == STATE_RALLY:
        return {
            "status": "wait",
            "label": "Wait for a follow-through day",
            "detail": "A rally attempt is not confirmation; days 1-3 bounces fail more often than not.",
        }
    if s == STATE_CORRECTION:
        return {
            "status": "no_entry",
            "label": "No entries - correction",
            "detail": "O'Neil made no new buys while the market was in a correction.",
        }
    if s not in (STATE_CONFIRMED, STATE_PRESSURE):
        return None

    pullback = next(
        (
            e
            for e in emas
            if e.get("value") is not None
            and e["period"] in (20, 50)
            and (e.get("touched") or abs(e.get("distance_pct") or 99) <= cfg["ema_pullback_band_pct"])
            and e.get("above", True)
        ),
        None,
    )
    if pullback:
        return {
            "status": "pullback_entry",
            "label": f"Pullback entry - testing the {pullback['period']}-day EMA",
            "detail": (
                f"The first orderly test of the {pullback['period']}-day EMA in an uptrend is the classic "
                "second-chance entry for those who missed the follow-through day."
            ),
        }

    ftd = state.get("last_ftd") or {}
    ftd_close = ftd.get("close")
    if ftd_close:
        extension = (last_close / float(ftd_close) - 1.0) * 100.0
        if extension <= cfg["entry_extended_pct"]:
            return {
                "status": "buyable",
                "label": f"Buyable - {extension:+.1f}% vs the follow-through close",
                "detail": (
                    f"Still within {cfg['entry_extended_pct']:.0f}% of the follow-through close, so an entry "
                    "here is not chasing."
                ),
            }
        return {
            "status": "extended",
            "label": f"Extended - {extension:+.1f}% past the follow-through close",
            "detail": (
                f"More than {cfg['entry_extended_pct']:.0f}% above the confirmation. Chasing here puts a "
                "7-8% stop inside the range of a normal pullback; wait for a test of the 20/50-day EMA."
            ),
        }

    ema20 = next((e for e in emas if e.get("period") == 20 and e.get("value") is not None), None)
    if ema20 is not None:
        distance = ema20.get("distance_pct") or 0.0
        if distance > cfg["entry_extended_pct"]:
            return {
                "status": "extended",
                "label": f"Extended - {distance:+.1f}% above the 20-day EMA",
                "detail": "Stretched above its short-term average; wait for a pullback.",
            }
        return {
            "status": "buyable",
            "label": "In trend - near the 20-day EMA",
            "detail": "Trading close to a rising short-term average, not extended.",
        }
    return None


# ---------------------------------------------------------------------------
# Index-adapted CAN SLIM scorecard
# ---------------------------------------------------------------------------


def _row(
    letter: str, name: str, status: str, value: str, detail: str
) -> Dict[str, str]:
    return {
        "letter": letter,
        "name": name,
        "status": status,
        "value": value,
        "detail": detail,
    }


def evaluate_index_scorecard(
    bars: List[Dict[str, Any]],
    benchmark_return_63: Optional[float],
    rank: Optional[int],
    universe_size: Optional[int],
    market_state: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """CAN SLIM adapted to indices and sector ETFs.

    Earnings-based letters do not exist for an index, so each letter maps to
    the closest computable price/volume proxy. The mapping is documented in
    the UI help so nobody mistakes this for per-stock fundamental CAN SLIM.
    """

    cfg = merged_config(cfg)
    closes = [float(b["close"]) for b in bars]
    rows: List[Dict[str, str]] = []

    r63 = pct_return(closes, 63)
    ema50 = compute_ema(closes, 50)[-1] if len(closes) >= 50 else None
    if r63 is None or ema50 is None:
        rows.append(_row("C", "Current trend (3-month)", "unavailable", "n/a",
                         "Needs at least 63 sessions of history."))
    else:
        above = closes[-1] >= ema50
        status = "met" if r63 > 0 and above else ("borderline" if r63 > 0 or above else "not_met")
        rows.append(_row(
            "C", "Current trend (3-month)", status,
            f"{r63:+.1f}% / {'above' if above else 'below'} 50-EMA",
            f"3-month return {r63:+.1f}%, closing {'above' if above else 'below'} the 50-day EMA.",
        ))

    r252 = pct_return(closes, 252)
    ema200 = compute_ema(closes, 200)[-1] if len(closes) >= 200 else None
    if r252 is None or ema200 is None:
        rows.append(_row("A", "Annual trend (12-month)", "unavailable", "n/a",
                         "Needs at least 252 sessions of history."))
    else:
        above = closes[-1] >= ema200
        status = "met" if r252 > 0 and above else ("borderline" if r252 > 0 or above else "not_met")
        rows.append(_row(
            "A", "Annual trend (12-month)", status,
            f"{r252:+.1f}% / {'above' if above else 'below'} 200-EMA",
            f"12-month return {r252:+.1f}%, closing {'above' if above else 'below'} the 200-day EMA.",
        ))

    window = closes[-252:] if len(closes) >= 2 else closes
    high_52w = max(window) if window else None
    if not high_52w:
        rows.append(_row("N", "New high proximity", "unavailable", "n/a",
                         "No price history."))
    else:
        off_high = (closes[-1] / high_52w - 1.0) * 100.0
        status = "met" if off_high >= -5 else ("borderline" if off_high >= -15 else "not_met")
        rows.append(_row(
            "N", "New high proximity", status, f"{off_high:.1f}% off 52w high",
            f"Closing {abs(off_high):.1f}% below the 52-week high. O'Neil bought strength near new highs, not deep discounts.",
        ))

    up_vol = 0.0
    down_vol = 0.0
    for i in range(max(1, len(bars) - 50), len(bars)):
        vol = float(bars[i].get("volume") or 0)
        if closes[i] > closes[i - 1]:
            up_vol += vol
        elif closes[i] < closes[i - 1]:
            down_vol += vol
    if up_vol + down_vol <= 0:
        rows.append(_row("S", "Supply and demand (volume)", "unavailable", "n/a",
                         "Volume data unavailable for this symbol."))
    else:
        ratio = up_vol / down_vol if down_vol else float("inf")
        status = "met" if ratio > 1.0 else ("borderline" if ratio >= 0.85 else "not_met")
        shown = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        rows.append(_row(
            "S", "Supply and demand (volume)", status, f"up/down vol {shown}",
            f"Volume on up days vs down days over 50 sessions is {shown}. Above 1.00 means demand is absorbing supply.",
        ))

    if r63 is None or benchmark_return_63 is None:
        rows.append(_row("L", "Leader or laggard", "unavailable", "n/a",
                         "Needs 63 sessions for the index and the benchmark."))
    else:
        excess = r63 - benchmark_return_63
        status = "met" if excess > 0 else ("borderline" if excess > -1.0 else "not_met")
        rank_part = (
            f" · #{rank} of {universe_size}" if rank and universe_size else ""
        )
        rows.append(_row(
            "L", "Leader or laggard", status,
            f"{excess:+.1f}% vs SPY{rank_part}",
            f"3-month return is {excess:+.1f}% versus the broad market. O'Neil bought leaders, not laggards.",
        ))

    acc = 0
    dist = 0
    for i in range(max(1, len(bars) - 25), len(bars)):
        prev_vol = float(bars[i - 1].get("volume") or 0)
        vol = float(bars[i].get("volume") or 0)
        if not (vol > prev_vol > 0):
            continue
        chg = (closes[i] / closes[i - 1] - 1.0) * 100.0 if closes[i - 1] else 0.0
        if chg >= cfg["distribution_decline_pct"]:
            acc += 1
        elif chg <= -cfg["distribution_decline_pct"]:
            dist += 1
    net = acc - dist
    status = "met" if net > 0 else ("borderline" if net == 0 else "not_met")
    rows.append(_row(
        "I", "Institutional accumulation", status,
        f"{acc} acc / {dist} dist days",
        f"Net {net:+d} accumulation-minus-distribution days over 25 sessions. Rising-volume up days are the fingerprint of institutional buying.",
    ))

    if market_state == STATE_CONFIRMED:
        status, detail = "met", "The broad market is in a confirmed uptrend."
    elif market_state in (STATE_PRESSURE, STATE_RALLY):
        status = "borderline"
        detail = (
            "The broad market is under pressure or attempting a rally; three of four stocks follow the market."
        )
    elif market_state == STATE_CORRECTION:
        status, detail = "not_met", "The broad market is in a correction."
    else:
        status, detail = "unavailable", "Broad market state unavailable."
    rows.append(_row(
        "M", "Market direction", status, STATE_LABELS.get(market_state, "n/a"), detail
    ))

    return rows


def scorecard_summary(rows: List[Dict[str, str]]) -> Dict[str, int]:
    met = sum(1 for r in rows if r["status"] == "met")
    scored = sum(1 for r in rows if r["status"] != "unavailable")
    return {"met": met, "scored": scored, "total": len(rows)}


# ---------------------------------------------------------------------------
# Narrative builders
# ---------------------------------------------------------------------------


def _ema_sentence(statuses: List[Dict[str, Any]]) -> Optional[str]:
    parts = []
    for s in statuses:
        if s.get("value") is None:
            continue
        parts.append(
            f"{s['distance_pct']:+.1f}% vs {s['period']}-day EMA"
        )
    if not parts:
        return None
    return "Price sits " + ", ".join(parts) + "."


def build_state_narrative(
    label: str,
    state: Dict[str, Any],
    timing: Dict[str, Any],
    emas: List[Dict[str, Any]],
) -> List[str]:
    """Plain-English read of the current state, 2-4 sentences."""

    cfg = state.get("config", DEFAULTS)
    lines: List[str] = []
    s = state["state"]

    if s == STATE_CORRECTION:
        lines.append(
            f"{label} is in a correction, {abs(state['drawdown_pct']):.1f}% below its recent high."
        )
        lines.append(
            "Watching for a rally attempt: the first up close off the low starts the day count, and that day's low must hold."
        )
    elif s == STATE_RALLY:
        day = state.get("rally_day") or 1
        lines.append(
            f"{label} is on day {day} of a rally attempt (day-1 low {state.get('rally_day1_low')})."
        )
        if day < cfg["ftd_min_day"]:
            lines.append(
                f"A follow-through day cannot confirm before day {cfg['ftd_min_day']}; an early surge is treated as a reflexive bounce, not accumulation."
            )
        else:
            lines.append(
                f"A gain of {cfg['ftd_gain_pct']}%+ on volume above the prior day would confirm a new uptrend. The attempt fails if the day-1 low is undercut."
            )
    elif s == STATE_PRESSURE:
        lines.append(
            f"{label} is in an uptrend under pressure: {state.get('distribution_count')} distribution days in the last {cfg['distribution_lookback']} sessions."
        )
        lines.append(
            "Clusters of high-volume down days are the fingerprint of institutional selling; the uptrend degrades to a correction if the cluster keeps building."
        )
    elif s == STATE_CONFIRMED:
        ftd = state.get("last_ftd")
        if ftd:
            lines.append(
                f"{label} is in a confirmed uptrend since the follow-through day on {ftd['date']} "
                f"(day {ftd['day_number']}, {ftd['gain_pct']:+.1f}% on rising volume)."
            )
        else:
            lines.append(f"{label} is in a confirmed uptrend.")
        count = state.get("distribution_count") or 0
        lines.append(
            f"{count} distribution day(s) in the last {cfg['distribution_lookback']} sessions."
        )
    else:
        lines.append(f"Not enough history to classify {label}.")
        return lines

    if timing.get("rsi") is not None:
        lines.append(f"RSI({cfg['rsi_period']}) is {timing['rsi']} ({timing['label'].lower()}).")
    ema_line = _ema_sentence(emas)
    if ema_line:
        lines.append(ema_line)
    return lines


def describe_signal(
    label: str,
    event: Dict[str, Any],
    state: Dict[str, Any],
    timing: Dict[str, Any],
) -> Dict[str, str]:
    """Title and message for a persisted signal event."""

    etype = event["type"]
    detail = event.get("detail", {})
    cfg = state.get("config", DEFAULTS)

    if etype == "follow_through_day":
        title = f"{label}: follow-through day (day {detail.get('day_number')})"
        anchor = detail.get("anchor_low")
        invalidation = (
            f"The rally-attempt low of {anchor} is the level that invalidates this signal"
            if anchor is not None
            else "An undercut of the rally-attempt low invalidates this signal"
        )
        message = (
            f"{label} rose {detail.get('gain_pct')}% on volume above the prior day, on day "
            f"{detail.get('day_number')} of the rally attempt - a follow-through day by the O'Neil rule, "
            "confirming a new uptrend. This is permission to start buying, not a signal to buy the index "
            "itself: O'Neil took a pilot position here and added only as the rally proved itself, with new "
            f"money going into leading stocks clearing their own pivots. {invalidation} - index exposure is "
            "managed by that level and by distribution days, not by the 7-8% stock stop. Not every "
            "follow-through works; a distribution cluster in the next sessions is the classic warning that "
            "the bottom is not holding."
        )
    elif etype == "rally_day1":
        title = f"{label}: rally attempt, day 1"
        message = (
            f"{label} closed up off its correction low - day 1 of a rally attempt. The day-1 low of "
            f"{detail.get('day1_low')} must hold; a follow-through day from day {cfg['ftd_min_day']} on would confirm."
        )
    elif etype == "rally_failed":
        title = f"{label}: rally attempt failed"
        message = (
            f"{label} undercut its day-1 low of {detail.get('day1_low')} after {detail.get('days')} day(s). "
            "The attempt resets; the correction resumes until a new up day starts the count again."
        )
    elif etype == "correction_entered":
        title = f"{label}: correction"
        message = (
            f"{label} entered a correction: it {detail.get('reason', 'broke down')}. "
            "O'Neil's rule is to wait for the market to prove a bottom, not predict one."
        )
    elif etype == "under_pressure":
        title = f"{label}: distribution cluster"
        message = (
            f"{label} has logged {detail.get('count')} distribution days in the last "
            f"{cfg['distribution_lookback']} sessions - institutional selling pressure. The uptrend is under pressure."
        )
    elif etype == "distribution_day":
        title = f"{label}: distribution day ({detail.get('count')})"
        message = (
            f"{label} fell {detail.get('change_pct')}% on volume above the prior day - distribution day "
            f"number {detail.get('count')} in the rolling window."
        )
    elif etype.startswith("ema_touch_"):
        period = detail.get("period")
        above = bool(detail.get("above"))
        side = "above" if above else "below"
        role = "support" if above else "resistance"
        title = f"{label}: tagged the {period}-day EMA"
        message = (
            f"{label} traded into its {period}-day EMA ({detail.get('value')}) and closed {side} it - a "
            f"widely watched {role} test."
        )
        # In an uptrend, the first orderly pullback to a rising 20/50-day EMA
        # is the textbook second-chance entry - the answer to "the trend
        # confirmed while I was not watching, do I chase?"
        if above and period in (20, 50) and state.get("state") in (
            STATE_CONFIRMED,
            STATE_PRESSURE,
        ):
            message += (
                f" With the trend intact, a pullback to the {period}-day EMA is the standard "
                "second-chance entry for anyone who missed the follow-through day - an entry here sits "
                "near support rather than extended above it, so a stop has room to work."
            )
        if timing.get("zone") == "oversold":
            message += f" RSI is {timing.get('rsi')} (oversold), the zone where durable bottoms tend to start."
    else:
        title = f"{label}: {etype}"
        message = f"{label} event {etype}."
    return {"title": title, "message": message}
