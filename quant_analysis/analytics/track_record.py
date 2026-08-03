"""Realized-outcome scoring for logged gamma-gap signals.

The gamma-gap thesis is that spot gets pulled toward the magnet strike.
A signal's outcome is therefore defined as: did the magnet strike trade
(session low <= magnet <= session high) within the next N sessions?

Honesty rules baked in:
- Evaluation starts the session AFTER the signal date. A same-day daily bar
  cannot show whether the magnet printed before or after the scan, so the
  signal day never counts as a hit.
- A signal with fewer than N completed post-signal sessions and no hit yet
  is "pending", never a miss.
- Signal timestamps are stored in UTC; the date is taken from the timestamp
  as-is. Scans between midnight and ~4am UTC land on the prior US session's
  date; scheduler scans run during US market hours so this is not hit in
  practice.

Pure functions over candle rows - no network or storage access.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

OUTCOME_HIT = "hit"
OUTCOME_MISS = "miss"
OUTCOME_PENDING = "pending"

DEFAULT_HORIZON_SESSIONS = 5

# Score buckets for the hit-rate breakdown (0-120 scale).
SCORE_BUCKETS = [(0.0, 40.0), (40.0, 80.0), (80.0, 120.0 + 1e-9)]


def evaluate_gamma_gap_outcome(
    signal_date: str,
    magnet_strike: float,
    candles: Sequence[Dict[str, Any]],
    horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
) -> Dict[str, Any]:
    """Outcome of one signal against daily candles.

    ``candles`` are dicts with ``date`` (YYYY-MM-DD), ``high``, ``low``,
    in any order. Returns ``outcome`` (hit/miss/pending),
    ``sessions_to_hit`` (1-based, None unless hit) and
    ``evaluated_sessions`` (post-signal sessions actually inspected).
    """

    window: List[Dict[str, Any]] = sorted(
        (c for c in candles if str(c.get("date", ""))[:10] > signal_date[:10]),
        key=lambda c: str(c["date"]),
    )[:horizon_sessions]

    for index, bar in enumerate(window, start=1):
        try:
            low, high = float(bar["low"]), float(bar["high"])
        except (KeyError, TypeError, ValueError):
            continue
        if low <= magnet_strike <= high:
            return {
                "outcome": OUTCOME_HIT,
                "sessions_to_hit": index,
                "evaluated_sessions": index,
            }

    evaluated = len(window)
    return {
        "outcome": OUTCOME_MISS if evaluated >= horizon_sessions else OUTCOME_PENDING,
        "sessions_to_hit": None,
        "evaluated_sessions": evaluated,
    }


def summarize_outcomes(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate evaluated signal rows into an honest hit-rate summary.

    ``rows`` carry ``outcome``, ``sessions_to_hit`` and ``score``. The hit
    rate is computed over decided signals only (hits + misses); pending
    signals are reported but never counted toward the rate.
    """

    hits = [r for r in rows if r.get("outcome") == OUTCOME_HIT]
    misses = [r for r in rows if r.get("outcome") == OUTCOME_MISS]
    pending = [r for r in rows if r.get("outcome") == OUTCOME_PENDING]
    decided = len(hits) + len(misses)

    sessions = [r["sessions_to_hit"] for r in hits if r.get("sessions_to_hit")]
    by_score = []
    for low, high in SCORE_BUCKETS:
        bucket = [
            r
            for r in rows
            if r.get("score") is not None and low <= float(r["score"]) < high
        ]
        bucket_hits = sum(1 for r in bucket if r.get("outcome") == OUTCOME_HIT)
        bucket_decided = sum(
            1 for r in bucket if r.get("outcome") in (OUTCOME_HIT, OUTCOME_MISS)
        )
        by_score.append(
            {
                "score_min": low,
                "score_max": min(high, 120.0),
                "signals": len(bucket),
                "decided": bucket_decided,
                "hits": bucket_hits,
                "hit_rate": (bucket_hits / bucket_decided) if bucket_decided else None,
            }
        )

    return {
        "signals": len(rows),
        "decided": decided,
        "hits": len(hits),
        "misses": len(misses),
        "pending": len(pending),
        "hit_rate": (len(hits) / decided) if decided else None,
        "avg_sessions_to_hit": (sum(sessions) / len(sessions)) if sessions else None,
        "by_score": by_score,
    }


# ---------------------------------------------------------------------------
# O'Neil direction-signal outcomes (follow-through days, stock breakouts)
# ---------------------------------------------------------------------------

DIRECTION_OUTCOME_HELD = "held"
DIRECTION_OUTCOME_FAILED = "failed"
DEFAULT_DIRECTION_HORIZON = 25


def evaluate_direction_signal_outcome(
    signal_date: str,
    entry_price: float,
    invalidation_level: float,
    candles: Sequence[Dict[str, Any]],
    horizon_sessions: int = DEFAULT_DIRECTION_HORIZON,
    stop_pct: float = 8.0,
) -> Dict[str, Any]:
    """Outcome of one follow-through or breakout signal.

    Failure means a close below the signal's own invalidation level (the
    rally low for a follow-through day, the stop price for a breakout)
    within the horizon; surviving the full horizon is held. Alongside the
    verdict, the excursion numbers answer the practical question "if I had
    bought at the signal, what happened": max gain, max drawdown, and
    whether a mechanical stop ``stop_pct`` below entry would have fired.
    The signal day itself never counts.
    """

    window: List[Dict[str, Any]] = sorted(
        (c for c in candles if str(c.get("date", ""))[:10] > signal_date[:10]),
        key=lambda c: str(c["date"]),
    )[:horizon_sessions]

    max_gain = None
    max_drawdown = None
    latest_gain = None
    stop_hit = False
    failed_at = None
    stop_level = entry_price * (1.0 - stop_pct / 100.0)

    for index, bar in enumerate(window, start=1):
        try:
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            continue
        gain = (high / entry_price - 1.0) * 100.0
        drawdown = (low / entry_price - 1.0) * 100.0
        max_gain = gain if max_gain is None else max(max_gain, gain)
        max_drawdown = drawdown if max_drawdown is None else min(max_drawdown, drawdown)
        latest_gain = (close / entry_price - 1.0) * 100.0
        if low <= stop_level:
            stop_hit = True
        if failed_at is None and close < invalidation_level:
            failed_at = index

    evaluated = len(window)
    if failed_at is not None:
        outcome = DIRECTION_OUTCOME_FAILED
    elif evaluated >= horizon_sessions:
        outcome = DIRECTION_OUTCOME_HELD
    else:
        outcome = OUTCOME_PENDING
    return {
        "outcome": outcome,
        "sessions_to_fail": failed_at,
        "evaluated_sessions": evaluated,
        "max_gain_pct": round(max_gain, 2) if max_gain is not None else None,
        "max_drawdown_pct": round(max_drawdown, 2) if max_drawdown is not None else None,
        "latest_gain_pct": round(latest_gain, 2) if latest_gain is not None else None,
        "stop_hit": stop_hit if evaluated else None,
    }


def summarize_direction_outcomes(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate direction-signal outcomes, decided signals only for rates."""

    held = [r for r in rows if r.get("outcome") == DIRECTION_OUTCOME_HELD]
    failed = [r for r in rows if r.get("outcome") == DIRECTION_OUTCOME_FAILED]
    pending = [r for r in rows if r.get("outcome") == OUTCOME_PENDING]
    decided = held + failed

    def _avg(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 2) if values else None

    drawdowns = [
        float(r["max_drawdown_pct"])
        for r in decided
        if r.get("max_drawdown_pct") is not None
    ]
    gains = [
        float(r["max_gain_pct"]) for r in decided if r.get("max_gain_pct") is not None
    ]
    stops = [r for r in decided if r.get("stop_hit") is not None]
    stop_hits = sum(1 for r in stops if r["stop_hit"])
    return {
        "signals": len(rows),
        "decided": len(decided),
        "held": len(held),
        "failed": len(failed),
        "pending": len(pending),
        "hold_rate": round(len(held) / len(decided), 3) if decided else None,
        "avg_max_drawdown_pct": _avg(drawdowns),
        "avg_max_gain_pct": _avg(gains),
        "stop_hit_rate": round(stop_hits / len(stops), 3) if stops else None,
    }
