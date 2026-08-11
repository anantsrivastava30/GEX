"""Tests for gamma-gap realized-outcome scoring."""

import pytest

from quant_analysis.analytics.track_record import (
    evaluate_gamma_gap_outcome,
    summarize_outcomes,
)


def bar(date, low, high):
    return {"date": date, "low": low, "high": high, "open": low, "close": high}


CANDLES = [
    bar("2026-07-06", 98.0, 101.0),
    bar("2026-07-07", 99.0, 102.0),
    bar("2026-07-08", 100.0, 103.0),
    bar("2026-07-09", 101.0, 104.0),
    bar("2026-07-10", 102.0, 105.0),
    bar("2026-07-13", 103.0, 106.0),
    bar("2026-07-14", 104.0, 107.0),
]


def test_hit_counts_sessions_after_signal_day():
    # Magnet 104 first trades on 07-09 (2 sessions after the 07-07 signal).
    result = evaluate_gamma_gap_outcome("2026-07-07", 104.0, CANDLES)
    assert result["outcome"] == "hit"
    assert result["sessions_to_hit"] == 2


def test_signal_day_bar_never_counts():
    # Magnet 102 was inside the signal day's own range (07-07: 99-102) but
    # must only count from the next session (07-08: 100-103) onward.
    result = evaluate_gamma_gap_outcome("2026-07-07", 102.0, CANDLES)
    assert result["outcome"] == "hit"
    assert result["sessions_to_hit"] == 1


def test_miss_when_horizon_exhausted():
    # Magnet 90 never trades in the five sessions after 07-06.
    result = evaluate_gamma_gap_outcome("2026-07-06", 90.0, CANDLES)
    assert result["outcome"] == "miss"
    assert result["sessions_to_hit"] is None
    assert result["evaluated_sessions"] == 5


def test_pending_when_not_enough_sessions():
    # Only one post-signal session exists after 07-13.
    result = evaluate_gamma_gap_outcome("2026-07-13", 90.0, CANDLES)
    assert result["outcome"] == "pending"
    assert result["evaluated_sessions"] == 1

    # No post-signal sessions at all.
    result = evaluate_gamma_gap_outcome("2026-07-14", 104.0, CANDLES)
    assert result["outcome"] == "pending"
    assert result["evaluated_sessions"] == 0


def test_candle_order_does_not_matter():
    shuffled = list(reversed(CANDLES))
    result = evaluate_gamma_gap_outcome("2026-07-07", 104.0, shuffled)
    assert result["outcome"] == "hit"
    assert result["sessions_to_hit"] == 2


def test_summary_hit_rate_excludes_pending():
    rows = [
        {"outcome": "hit", "sessions_to_hit": 2, "score": 90.0},
        {"outcome": "hit", "sessions_to_hit": 4, "score": 55.0},
        {"outcome": "miss", "sessions_to_hit": None, "score": 20.0},
        {"outcome": "pending", "sessions_to_hit": None, "score": 100.0},
    ]
    summary = summarize_outcomes(rows)
    assert summary["signals"] == 4
    assert summary["decided"] == 3
    assert summary["pending"] == 1
    assert summary["hit_rate"] == pytest.approx(2 / 3)
    assert summary["avg_sessions_to_hit"] == pytest.approx(3.0)

    top_bucket = summary["by_score"][2]  # score >= 35
    assert top_bucket["signals"] == 3  # score 90 hit + 55 hit + 100 pending
    assert top_bucket["decided"] == 2
    assert top_bucket["hit_rate"] == pytest.approx(1.0)


def test_summary_empty():
    summary = summarize_outcomes([])
    assert summary["hit_rate"] is None
    assert summary["decided"] == 0
