import pandas as pd
import pytest

from quant_analysis.storage.snapshots import (
    build_snapshot,
    compute_iv_rank,
    compute_oi_change,
    load_contract_history,
    load_daily_metrics,
    upsert_daily_metrics,
    write_snapshot,
)


def make_contracts(spot=100.0, oi_bump=0):
    """Synthetic frame shaped like market_data.load_options_data output."""

    rows = []
    for exp, dte in (("2026-08-21", 7), ("2026-09-18", 14)):
        for strike in (95.0, 100.0, 105.0):
            for option_type in ("call", "put"):
                # Asymmetric call/put OI so net GEX per strike is non-zero.
                oi = int(strike * (10 if option_type == "call" else 5)) + oi_bump
                gamma = 0.05 if strike == 100.0 else 0.02
                gex = gamma * oi * 100 * (-1 if option_type == "put" else 1)
                rows.append(
                    {
                        "expiration_date": exp,
                        "strike": strike,
                        "option_type": option_type,
                        "open_interest": oi,
                        "volume": 500 if option_type == "call" else 250,
                        "last": 1.5,
                        "bid": 1.4,
                        "ask": 1.6,
                        "contract_size": 100,
                        "delta": 0.5 if option_type == "call" else -0.5,
                        "gamma": gamma,
                        "theta": -0.01,
                        "vega": 0.1,
                        "mid_iv": 0.20 + (0.01 if dte == 7 else 0.02),
                        "smv_vol": 0.21,
                        "DTE": dte,
                        "GammaExposure": gex,
                        "DeltaExposure": 0.5 * oi * 100,
                    }
                )
    return pd.DataFrame(rows)


def test_build_snapshot_derives_metrics():
    snapshot = build_snapshot("SPY", "2026-07-09", 100.0, make_contracts())

    assert snapshot is not None
    metrics = snapshot["metrics"]
    assert metrics["ticker"] == "SPY"
    assert metrics["num_contracts"] == 12
    assert metrics["call_volume"] == 6 * 500
    assert metrics["put_volume"] == 6 * 250
    assert metrics["pc_volume_ratio"] == pytest.approx(0.5)
    # ATM IV comes from the nearest expiration at the strike nearest spot.
    assert metrics["atm_iv"] == pytest.approx(0.21)
    # Calls carry double the put OI, so the net-GEX magnet sits at the
    # high-gamma 100 strike.
    assert metrics["gamma_magnet_strike"] == pytest.approx(100.0)
    assert 0 <= metrics["gamma_gap_score"] <= 120
    # Only persisted columns are kept.
    assert "symbol" not in snapshot["contracts"].columns
    assert "open_interest" in snapshot["contracts"].columns


def test_build_snapshot_empty_returns_none():
    assert build_snapshot("SPY", "2026-07-09", 100.0, pd.DataFrame()) is None


def test_write_and_load_roundtrip(tmp_path):
    snapshot = build_snapshot("SPY", "2026-07-08", 100.0, make_contracts())
    path = write_snapshot(snapshot, tmp_path)

    assert path == tmp_path / "2026-07-08" / "SPY.csv.gz"
    history = load_contract_history("SPY", tmp_path)
    assert len(history) == 12
    assert set(history["snapshot_date"]) == {"2026-07-08"}

    metrics = load_daily_metrics(tmp_path, "SPY")
    assert len(metrics) == 1
    assert metrics.iloc[0]["date"] == "2026-07-08"


def test_upsert_daily_metrics_replaces_same_day(tmp_path):
    upsert_daily_metrics(
        [{"date": "2026-07-08", "ticker": "SPY", "atm_iv": 0.20}], tmp_path
    )
    upsert_daily_metrics(
        [
            {"date": "2026-07-08", "ticker": "SPY", "atm_iv": 0.25},
            {"date": "2026-07-08", "ticker": "QQQ", "atm_iv": 0.30},
        ],
        tmp_path,
    )

    metrics = load_daily_metrics(tmp_path)
    assert len(metrics) == 2
    spy = metrics[metrics["ticker"] == "SPY"]
    assert spy.iloc[0]["atm_iv"] == pytest.approx(0.25)


def test_compute_oi_change_between_days(tmp_path):
    day1 = build_snapshot("SPY", "2026-07-07", 100.0, make_contracts())
    day2 = build_snapshot("SPY", "2026-07-08", 100.0, make_contracts(oi_bump=50))
    write_snapshot(day1, tmp_path)
    write_snapshot(day2, tmp_path)

    changes = compute_oi_change("SPY", tmp_path)
    assert not changes.empty
    assert (changes["oi_change"] == 50).all()
    assert changes.iloc[0]["from_date"] == "2026-07-07"
    assert changes.iloc[0]["to_date"] == "2026-07-08"


def test_compute_oi_change_needs_two_days(tmp_path):
    write_snapshot(build_snapshot("SPY", "2026-07-08", 100.0, make_contracts()), tmp_path)
    assert compute_oi_change("SPY", tmp_path).empty


def test_compute_iv_rank(tmp_path):
    rows = [
        {"date": f"2026-07-0{i}", "ticker": "SPY", "atm_iv": iv}
        for i, iv in enumerate((0.10, 0.30, 0.20), start=1)
    ]
    upsert_daily_metrics(rows, tmp_path)

    rank = compute_iv_rank("SPY", tmp_path)
    assert rank is not None
    assert rank["iv"] == pytest.approx(0.20)
    assert rank["iv_rank"] == pytest.approx(50.0)
    assert rank["iv_percentile"] == pytest.approx(100 / 3)
    assert rank["days_of_history"] == 3


def test_compute_iv_rank_insufficient_history(tmp_path):
    upsert_daily_metrics(
        [{"date": "2026-07-08", "ticker": "SPY", "atm_iv": 0.2}], tmp_path
    )
    assert compute_iv_rank("SPY", tmp_path) is None
