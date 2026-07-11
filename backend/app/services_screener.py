"""Snapshot-backed candidate screens with no live market-data reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from backend.app.config import get_settings
from quant_analysis.storage import snapshots as snapshot_store

ScreenerPreset = Literal["high_vol_oi", "unusually_bullish", "gamma_squeeze"]

_METHODOLOGY = {
    "high_vol_oi": (
        "Contract candidates with volume divided by open interest of at least 1.0, "
        "ranked by that persisted ratio."
    ),
    "unusually_bullish": (
        "Call-contract candidates with volume divided by open interest of at least 1.0, "
        "ranked by that persisted ratio. Call activity does not establish customer intent."
    ),
    "gamma_squeeze": (
        "Ticker candidates with a persisted positive local gamma zone, an above-spot "
        "gamma magnet, and gamma-gap score of at least 50. This is not a squeeze forecast."
    ),
}


def _configured_symbols() -> List[str]:
    return list(dict.fromkeys(symbol.upper() for symbol in get_settings().snapshot_tickers))


def _snapshot_dir() -> Path:
    return Path(get_settings().snapshot_dir)


def _number(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _latest_contracts(symbol: str, base_dir: Path) -> tuple[pd.DataFrame, Optional[str]]:
    history = snapshot_store.load_contract_history(symbol, base_dir)
    if history.empty:
        return pd.DataFrame(), None
    snapshot_date = str(max(history["snapshot_date"].unique()))
    return history[history["snapshot_date"] == snapshot_date].copy(), snapshot_date


def _latest_metric(symbol: str, base_dir: Path) -> tuple[Optional[pd.Series], Optional[str]]:
    metrics = snapshot_store.load_daily_metrics(base_dir, symbol)
    if metrics.empty or "date" not in metrics:
        return None, None
    latest_date = str(metrics["date"].max())
    return metrics[metrics["date"].astype(str) == latest_date].iloc[-1], latest_date


def _contract_rows(
    symbol: str,
    contracts: pd.DataFrame,
    snapshot_date: str,
    preset: ScreenerPreset,
    min_vol_oi: Optional[float],
    min_open_interest: Optional[float],
) -> List[Dict[str, Any]]:
    work = contracts.copy()
    for column in ("volume", "open_interest"):
        if column not in work:
            work[column] = pd.NA
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["volume_oi"] = work["volume"].div(work["open_interest"].where(work["open_interest"] > 0))

    required_vol_oi = max(1.0, min_vol_oi or 0.0)
    work = work[work["volume_oi"] >= required_vol_oi]
    if min_open_interest is not None:
        work = work[work["open_interest"] >= min_open_interest]
    if preset == "unusually_bullish":
        work = work[work["option_type"].astype(str).str.lower() == "call"]

    rows = []
    for _, row in work.sort_values(["volume_oi", "volume"], ascending=False).iterrows():
        option_type = str(row.get("option_type", "")).lower()
        if option_type not in {"call", "put"}:
            continue
        rows.append(
            {
                "symbol": symbol,
                "snapshot_date": snapshot_date,
                "expiration_date": str(row["expiration_date"]),
                "strike": _number(row.get("strike")),
                "option_type": option_type,
                "volume": _number(row.get("volume")),
                "open_interest": _number(row.get("open_interest")),
                "volume_oi": _number(row.get("volume_oi")),
            }
        )
    return rows


def _gamma_row(
    symbol: str,
    metric: pd.Series,
    snapshot_date: str,
    min_vol_oi: Optional[float],
    min_open_interest: Optional[float],
) -> Optional[Dict[str, Any]]:
    score = _number(metric.get("gamma_gap_score"))
    distance = _number(metric.get("gamma_gap_distance"))
    positive_zone = _number(metric.get("gamma_positive_zone"))
    call_volume = _number(metric.get("call_volume")) or 0.0
    put_volume = _number(metric.get("put_volume")) or 0.0
    call_oi = _number(metric.get("call_oi")) or 0.0
    put_oi = _number(metric.get("put_oi")) or 0.0
    open_interest = call_oi + put_oi
    volume_oi = (call_volume + put_volume) / open_interest if open_interest > 0 else None

    if score is None or score < 50 or distance is None or distance <= 0 or positive_zone != 1:
        return None
    if min_vol_oi is not None and (volume_oi is None or volume_oi < min_vol_oi):
        return None
    if min_open_interest is not None and open_interest < min_open_interest:
        return None
    return {
        "symbol": symbol,
        "snapshot_date": snapshot_date,
        "volume": call_volume + put_volume,
        "open_interest": open_interest,
        "volume_oi": volume_oi,
        "spot": _number(metric.get("spot")),
        "net_gex_total": _number(metric.get("net_gex_total")),
        "gamma_magnet_strike": _number(metric.get("gamma_magnet_strike")),
        "gamma_gap_distance": distance,
        "gamma_gap_score": score,
        "gamma_positive_zone": True,
    }


def get_screener(
    preset: ScreenerPreset,
    symbols: Optional[List[str]],
    min_vol_oi: Optional[float],
    min_open_interest: Optional[float],
    limit: int,
) -> Dict[str, Any]:
    """Return candidates from persisted snapshots in the configured universe only."""

    configured = _configured_symbols()
    requested = list(dict.fromkeys(symbol.upper() for symbol in symbols)) if symbols else configured
    selected = [symbol for symbol in requested if symbol in configured]
    unavailable = [symbol for symbol in requested if symbol not in configured]
    observed_dates: List[str] = []
    rows: List[Dict[str, Any]] = []
    base_dir = _snapshot_dir()

    for symbol in selected:
        if preset == "gamma_squeeze":
            metric, snapshot_date = _latest_metric(symbol, base_dir)
            if metric is None or snapshot_date is None:
                unavailable.append(symbol)
                continue
            observed_dates.append(snapshot_date)
            row = _gamma_row(symbol, metric, snapshot_date, min_vol_oi, min_open_interest)
            if row:
                rows.append(row)
            continue

        contracts, snapshot_date = _latest_contracts(symbol, base_dir)
        if contracts.empty or snapshot_date is None:
            unavailable.append(symbol)
            continue
        observed_dates.append(snapshot_date)
        rows.extend(
            _contract_rows(
                symbol, contracts, snapshot_date, preset, min_vol_oi, min_open_interest
            )
        )

    sort_key = "gamma_gap_score" if preset == "gamma_squeeze" else "volume_oi"
    rows.sort(key=lambda row: row.get(sort_key) or 0.0, reverse=True)
    as_of = max(observed_dates) if observed_dates else None
    today = snapshot_store.market_date_today()
    return {
        "preset": preset,
        "as_of": as_of,
        "stale": not observed_dates
        or bool(unavailable)
        or any(snapshot_date != today for snapshot_date in observed_dates),
        "unavailable_symbols": unavailable,
        "methodology": _METHODOLOGY[preset],
        "rows": rows[:limit],
    }
