"""Snapshot-backed candidate screens with no live market-data reads."""

from __future__ import annotations

from pathlib import Path
from operator import eq, ge, gt, le, lt, ne
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd

from backend.app.config import get_settings
from quant_analysis.storage import snapshots as snapshot_store

ScreenerPreset = Literal["high_vol_oi", "unusually_bullish", "gamma_squeeze"]

_COMPARATORS: Dict[str, Callable[[Any, Any], Any]] = {
    "eq": eq,
    "ne": ne,
    "gt": gt,
    "gte": ge,
    "lt": lt,
    "lte": le,
}
_NUMERIC_OPERATORS = ["eq", "ne", "gt", "gte", "lt", "lte"]
_EQUALITY_OPERATORS = ["eq", "ne"]

FIELD_SPECS: List[Dict[str, Any]] = [
    {"name": "symbol", "label": "Symbol", "type": "string", "scopes": ["contract", "ticker"], "operators": _EQUALITY_OPERATORS},
    {"name": "expiration_date", "label": "Expiration", "type": "date", "scopes": ["contract"], "operators": _EQUALITY_OPERATORS},
    {"name": "option_type", "label": "Option side", "type": "string", "scopes": ["contract"], "operators": _EQUALITY_OPERATORS},
    {"name": "strike", "label": "Strike", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "dte", "label": "DTE", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "volume", "label": "Volume", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "open_interest", "label": "Open interest", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "volume_oi", "label": "Vol/OI", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "mid_iv", "label": "Implied volatility", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "delta", "label": "Delta", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "gamma", "label": "Gamma", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "oi_change", "label": "OI change", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "abs_oi_change", "label": "Absolute OI change", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "iv_change", "label": "IV change", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "abs_iv_change", "label": "Absolute IV change", "type": "number", "scopes": ["contract"], "operators": _NUMERIC_OPERATORS},
    {"name": "spot", "label": "Spot", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "call_volume", "label": "Call volume", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "put_volume", "label": "Put volume", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "call_oi", "label": "Call OI", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "put_oi", "label": "Put OI", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "pc_volume_ratio", "label": "Put/call volume", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "pc_oi_ratio", "label": "Put/call OI", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "atm_iv", "label": "ATM IV", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "net_gex_total", "label": "Net GEX", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "gamma_magnet_strike", "label": "Gamma magnet", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "gamma_gap_distance", "label": "Gamma-gap distance", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "gamma_gap_score", "label": "Gamma-gap score", "type": "number", "scopes": ["ticker"], "operators": _NUMERIC_OPERATORS},
    {"name": "gamma_positive_zone", "label": "Positive gamma zone", "type": "boolean", "scopes": ["ticker"], "operators": _EQUALITY_OPERATORS},
]
_FIELD_BY_NAME = {spec["name"]: spec for spec in FIELD_SPECS}

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


def get_screener_field_specs() -> Dict[str, Any]:
    return {"fields": FIELD_SPECS}


def _custom_contracts(
    symbol: str, base_dir: Path
) -> tuple[pd.DataFrame, Optional[str]]:
    history = snapshot_store.load_contract_history(symbol, base_dir)
    if history.empty:
        return pd.DataFrame(), None
    dates = sorted(history["snapshot_date"].astype(str).unique())
    latest_date = dates[-1]
    current = history[history["snapshot_date"].astype(str) == latest_date].copy()
    current["symbol"] = symbol
    if "DTE" in current:
        current["dte"] = current["DTE"]
    for column in ("volume", "open_interest", "strike", "dte", "mid_iv", "delta", "gamma"):
        if column not in current:
            current[column] = pd.NA
        current[column] = pd.to_numeric(current[column], errors="coerce")
    current["volume_oi"] = current["volume"].div(
        current["open_interest"].where(current["open_interest"] > 0)
    )
    current["oi_change"] = pd.NA
    current["iv_change"] = pd.NA
    if len(dates) >= 2 and snapshot_store.are_consecutive_market_sessions(
        dates[-2], latest_date
    ):
        previous = history[history["snapshot_date"].astype(str) == dates[-2]]
        diff = snapshot_store.compute_contract_snapshot_diff(
            previous, current, dates[-2], latest_date
        )
        current = current.drop(columns=["oi_change", "iv_change"]).merge(
            diff[snapshot_store.CONTRACT_KEY + ["oi_change", "iv_change"]],
            on=snapshot_store.CONTRACT_KEY,
            how="left",
        )
    current["abs_oi_change"] = pd.to_numeric(
        current["oi_change"], errors="coerce"
    ).abs()
    current["abs_iv_change"] = pd.to_numeric(
        current["iv_change"], errors="coerce"
    ).abs()
    return current, latest_date


def _custom_ticker_metrics(
    symbols: List[str], base_dir: Path
) -> tuple[pd.DataFrame, List[str]]:
    rows = []
    unavailable = []
    for symbol in symbols:
        metric, snapshot_date = _latest_metric(symbol, base_dir)
        if metric is None or snapshot_date is None:
            unavailable.append(symbol)
            continue
        row = metric.to_dict()
        row["symbol"] = symbol
        row["snapshot_date"] = snapshot_date
        positive_zone = row.get("gamma_positive_zone")
        row["gamma_positive_zone"] = (
            None if positive_zone is None or pd.isna(positive_zone) else bool(positive_zone)
        )
        rows.append(row)
    return pd.DataFrame(rows), unavailable


def _coerce_condition_value(spec: Dict[str, Any], value: Any) -> Any:
    if spec["type"] == "number":
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec['label']} requires a numeric value") from exc
    if spec["type"] == "boolean":
        if isinstance(value, bool):
            return value
        if str(value).lower() in {"true", "1", "yes"}:
            return True
        if str(value).lower() in {"false", "0", "no"}:
            return False
        raise ValueError(f"{spec['label']} requires true or false")
    return str(value).lower() if spec["name"] in {"symbol", "option_type"} else str(value)


def _apply_conditions(
    frame: pd.DataFrame, scope: str, conditions: List[Any]
) -> pd.DataFrame:
    work = frame.copy()
    for condition in conditions:
        spec = _FIELD_BY_NAME.get(condition.field)
        if not spec or scope not in spec["scopes"]:
            raise ValueError(f"Unsupported {scope} field: {condition.field}")
        if condition.operator not in spec["operators"]:
            raise ValueError(
                f"Operator {condition.operator} is not valid for {condition.field}"
            )
        if condition.field not in work:
            work[condition.field] = pd.NA
        value = _coerce_condition_value(spec, condition.value)
        series = work[condition.field]
        available = series.notna()
        if spec["type"] == "number":
            series = pd.to_numeric(series, errors="coerce")
        elif spec["type"] == "boolean":
            series = series.astype("boolean")
        elif spec["name"] in {"symbol", "option_type"}:
            series = series.astype(str).str.lower()
        mask = _COMPARATORS[condition.operator](series, value)
        work = work[(mask & available).fillna(False)]
    return work


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def run_custom_screener(request: Any) -> Dict[str, Any]:
    """Apply a whitelisted AND-filter query to the latest persisted rows."""

    configured = _configured_symbols()
    requested = (
        list(dict.fromkeys(symbol.upper() for symbol in request.symbols))
        if request.symbols
        else configured
    )
    selected = [symbol for symbol in requested if symbol in configured]
    unavailable = [symbol for symbol in requested if symbol not in configured]
    base_dir = _snapshot_dir()

    if request.scope == "contract":
        frames = []
        for symbol in selected:
            frame, snapshot_date = _custom_contracts(symbol, base_dir)
            if frame.empty or snapshot_date is None:
                unavailable.append(symbol)
                continue
            frames.append(frame)
        work = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        work, missing = _custom_ticker_metrics(selected, base_dir)
        unavailable.extend(missing)

    source_dates = [
        str(value)
        for value in work.get("snapshot_date", pd.Series(dtype=str)).dropna().unique()
    ]
    work = _apply_conditions(work, request.scope, request.conditions)
    if request.sort:
        sort_spec = _FIELD_BY_NAME.get(request.sort.field)
        if not sort_spec or request.scope not in sort_spec["scopes"]:
            raise ValueError(f"Unsupported sort field: {request.sort.field}")
        if request.sort.field not in work:
            work[request.sort.field] = pd.NA
        work = work.sort_values(
            request.sort.field,
            ascending=request.sort.direction == "asc",
            na_position="last",
        )

    allowed_fields = [
        spec["name"] for spec in FIELD_SPECS if request.scope in spec["scopes"]
    ]
    rows = []
    for _, row in work.head(request.limit).iterrows():
        symbol = str(row.get("symbol", "")).upper()
        snapshot_date = str(row.get("snapshot_date", ""))
        if request.scope == "contract":
            row_key = "|".join(
                [
                    symbol,
                    str(row.get("expiration_date", "")),
                    str(row.get("strike", "")),
                    str(row.get("option_type", "")),
                ]
            )
        else:
            row_key = symbol
        rows.append(
            {
                "row_key": row_key,
                "symbol": symbol,
                "snapshot_date": snapshot_date,
                "values": {
                    field: _json_value(row.get(field))
                    for field in allowed_fields
                    if field in row.index
                },
            }
        )

    as_of = max(source_dates) if source_dates else None
    current_date = snapshot_store.last_trading_date()
    return {
        "scope": request.scope,
        "as_of": as_of,
        "stale": not source_dates
        or bool(unavailable)
        or any(value != current_date for value in source_dates),
        "unavailable_symbols": list(dict.fromkeys(unavailable)),
        "methodology": (
            "Filtered from the latest persisted option snapshots. No live chain scan was performed."
        ),
        "rows": rows,
    }


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
    today = snapshot_store.last_trading_date()
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
