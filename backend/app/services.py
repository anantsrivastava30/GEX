"""Service layer: cached, rate-limited adapters over quant_analysis."""

from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import HTTPException

from quant_analysis.analytics.greeks import (
    compute_max_pain,
    compute_vanna_charm_exposures,
    get_risk_free_rate,
)
from quant_analysis.analytics.track_record import (
    evaluate_gamma_gap_outcome,
    summarize_outcomes,
)
from quant_analysis.analytics.visualization import (
    compute_delta_exposure_series,
    compute_gamma_gap_metrics,
    compute_net_gamma_exposure,
    describe_gamma_gap,
    generate_binomial_tree,
    get_intraday_prices_with_prev_close,
    interpret_net_gex,
)
from quant_analysis.services.market_data import (
    compute_iv_skew,
    compute_put_call_ratios,
    compute_term_structure,
    compute_unusual_spikes,
    fetch_and_filter_rss,
    get_bond_yield_info,
    get_futures_quotes,
    get_vix_info,
    process_options_data,
)
from quant_analysis.storage import snapshots as snapshot_store
from quant_analysis.storage.db import load_gamma_gap_history

from backend.app.cache import cache
from backend.app.config import get_settings
from backend.app.deps import get_tradier, tradier_call

# Cache TTLs (seconds) — see docs/UW_PARITY_PLAN.md.
TTL_QUOTE = 15
TTL_CHAIN = 60
TTL_EXPIRATIONS = 24 * 3600
TTL_MACRO = 300
TTL_NEWS = 600
TTL_HISTORY = 3600
TTL_RISK_FREE = 6 * 3600


def _quote(symbol: str) -> Dict[str, Any]:
    api = get_tradier()
    return cache.get_or_compute(
        f"quote:{symbol}", TTL_QUOTE, lambda: tradier_call(api.quote, symbol)
    )


def _chain(symbol: str, expiration: str) -> List[Dict[str, Any]]:
    api = get_tradier()
    return cache.get_or_compute(
        f"chain:{symbol}:{expiration}",
        TTL_CHAIN,
        lambda: tradier_call(api.option_chain, symbol, expiration),
    )


def _spot(symbol: str) -> float:
    last = _quote(symbol).get("last")
    if last is None:
        raise HTTPException(status_code=404, detail=f"No quote for {symbol}")
    return float(last)


def get_expirations(symbol: str) -> List[str]:
    api = get_tradier()
    return cache.get_or_compute(
        f"expirations:{symbol}",
        TTL_EXPIRATIONS,
        lambda: tradier_call(api.expirations, symbol),
    )


def get_ticker_snapshot(symbol: str) -> Dict[str, Any]:
    q = _quote(symbol)
    if not q:
        raise HTTPException(status_code=404, detail=f"No quote for {symbol}")
    return {"symbol": symbol.upper(), **q}


def get_gex_profile(symbol: str, expiration: str, offset: int) -> Dict[str, Any]:
    spot = _spot(symbol)
    chain = _chain(symbol, expiration)
    df_net = compute_net_gamma_exposure(chain, spot, offset=offset)
    if df_net.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No option data for {symbol} {expiration} within ±{offset}",
        )

    df_net = df_net.rename(columns={"Net GEX": "GEX"})
    gap = compute_gamma_gap_metrics(df_net, spot, offset=offset)
    gap_payload = None
    if gap:
        gap_payload = {
            key: gap[key]
            for key in (
                "magnet_strike",
                "magnet_gex",
                "distance",
                "distance_pct",
                "score",
                "positive_zone",
                "band_low",
                "band_high",
            )
        }
        gap_payload["commentary"] = describe_gamma_gap(gap)

    return {
        "symbol": symbol.upper(),
        "expiration": expiration,
        "spot": spot,
        "offset": offset,
        "profile": [
            {"strike": float(row["Strike"]), "net_gex": float(row["GEX"])}
            for _, row in df_net.iterrows()
        ],
        "interpretation": interpret_net_gex(df_net, spot, offset=offset),
        "gamma_gap": gap_payload,
    }


def get_skew(symbol: str, expiration: str) -> Dict[str, Any]:
    chain = _chain(symbol, expiration)
    if not chain:
        raise HTTPException(status_code=404, detail=f"No chain for {symbol} {expiration}")
    skew = compute_iv_skew(chain)
    return {
        "symbol": symbol.upper(),
        "expiration": expiration,
        "points": skew.rename(columns={"strike": "strike"}).to_dict(orient="records"),
    }


def _load_chain_df(symbol: str, expirations: List[str]) -> pd.DataFrame:
    frames = []
    for expiration in expirations:
        chain = _chain(symbol, expiration)
        if chain:
            df = pd.DataFrame(chain)
            if "greeks" in df.columns:
                df = pd.concat(
                    [df.drop(columns=["greeks"]), pd.json_normalize(df["greeks"])],
                    axis=1,
                )
            frames.append(df)
    if not frames:
        raise HTTPException(
            status_code=404, detail=f"No option data for {symbol} {expirations}"
        )
    return pd.concat(frames, ignore_index=True)


def get_ratios(symbol: str, expirations: List[str]) -> Dict[str, Any]:
    df = _load_chain_df(symbol, expirations)
    vol_ratio, oi_ratio = compute_put_call_ratios(df)
    return {
        "symbol": symbol.upper(),
        "expirations": expirations,
        "pc_volume_ratio": float(vol_ratio),
        "pc_oi_ratio": float(oi_ratio),
    }


def get_candles(symbol: str, days: int) -> List[Dict[str, Any]]:
    """Daily OHLC bars from Tradier for the candlestick chart."""

    api = get_tradier()
    end = date.today()
    start = end - timedelta(days=days)

    def compute() -> List[Dict[str, Any]]:
        rows = tradier_call(
            api.history, symbol, "daily", start.isoformat(), end.isoformat()
        )
        out: List[Dict[str, Any]] = []
        for r in rows or []:
            try:
                out.append(
                    {
                        "date": str(r["date"]),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "volume": int(r.get("volume") or 0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return out

    return cache.get_or_compute(f"candles:{symbol}:{days}", TTL_HISTORY, compute)


def get_unusual(symbol: str, expirations: List[str], top_n: int) -> Dict[str, Any]:
    df = _load_chain_df(symbol, expirations)
    spikes = compute_unusual_spikes(df, top_n=top_n)
    return {
        "symbol": symbol.upper(),
        "expirations": expirations,
        "rows": spikes.where(pd.notnull(spikes), None).to_dict(orient="records"),
    }


def _configured_snapshot_tickers() -> List[str]:
    """Symbols covered by configured and shared-watchlist snapshots."""

    from backend.app.services_watchlists import snapshot_symbols

    return snapshot_symbols()


def _cached_contract_proxy_data() -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load latest cached contracts and their preceding daily snapshot, if any.

    This intentionally never uses the live-chain service. The scheduler and
    snapshot CLI are the only producers of data consumed by flow endpoints.
    """

    base_dir = _snapshot_dir()
    today = snapshot_store.last_trading_date()
    frames = []
    unavailable_tickers = []
    unavailable_history_tickers = []
    observed_dates = []

    for ticker in _configured_snapshot_tickers():
        history = snapshot_store.load_contract_history(ticker, base_dir)
        if history.empty:
            unavailable_tickers.append(ticker)
            continue

        dates = sorted(history["snapshot_date"].unique())
        latest_date = dates[-1]
        observed_dates.append(latest_date)
        latest = history[history["snapshot_date"] == latest_date].copy()
        latest["ticker"] = ticker

        if len(dates) < 2 or not snapshot_store.are_consecutive_market_sessions(
            str(dates[-2]), str(latest_date)
        ):
            unavailable_history_tickers.append(ticker)
            latest["open_interest_prev"] = pd.NA
            latest["oi_change"] = pd.NA
            latest["mid_iv_prev"] = pd.NA
            latest["iv_change"] = pd.NA
            latest["history_available"] = False
        else:
            previous_date = dates[-2]
            previous = history[history["snapshot_date"] == previous_date]
            latest = snapshot_store.compute_contract_snapshot_diff(
                previous, latest, previous_date, latest_date
            )
            latest["ticker"] = ticker
            latest["snapshot_date"] = latest_date
            latest["history_available"] = True
        frames.append(latest)

    as_of = max(observed_dates) if observed_dates else None
    stale = not observed_dates or bool(unavailable_tickers) or any(
        observed_date != today for observed_date in observed_dates
    )
    status = {
        "as_of": as_of,
        "stale": stale,
        "unavailable_history": bool(unavailable_tickers or unavailable_history_tickers),
        "unavailable_tickers": unavailable_tickers,
        "unavailable_history_tickers": unavailable_history_tickers,
    }
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), status)


def _rank_proxy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rank cached anomalies without inferring trade execution or direction."""

    ranked = df.copy()
    def numeric_column(column: str) -> pd.Series:
        if column not in ranked:
            return pd.Series(pd.NA, index=ranked.index, dtype="float64")
        return pd.to_numeric(ranked[column], errors="coerce")

    oi = numeric_column("open_interest")
    volume = numeric_column("volume").fillna(0)
    ranked["volume_oi"] = volume.div(oi.where(oi > 0))
    oi_change = numeric_column("oi_change").abs()
    iv_change = numeric_column("iv_change").abs()
    ranked["score"] = (
        ranked["volume_oi"].fillna(0).rank(pct=True) * 100
        + oi_change.fillna(0).rank(pct=True) * 100
        + iv_change.fillna(0).rank(pct=True) * 100
    )
    return ranked.sort_values("score", ascending=False).reset_index(drop=True)


def _json_number(value: Any) -> Optional[float]:
    return None if pd.isna(value) else float(value)


def get_cached_flow_feed(
    limit: int,
    ticker: Optional[str] = None,
    expiration_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Cached multi-ticker contract ranking for the proxy flow page."""

    contracts, status = _cached_contract_proxy_data()
    if contracts.empty:
        return {**status, "rows": []}

    ranked = _rank_proxy_rows(contracts)
    if ticker:
        ranked = ranked[ranked["ticker"] == ticker.upper()]
    if expiration_date:
        ranked = ranked[ranked["expiration_date"] == expiration_date]
    ranked = ranked.head(limit)
    rows = []
    for _, row in ranked.iterrows():
        rows.append(
            {
                "ticker": str(row["ticker"]),
                "expiration_date": str(row["expiration_date"]),
                "strike": float(row["strike"]),
                "option_type": str(row["option_type"]),
                "snapshot_date": str(row["snapshot_date"]),
                "volume": _json_number(row.get("volume")),
                "open_interest": _json_number(row.get("open_interest")),
                "volume_oi": _json_number(row.get("volume_oi")),
                "oi_change": _json_number(row.get("oi_change")),
                "mid_iv": _json_number(row.get("mid_iv")),
                "iv_change": _json_number(row.get("iv_change")),
                "history_available": bool(row["history_available"]),
                "score": float(row["score"]),
            }
        )
    return {**status, "rows": rows}


def get_hottest_chains(limit: int) -> Dict[str, Any]:
    """Cached chain-level ranking over the scheduled snapshot universe."""

    contracts, status = _cached_contract_proxy_data()
    if contracts.empty:
        return {**status, "rows": []}

    contracts = _rank_proxy_rows(contracts)
    grouped = contracts.groupby(
        ["ticker", "expiration_date", "snapshot_date", "history_available"], dropna=False
    ).agg(
        contracts=("strike", "size"),
        total_volume=("volume", "sum"),
        total_open_interest=("open_interest", "sum"),
        oi_change=("oi_change", lambda values: values.sum(min_count=1)),
        iv_change=("iv_change", "mean"),
    ).reset_index()
    grouped["volume_oi"] = grouped["total_volume"].div(
        grouped["total_open_interest"].where(grouped["total_open_interest"] > 0)
    )
    ranked = _rank_proxy_rows(
        grouped.rename(
            columns={"total_volume": "volume", "total_open_interest": "open_interest"}
        )
    ).head(limit)
    rows = []
    for _, row in ranked.iterrows():
        rows.append(
            {
                "ticker": str(row["ticker"]),
                "expiration_date": str(row["expiration_date"]),
                "snapshot_date": str(row["snapshot_date"]),
                "contracts": int(row["contracts"]),
                "total_volume": float(row["volume"]),
                "total_open_interest": float(row["open_interest"]),
                "volume_oi": _json_number(row.get("volume_oi")),
                "oi_change": _json_number(row.get("oi_change")),
                "iv_change": _json_number(row.get("iv_change")),
                "history_available": bool(row["history_available"]),
                "score": float(row["score"]),
            }
        )
    return {**status, "rows": rows}


def _risk_free_rate() -> float:
    return cache.get_or_compute("risk_free_rate", TTL_RISK_FREE, get_risk_free_rate)


def _positioning_frame(symbol: str, expirations: List[str]) -> pd.DataFrame:
    """Flattened chain frame with exposures from cached per-expiration chains.

    Copies each contract dict before tagging ``expiration_date`` so the
    cached chain lists are never mutated.
    """

    all_opts: List[Dict[str, Any]] = []
    for expiration in expirations:
        chain = _chain(symbol, expiration)
        for opt in chain or []:
            all_opts.append({**opt, "expiration_date": expiration})
    df = process_options_data(all_opts)
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No option data for {symbol} {expirations}"
        )
    return df


def get_exposure(symbol: str, expirations: List[str], offset: int) -> Dict[str, Any]:
    """Per-strike net vanna/charm exposure (locally derived Black-Scholes)."""

    spot = _spot(symbol)
    df = _positioning_frame(symbol, expirations)
    rate = _risk_free_rate()
    out = compute_vanna_charm_exposures(df, spot, r=rate, offset=offset)
    if out.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No usable IV data for {symbol} within ±{offset} of spot.",
        )
    return {
        "symbol": symbol.upper(),
        "expirations": expirations,
        "spot": spot,
        "offset": offset,
        "risk_free_rate": rate,
        "points": [
            {
                "strike": float(row["strike"]),
                "vanna": float(row["vanna_exposure"]),
                "charm": float(row["charm_exposure"]),
            }
            for _, row in out.iterrows()
        ],
    }


def get_max_pain(symbol: str, expiration: str) -> Dict[str, Any]:
    spot = _spot(symbol)
    chain = _chain(symbol, expiration)
    if not chain:
        raise HTTPException(status_code=404, detail=f"No chain for {symbol} {expiration}")
    result = compute_max_pain(pd.DataFrame(chain))
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No open interest for {symbol} {expiration} yet.",
        )
    return {
        "symbol": symbol.upper(),
        "expiration": expiration,
        "spot": spot,
        **result,
    }


def get_term_structure(symbol: str, num_expirations: int) -> Dict[str, Any]:
    spot = _spot(symbol)
    expirations = get_expirations(symbol)[:num_expirations]
    chains = {exp: _chain(symbol, exp) for exp in expirations}
    ts = compute_term_structure(chains, spot)
    if ts.empty:
        raise HTTPException(
            status_code=404, detail=f"No IV data for {symbol} term structure."
        )
    ts = ts.where(pd.notnull(ts), None)
    return {
        "symbol": symbol.upper(),
        "spot": spot,
        "points": ts.to_dict(orient="records"),
    }


def _market_iv(chain: List[Dict[str, Any]], strike: float, option_type: str) -> Optional[float]:
    """Return a usable selected-contract IV, preferring Tradier's mid IV."""

    for contract in chain:
        try:
            contract_strike = float(contract["strike"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isclose(contract_strike, strike, rel_tol=0.0, abs_tol=1e-8):
            continue
        if str(contract.get("option_type", "")).lower() != option_type:
            continue
        greeks = contract.get("greeks") or {}
        for field in ("mid_iv", "smv_vol", "iv"):
            try:
                iv = float(greeks[field])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(iv) and iv > 0:
                return iv
    return None


def get_binomial_tree(
    symbol: str,
    expiration: str,
    strike: float,
    option_type: str,
    steps: int,
    days_to_exp: int,
    rate_override: Optional[float],
    iv_override: Optional[float],
) -> Dict[str, Any]:
    """Price a CRR tree from cached market calibration or supplied overrides."""

    spot = _spot(symbol)
    rate_source = "override" if rate_override is not None else "market"
    rate = rate_override if rate_override is not None else _risk_free_rate()

    iv_source = "override" if iv_override is not None else "market"
    iv = iv_override
    if iv is None:
        iv = _market_iv(_chain(symbol, expiration), strike, option_type)
    if iv is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No usable implied volatility for {symbol} {expiration} "
                f"{strike:g} {option_type}; provide iv_override."
            ),
        )

    rate = float(rate)
    iv = float(iv)
    if not math.isfinite(rate) or not math.isfinite(iv):
        raise HTTPException(status_code=422, detail="Rate and IV must be finite values.")

    maturity = days_to_exp / 365.0
    dt = maturity / steps
    up = math.exp(iv * math.sqrt(dt))
    probability = (math.exp(rate * dt) - (1.0 / up)) / (up - (1.0 / up))
    if not 0.0 <= probability <= 1.0:
        raise HTTPException(
            status_code=422,
            detail="Rate, IV, days_to_exp, and steps produce an invalid CRR probability.",
        )

    tree = generate_binomial_tree(
        spot, strike, maturity, rate, iv, steps, option_type
    )
    return {
        "symbol": symbol.upper(),
        "expiration": expiration,
        "spot": spot,
        "strike": strike,
        "option_type": option_type,
        "steps": steps,
        "days_to_exp": days_to_exp,
        "calibration": {
            "risk_free_rate": rate,
            "risk_free_rate_source": rate_source,
            "implied_volatility": iv,
            "implied_volatility_source": iv_source,
        },
        "nodes": [
            {
                "step": int(row["step"]),
                "node": int(row["node"]),
                "price": float(row["price"]),
                "option": float(row["option"]),
            }
            for _, row in tree.iterrows()
        ],
    }


def get_gamma_gap_history(ticker: Optional[str], limit: int) -> List[Dict[str, Any]]:
    rows = load_gamma_gap_history(limit=limit if not ticker else max(limit * 10, limit))
    if ticker:
        upper = ticker.upper()
        rows = [r for r in rows if str(r.get("ticker", "")).upper() == upper][:limit]
    for row in rows:
        row.pop("payload", None)
    return rows


def get_delta_projection(symbol: str, expiration: str, offset: int) -> Dict[str, Any]:
    """Intraday spot path + per-bar net dealer delta exposure + EOD trend.

    Resurrects the legacy ``plot_price_and_delta_projection`` data pipeline:
    yfinance 30-minute bars (prev close prefixed) and ONE cached chain
    snapshot, with the exposure window sliding along the price path.
    """

    def load_prices():
        return get_intraday_prices_with_prev_close(symbol)

    try:
        prices = cache.get_or_compute(f"intraday:{symbol}", TTL_MACRO, load_prices)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Intraday prices unavailable for {symbol}."
        ) from exc
    if prices is None or len(prices) < 2:
        raise HTTPException(
            status_code=404, detail=f"No intraday bars for {symbol} yet."
        )

    chain = _chain(symbol, expiration)
    if not chain:
        raise HTTPException(status_code=404, detail=f"No chain for {symbol} {expiration}")

    delta_series = compute_delta_exposure_series(chain, prices, offset)
    if delta_series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No delta exposure computable for {symbol} {expiration}.",
        )

    projection_ts = None
    projection_exposure = None
    if len(delta_series) >= 2:
        from datetime import datetime as dt_type, time as time_type

        hours = [
            (ts - delta_series.index[0]).total_seconds() / 3600
            for ts in delta_series.index
        ]
        slope, intercept = np.polyfit(hours, delta_series.values, 1)
        last_ts = delta_series.index[-1]
        end_ts = dt_type.combine(last_ts.date(), time_type(16, 0)).replace(
            tzinfo=last_ts.tzinfo
        )
        end_hour = (end_ts - delta_series.index[0]).total_seconds() / 3600
        projection_ts = end_ts.isoformat()
        projection_exposure = float(slope * end_hour + intercept)

    return {
        "symbol": symbol.upper(),
        "expiration": expiration,
        "offset": offset,
        "prev_close": {
            "ts": prices.index[0].isoformat(),
            "price": float(prices.iloc[0]),
        },
        "bars": [
            {
                "ts": ts.isoformat(),
                "price": float(prices[ts]),
                "delta_exposure": float(value),
            }
            for ts, value in delta_series.items()
        ],
        "projection_ts": projection_ts,
        "projection_exposure": projection_exposure,
    }


def get_gamma_gap_outcomes(
    ticker: Optional[str], horizon: int, limit: int
) -> Dict[str, Any]:
    """Logged signals scored against realized daily ranges.

    A hit means the magnet strike traded (low <= magnet <= high) within
    ``horizon`` sessions AFTER the signal date; the signal day itself never
    counts. Signals whose candle history cannot be fetched stay "pending"
    with zero evaluated sessions rather than being dropped or guessed.
    """

    # Pull extra history because we collapse repeated scans of one setup into
    # a single logged call below; the raw table holds many scans per call.
    signals = get_gamma_gap_history(ticker, max(limit * 8, limit))

    # Dedup to distinct calls: one (ticker, magnet, session) is a single
    # prediction, not one per scheduler scan. Keep the newest scan's numbers
    # (history is newest-first) and remember how many scans backed it.
    distinct: Dict[tuple, Dict[str, Any]] = {}
    for signal in signals:
        symbol = str(signal.get("ticker", "")).upper()
        magnet = signal.get("magnet_strike")
        ts = str(signal.get("ts", ""))
        if not symbol or magnet is None or not ts:
            continue
        key = (symbol, round(float(magnet), 4), ts[:10])
        existing = distinct.get(key)
        if existing is None:
            distinct[key] = {**signal, "scan_count": 1}
        else:
            existing["scan_count"] += 1
    ordered = sorted(
        distinct.values(), key=lambda s: str(s.get("ts", "")), reverse=True
    )[:limit]

    candles_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []

    for signal in ordered:
        symbol = str(signal.get("ticker", "")).upper()
        magnet = float(signal["magnet_strike"])
        ts = str(signal.get("ts", ""))
        if symbol not in candles_by_ticker:
            try:
                candles_by_ticker[symbol] = get_candles(symbol, 365)
            except HTTPException:
                candles_by_ticker[symbol] = []
        result = evaluate_gamma_gap_outcome(
            ts[:10], magnet, candles_by_ticker[symbol], horizon
        )
        # Plain-language direction of the call: is the magnet above or below
        # spot, and how far, as a percentage a customer can read at a glance.
        spot = signal.get("spot")
        direction = None
        gap_pct = None
        if spot not in (None, 0):
            direction = "up" if magnet >= float(spot) else "down"
            gap_pct = round((magnet / float(spot) - 1.0) * 100.0, 2)
        rows.append({**signal, **result, "direction": direction, "gap_pct": gap_pct})

    return {
        "horizon_sessions": horizon,
        "summary": summarize_outcomes(rows),
        "rows": rows,
    }


def _gamma_gap_read(
    direction: Optional[str],
    magnet: float,
    gap_pct: Optional[float],
    positive_zone: Optional[int],
) -> str:
    """One-sentence plain-English read for a radar row."""

    where = (
        f"toward {magnet:.1f}"
        if gap_pct is None
        else f"{'up' if direction == 'up' else 'down'} to {magnet:.1f} "
        f"({abs(gap_pct):.1f}% {'above' if direction == 'up' else 'below'} spot)"
    )
    if positive_zone == 1:
        posture = "dealers are long gamma here, so a pin to the magnet is more likely"
    else:
        posture = "dealers are short gamma here, so the pull is weaker and price can run"
    return f"Pull {where}; {posture}."


def get_gamma_gap_radar(min_score: float = 0.0) -> Dict[str, Any]:
    """Latest gamma-gap read per ticker, ranked by score - the pin radar.

    Collapses the raw scan log to one row per ticker (its most recent scan),
    ranks by gap-fill score, and attaches a plain-English read and the
    direction of the expected move. Rows whose latest scan predates the newest
    session are flagged stale rather than dropped.
    """

    history = get_gamma_gap_history(None, 3000)
    latest_by_ticker: Dict[str, Dict[str, Any]] = {}
    sessions: set = set()
    for signal in history:  # newest-first
        ticker = str(signal.get("ticker", "")).upper()
        ts = str(signal.get("ts", ""))
        if ts:
            sessions.add(ts[:10])
        if not ticker or ticker in latest_by_ticker:
            continue
        latest_by_ticker[ticker] = signal

    as_of = max(sessions) if sessions else None
    rows: List[Dict[str, Any]] = []
    for ticker, signal in latest_by_ticker.items():
        magnet = signal.get("magnet_strike")
        spot = signal.get("spot")
        score = signal.get("score")
        ts = str(signal.get("ts", ""))
        if magnet is None:
            continue
        if score is not None and float(score) < min_score:
            continue
        direction = None
        gap_pct = None
        if spot not in (None, 0):
            direction = "up" if float(magnet) >= float(spot) else "down"
            gap_pct = round((float(magnet) / float(spot) - 1.0) * 100.0, 2)
        rows.append(
            {
                **signal,
                "direction": direction,
                "gap_pct": gap_pct,
                "stale": bool(as_of and ts[:10] < as_of),
                "read": _gamma_gap_read(direction, float(magnet), gap_pct, signal.get("positive_zone")),
            }
        )

    rows.sort(
        key=lambda r: (r.get("score") is not None, r.get("score") or 0.0),
        reverse=True,
    )
    return {
        "as_of": as_of,
        "sessions": len(sessions),
        "tickers": len(rows),
        "rows": rows,
    }


def get_data_coverage() -> Dict[str, Any]:
    """Freshness summary sourced from logged gamma-gap scans.

    Answers "how current is this?" - the latest session we have data for, how
    many distinct sessions are logged, and how many tickers the newest session
    covered - so history-dependent pages can show an honest freshness badge.
    """

    history = get_gamma_gap_history(None, 3000)
    sessions: set = set()
    latest_tickers: set = set()
    as_of = None
    for signal in history:
        ts = str(signal.get("ts", ""))
        if not ts:
            continue
        day = ts[:10]
        sessions.add(day)
        as_of = max(as_of, day) if as_of else day
    if as_of:
        latest_tickers = {
            str(s.get("ticker", "")).upper()
            for s in history
            if str(s.get("ts", ""))[:10] == as_of and s.get("ticker")
        }
    return {
        "as_of": as_of,
        "sessions": len(sessions),
        "latest_ticker_count": len(latest_tickers),
    }


def get_market_overview() -> Dict[str, Any]:
    def compute() -> Dict[str, Any]:
        overview: Dict[str, Any] = {"vix": None, "yields": None, "futures": {}}
        try:
            overview["vix"] = get_vix_info()
        except Exception:
            pass
        try:
            overview["yields"] = get_bond_yield_info("^TNX")
        except Exception:
            pass
        try:
            overview["futures"] = get_futures_quotes()
        except Exception:
            pass
        return overview

    return cache.get_or_compute("market:overview", TTL_MACRO, compute)


def get_news() -> List[Dict[str, Any]]:
    return cache.get_or_compute("news", TTL_NEWS, fetch_and_filter_rss)


def _snapshot_dir() -> Path:
    return Path(get_settings().snapshot_dir)


def get_iv_rank(symbol: str) -> Dict[str, Any]:
    rank = snapshot_store.compute_iv_rank(symbol, _snapshot_dir())
    if rank is None:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough snapshot history for {symbol} yet.",
        )
    return {"symbol": symbol.upper(), **rank}


def get_oi_change(symbol: str, limit: int) -> List[Dict[str, Any]]:
    changes = snapshot_store.compute_oi_change(symbol, _snapshot_dir())
    if changes.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Need at least two snapshot days for {symbol}.",
        )
    return changes.head(limit).to_dict(orient="records")


def get_daily_metrics(symbol: Optional[str]) -> List[Dict[str, Any]]:
    metrics = snapshot_store.load_daily_metrics(_snapshot_dir(), symbol)
    return metrics.where(pd.notnull(metrics), None).to_dict(orient="records")
