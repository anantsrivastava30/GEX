"""Service layer: cached, rate-limited adapters over quant_analysis."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import HTTPException

from quant_analysis.analytics.greeks import (
    compute_max_pain,
    compute_vanna_charm_exposures,
    get_risk_free_rate,
)
from quant_analysis.analytics.visualization import (
    compute_gamma_gap_metrics,
    compute_net_gamma_exposure,
    describe_gamma_gap,
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


def get_gamma_gap_history(ticker: Optional[str], limit: int) -> List[Dict[str, Any]]:
    rows = load_gamma_gap_history(limit=limit if not ticker else max(limit * 10, limit))
    if ticker:
        upper = ticker.upper()
        rows = [r for r in rows if str(r.get("ticker", "")).upper() == upper][:limit]
    for row in rows:
        row.pop("payload", None)
    return rows


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
