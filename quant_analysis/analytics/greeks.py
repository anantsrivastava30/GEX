"""Local Black-Scholes greeks that Tradier does not supply.

Tradier's chain greeks include delta/gamma/theta/vega plus ``mid_iv`` and
``smv_vol``, but not vanna or charm. This module derives them locally so the
platform can ship UW-style vanna/charm exposure charts on free data.

Conventions:
- ``t`` is time to expiry in years, ``sigma`` an annualised vol (0.20 = 20%),
  ``r`` the annualised risk-free rate, ``q`` a continuous dividend yield.
- Vanna is dDelta/dVol per 1.00 change in implied vol.
- Charm is dDelta/dTime per year of calendar time passing (delta decay);
  divide by 365 for a per-day read.
- Exposure aggregation mirrors the existing net-GEX convention
  (``visualization.compute_net_gamma_exposure``): per-strike net value =
  call exposure minus put exposure, scaled by open interest and contract size.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.0

# Fallback when the ^IRX fetch fails; roughly the post-2022 policy-rate era.
DEFAULT_RISK_FREE_RATE = 0.04


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_d1_d2(
    spot: float, strike: float, t: float, r: float, sigma: float, q: float = 0.0
) -> tuple:
    sig_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / sig_sqrt_t
    return d1, d1 - sig_sqrt_t


def bs_delta(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes delta. Used as the finite-difference anchor in tests."""

    d1, _ = bs_d1_d2(spot, strike, t, r, sigma, q)
    if option_type == "call":
        return math.exp(-q * t) * _norm_cdf(d1)
    return math.exp(-q * t) * (_norm_cdf(d1) - 1.0)


def bs_vanna(
    spot: float, strike: float, t: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """dDelta/dVol. Identical for calls and puts under Black-Scholes."""

    d1, d2 = bs_d1_d2(spot, strike, t, r, sigma, q)
    return -math.exp(-q * t) * _norm_pdf(d1) * d2 / sigma


def bs_charm(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """dDelta/dTime (delta decay) per year of calendar time passing.

    Equal for calls and puts when ``q`` is zero; the dividend term splits them.
    """

    d1, d2 = bs_d1_d2(spot, strike, t, r, sigma, q)
    sig_sqrt_t = sigma * math.sqrt(t)
    decay = (
        -math.exp(-q * t)
        * _norm_pdf(d1)
        * (2.0 * (r - q) * t - d2 * sig_sqrt_t)
        / (2.0 * t * sig_sqrt_t)
    )
    if option_type == "call":
        return decay + q * math.exp(-q * t) * _norm_cdf(d1)
    return decay - q * math.exp(-q * t) * _norm_cdf(-d1)


def get_risk_free_rate(default: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Annualised risk-free rate from the 13-week T-bill index (^IRX).

    yfinance quotes ^IRX as a percentage; falls back to ``default`` when the
    fetch fails so exposure endpoints never 500 on a macro-data hiccup.
    """

    try:
        import yfinance as yf

        hist = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if not hist.empty:
            rate = float(hist.iloc[-1]) / 100.0
            if 0.0 <= rate < 0.25:
                return rate
    except Exception as exc:
        logger.warning("Risk-free rate fetch failed (%s); using %.2f", exc, default)
    return default


def _column_or(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    """Numeric column with NaNs (or the whole column) defaulted."""

    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def iv_series(df: pd.DataFrame) -> pd.Series:
    """Implied vol per contract: ``mid_iv`` with ``smv_vol`` filling gaps.

    ``mid_iv`` is null or zero on illiquid strikes; Tradier's model vol
    (``smv_vol``) is the documented fallback (docs/UW_PARITY_PLAN.md Phase 2).
    """

    if "mid_iv" in df.columns:
        iv = pd.to_numeric(df["mid_iv"], errors="coerce")
    else:
        iv = pd.Series(np.nan, index=df.index)
    iv = iv.where(iv > 0)
    if "smv_vol" in df.columns:
        smv = pd.to_numeric(df["smv_vol"], errors="coerce")
        iv = iv.fillna(smv.where(smv > 0))
    return iv


def _years_to_expiry(df: pd.DataFrame) -> pd.Series:
    """Time to expiry in years from ``DTE`` or ``expiration_date``."""

    if "DTE" in df.columns:
        days = pd.to_numeric(df["DTE"], errors="coerce")
    elif "expiration_date" in df.columns:
        exp = pd.to_datetime(df["expiration_date"], errors="coerce")
        days = (exp - pd.Timestamp.now()).dt.total_seconds() / 86400.0
    else:
        return pd.Series(np.nan, index=df.index)
    # Same-day expiries still have a fraction of a session left; floor at a
    # half trading day so d1 stays finite without inflating far-dated rows.
    return days.clip(lower=0.5) / DAYS_PER_YEAR


def compute_vanna_charm_exposures(
    df: pd.DataFrame,
    spot: float,
    r: Optional[float] = None,
    offset: Optional[float] = None,
    q: float = 0.0,
) -> pd.DataFrame:
    """Per-strike net vanna and charm exposure from a flattened chain frame.

    ``df`` needs ``strike``, ``option_type``, ``open_interest`` and an IV
    source (``mid_iv``/``smv_vol``) plus ``DTE`` or ``expiration_date``;
    ``contract_size`` defaults to 100. Net = calls minus puts per strike,
    matching the net-GEX dealer-positioning convention.

    Returns a DataFrame with columns ``strike``, ``vanna_exposure``,
    ``charm_exposure`` sorted by strike.
    """

    empty = pd.DataFrame(columns=["strike", "vanna_exposure", "charm_exposure"])
    if df is None or df.empty or not spot or spot <= 0:
        return empty

    work = df.copy()
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    if offset is not None:
        work = work[
            (work["strike"] >= spot - offset) & (work["strike"] <= spot + offset)
        ]

    work["_iv"] = iv_series(work)
    work["_t"] = _years_to_expiry(work)
    work = work.dropna(subset=["strike", "_iv", "_t"])
    work = work[(work["_iv"] > 0) & (work["_t"] > 0) & (work["strike"] > 0)]
    if work.empty:
        return empty

    if r is None:
        r = get_risk_free_rate()

    oi = _column_or(work, "open_interest", 0.0)
    size = _column_or(work, "contract_size", 100.0).replace(0, 100)

    is_call = work["option_type"].astype(str).str.lower() == "call"
    sign = np.where(is_call, 1.0, -1.0)

    vanna = np.array(
        [
            bs_vanna(spot, k, t, r, sig, q)
            for k, t, sig in zip(work["strike"], work["_t"], work["_iv"])
        ]
    )
    charm = np.array(
        [
            bs_charm(spot, k, t, r, sig, "call" if call_ else "put", q)
            for k, t, sig, call_ in zip(
                work["strike"], work["_t"], work["_iv"], is_call
            )
        ]
    )

    scale = (oi * size).to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "strike": work["strike"].to_numpy(),
            "vanna_exposure": vanna * scale * sign,
            "charm_exposure": charm * scale * sign,
        }
    )
    return (
        out.groupby("strike", as_index=False)[["vanna_exposure", "charm_exposure"]]
        .sum()
        .sort_values("strike")
        .reset_index(drop=True)
    )


def compute_max_pain(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Max pain: the expiry price that minimises option holders' payout.

    Pure function over chain open interest. For each candidate settlement
    price (the strike grid), payout = call OI x max(0, S - K) x size +
    put OI x max(0, K - S) x size, summed over the chain.

    Returns ``{"max_pain": strike, "curve": [{"strike", "call_pain",
    "put_pain", "total_pain"}, ...]}`` or None when OI is absent.
    """

    if df is None or df.empty:
        return None

    work = df.copy()
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work["open_interest"] = _column_or(work, "open_interest", 0.0)
    work["_size"] = _column_or(work, "contract_size", 100.0).replace(0, 100)
    work = work.dropna(subset=["strike"])
    work = work[work["open_interest"] > 0]
    if work.empty:
        return None

    side = work["option_type"].astype(str).str.lower()
    calls = work[side == "call"]
    puts = work[side == "put"]

    strikes = np.sort(work["strike"].unique())
    call_k = calls["strike"].to_numpy()
    call_w = (calls["open_interest"] * calls["_size"]).to_numpy(dtype=float)
    put_k = puts["strike"].to_numpy()
    put_w = (puts["open_interest"] * puts["_size"]).to_numpy(dtype=float)

    curve: List[Dict[str, float]] = []
    for s in strikes:
        call_pain = float(np.sum(np.maximum(s - call_k, 0.0) * call_w))
        put_pain = float(np.sum(np.maximum(put_k - s, 0.0) * put_w))
        curve.append(
            {
                "strike": float(s),
                "call_pain": call_pain,
                "put_pain": put_pain,
                "total_pain": call_pain + put_pain,
            }
        )

    best = min(curve, key=lambda row: row["total_pain"])
    return {"max_pain": best["strike"], "curve": curve}
