"""Best-effort stock fundamentals from yfinance for CAN SLIM evaluation.

Yahoo's unofficial endpoints are fragile, so every field is fetched
independently and failures degrade that one field to None instead of losing
the symbol. Callers receive a ``missing`` list naming what could not be
fetched so the UI can label unavailable criteria honestly.

Fields map to O'Neil's letters:
- C: latest quarterly EPS vs the same quarter a year ago, plus quarterly
  revenue growth (accelerating sales).
- A: annual EPS growth across recent fiscal years and return on equity.
- S: float and shares outstanding (supply context).
- I: institutional ownership percent (13F-derived, quarterly, lagged).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_EPS_ROWS = ("Diluted EPS", "Basic EPS")


def _growth_pct(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    """YoY growth in percent; None when unavailable or prior is non-positive.

    O'Neil's growth screens assume a positive base; a loss-to-profit swing is
    reported as None rather than a misleading percentage.
    """

    if current is None or prior is None or prior <= 0:
        return None
    return (current / prior - 1.0) * 100.0


def _first_row(frame: Any, names: tuple) -> Optional[Any]:
    for name in names:
        if name in frame.index:
            return frame.loc[name]
        continue
    return None


def _yoy_series(values: List[float], max_points: int = 4) -> List[Optional[float]]:
    """Year-over-year growth for each recent quarter, newest first.

    ``values`` are quarterly figures newest first; element i is compared
    with element i+4 (the same quarter a year earlier), which is what makes
    the series seasonally comparable and therefore usable for acceleration.
    """

    out: List[Optional[float]] = []
    for i in range(min(max_points, max(0, len(values) - 4))):
        out.append(_growth_pct(values[i], values[i + 4]))
    return out


def _quarterly_eps_growth(ticker: Any, out: Dict[str, Any]) -> None:
    """Reported quarterly EPS growth, newest first, up to four quarters."""

    values: List[float] = []
    try:
        dates = ticker.get_earnings_dates(limit=16)
        reported = dates["Reported EPS"].dropna() if dates is not None else None
        if reported is not None and len(reported) >= 5:
            values = [float(v) for v in reported]
    except Exception:
        pass

    if len(values) < 5:
        try:
            q = ticker.quarterly_income_stmt
            eps = _first_row(q, _EPS_ROWS)
            if eps is not None:
                values = [float(v) for v in eps.dropna()]
        except Exception:
            pass

    if len(values) < 5:
        # Last resort: quarterly net income still answers "is the company
        # earning more than a year ago", which is what C asks.
        try:
            q = ticker.quarterly_income_stmt
            net = _first_row(q, ("Net Income", "Net Income Common Stockholders"))
            if net is not None:
                values = [float(v) for v in net.dropna()]
        except Exception:
            pass

    if len(values) >= 5:
        series = _yoy_series(values)
        out["quarterly_eps"] = values[0]
        out["quarterly_eps_prior"] = values[4]
        out["quarterly_eps_values"] = values[:8]
        out["quarterly_eps_growth_pct"] = series[0] if series else None
        out["quarterly_eps_growth_series"] = series
        if out["quarterly_eps_growth_pct"] is not None:
            return
        # A non-positive base is not missing data - the company was (or is)
        # losing money, which the evaluator reports as a failed criterion.
        return
    out["missing"].append("quarterly_eps_growth")


def _quarterly_revenue_growth(ticker: Any, out: Dict[str, Any]) -> None:
    """Quarterly revenue growth series - O'Neil's 'accelerating sales'."""

    try:
        q = ticker.quarterly_income_stmt
        rev = _first_row(q, ("Total Revenue", "Operating Revenue"))
        if rev is not None:
            values = [float(v) for v in rev.dropna()]
            series = _yoy_series(values)
            if series:
                out["quarterly_revenue_growth_pct"] = series[0]
                out["quarterly_revenue_growth_series"] = series
                if series[0] is not None:
                    return
    except Exception:
        pass
    out["missing"].append("quarterly_revenue_growth")


def _annual_eps_growth(ticker: Any, out: Dict[str, Any]) -> None:
    """Average YoY EPS growth across the available fiscal years (up to 4)."""

    try:
        a = ticker.income_stmt
        eps = _first_row(a, _EPS_ROWS)
        if eps is None:
            eps = _first_row(a, ("Net Income",))
        if eps is not None:
            values = [float(v) for v in eps.dropna()]
            if len(values) >= 2:
                out["annual_eps_values"] = values[:5]
                growths = [
                    g
                    for g in (
                        _growth_pct(values[i], values[i + 1])
                        for i in range(len(values) - 1)
                    )
                    if g is not None
                ]
                if growths:
                    out["annual_eps_growth_pct"] = sum(growths) / len(growths)
                    out["annual_growth_years"] = len(growths)
                return
    except Exception:
        pass
    out["missing"].append("annual_eps_growth")


def _info_fields(ticker: Any, out: Dict[str, Any]) -> None:
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    if not info:
        out["missing"].extend(["roe", "institutional_pct", "float_shares"])
        return

    out["name"] = info.get("shortName") or info.get("longName")
    out["quote_type"] = info.get("quoteType")

    roe = info.get("returnOnEquity")
    if isinstance(roe, (int, float)):
        out["roe_pct"] = float(roe) * 100.0
    else:
        out["missing"].append("roe")

    inst = info.get("heldPercentInstitutions")
    if isinstance(inst, (int, float)):
        out["institutional_pct"] = float(inst) * 100.0
    else:
        out["missing"].append("institutional_pct")

    # Number of reporting institutional holders. Yahoo exposes only the
    # largest holders, so this is a floor, not a census; it is tracked over
    # time so the direction of sponsorship is what actually gets used.
    try:
        holders = ticker.institutional_holders
        if holders is not None and not getattr(holders, "empty", True):
            out["institutional_holder_count"] = int(len(holders))
    except Exception:
        pass

    float_shares = info.get("floatShares")
    shares = info.get("sharesOutstanding")
    if isinstance(float_shares, (int, float)):
        out["float_shares"] = float(float_shares)
    else:
        out["missing"].append("float_shares")
    if isinstance(shares, (int, float)):
        out["shares_outstanding"] = float(shares)


def fetch_etf_holdings(symbol: str, limit: int = 12) -> List[str]:
    """Top holdings of an ETF, best-effort.

    Yahoo exposes only the largest holdings, which is the right slice for
    this use: O'Neil bought the leaders of a leading group, not its tail.
    Returns an empty list when the provider is unreachable so the caller
    can fall back to configured constituents.
    """

    import yfinance as yf

    try:
        funds = yf.Ticker(symbol).funds_data
        holdings = getattr(funds, "top_holdings", None)
        if holdings is None or getattr(holdings, "empty", True):
            return []
        symbols = [str(s).upper().strip() for s in holdings.index]
    except Exception:
        logger.info("ETF holdings unavailable for %s", symbol)
        return []

    out: List[str] = []
    for candidate in symbols:
        # Skip cash/derivative placeholder rows and non-equity tickers.
        if not candidate or not candidate.replace(".", "").replace("-", "").isalnum():
            continue
        if len(candidate) > 6:
            continue
        if candidate not in out:
            out.append(candidate)
        if len(out) >= limit:
            break
    return out


def fetch_stock_fundamentals(symbol: str) -> Dict[str, Any]:
    """All CAN SLIM fundamental inputs for one symbol, best-effort."""

    import yfinance as yf

    out: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "name": None,
        "quote_type": None,
        "quarterly_eps": None,
        "quarterly_eps_prior": None,
        "quarterly_eps_values": [],
        "annual_eps_values": [],
        "quarterly_eps_growth_pct": None,
        "quarterly_eps_growth_series": [],
        "quarterly_revenue_growth_pct": None,
        "quarterly_revenue_growth_series": [],
        "annual_eps_growth_pct": None,
        "annual_growth_years": None,
        "roe_pct": None,
        "institutional_pct": None,
        "institutional_holder_count": None,
        "float_shares": None,
        "shares_outstanding": None,
        "missing": [],
    }
    try:
        ticker = yf.Ticker(symbol)
    except Exception:
        out["missing"] = [
            "quarterly_eps_growth",
            "quarterly_revenue_growth",
            "annual_eps_growth",
            "roe",
            "institutional_pct",
            "float_shares",
        ]
        return out

    _quarterly_eps_growth(ticker, out)
    _quarterly_revenue_growth(ticker, out)
    _annual_eps_growth(ticker, out)
    _info_fields(ticker, out)
    missing: List[str] = list(dict.fromkeys(out["missing"]))
    out["missing"] = missing
    if missing:
        logger.info("Fundamentals for %s missing: %s", symbol, ", ".join(missing))
    return out
