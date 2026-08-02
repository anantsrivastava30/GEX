"""Headless CLI for ``quant_analysis``.

Sprint 1.1 deliverable. Provides three subcommands that emit JSON to
stdout, suitable for invocation from ``cli_trader`` strategies (either
directly via ``python -m quant_analysis.cli ...`` or as a subprocess).

Subcommands
-----------

``net-gex``
    Aggregate net dealer gamma exposure for ``--symbol`` at
    ``--expiry``. If ``--expiry`` is omitted, the next listed
    expiration is used.

``zero-gamma``
    Locate the zero-gamma flip strike for ``--symbol`` at the next
    expiration. Computed by linear interpolation on the cumulative net
    GEX series.

``iv-rank``
    Compute a 0..100 implied-volatility rank for ``--symbol`` over a
    ``--lookback`` (default 252 trading days). The Tradier free tier
    does **not** expose historical option-chain IV, so this command
    falls back to a realised-vol proxy based on daily underlying
    returns. The JSON payload carries a ``method`` field so callers can
    distinguish the proxy from a true ATM-IV rank.

Design notes
------------

This module is deliberately self-contained. It imports only
:mod:`quant_analysis._st_shim`, :mod:`quant_analysis.credentials`, and
:mod:`quant_analysis.integrations.tradier`. It avoids the heavier
:mod:`quant_analysis.services.market_data` and
:mod:`quant_analysis.analytics.visualization` modules so the CLI runs
in environments (such as the ``cli_trader`` virtualenv) that do not
have ``yfinance``, ``feedparser``, ``matplotlib``, ``plotly`` or
``seaborn`` installed.

JSON output is always written to stdout. Errors are written as a JSON
object with ``error`` and ``detail`` fields **and** the process exits
non-zero, so subprocess callers can branch on either signal.

Reference: ``docs/openclaw/13_SPRINT_PLAN.md`` §1.1;
``docs/openclaw/03_EXISTING_CODEBASE_MAP.md`` Sprint 1.1 deliverable.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import click

# Importing the package installs the Streamlit shim if needed.
import quant_analysis  # noqa: F401  (side-effect import)
from quant_analysis.credentials import (
    TradierCredentialsError,
    get_tradier_credentials,
)
from quant_analysis.integrations.tradier import TradierAPI, TradierAPIError


_LOGGER = logging.getLogger("quant_analysis.cli")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _emit(payload: Dict[str, Any]) -> None:
    """Write a JSON payload to stdout with a trailing newline."""

    json.dump(payload, sys.stdout, default=_json_default, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _json_default(value: Any) -> Any:
    """JSON fallback for ``datetime`` and numpy-ish numerics."""

    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _fail(error: str, detail: Optional[str] = None, *, exit_code: int = 1) -> None:
    """Emit an error JSON object and exit non-zero."""

    payload: Dict[str, Any] = {"status": "error", "error": error}
    if detail:
        payload["detail"] = detail
    _emit(payload)
    sys.exit(exit_code)


def _get_api(env_choice: Optional[str]) -> Tuple[TradierAPI, str]:
    """Resolve credentials and return ``(api, environment_label)``."""

    try:
        creds = get_tradier_credentials(env_choice)
    except TradierCredentialsError as exc:
        _fail("credentials_unavailable", str(exc))
        raise  # unreachable; satisfies type-checker

    api = TradierAPI(creds.api_key, base_url=creds.base_url)
    return api, creds.environment


def _next_expiry(api: TradierAPI, symbol: str, prefer: Optional[str]) -> str:
    """Return ``prefer`` when set, otherwise the nearest listed expiry."""

    expirations: List[str] = api.expirations(symbol, include_all_roots=True)
    if not expirations:
        _fail(
            "no_expirations",
            f"Tradier returned no expirations for symbol {symbol!r}.",
        )
    if prefer:
        if prefer not in expirations:
            _fail(
                "expiry_not_listed",
                f"Expiry {prefer!r} not in Tradier listing for {symbol!r}. "
                f"Available (first 8): {expirations[:8]}",
            )
        return prefer
    return expirations[0]


def _spot(api: TradierAPI, symbol: str) -> float:
    """Pull the last/quote price for a symbol."""

    quote = api.quote(symbol)
    for key in ("last", "close", "prevclose", "ask", "bid"):
        value = quote.get(key)
        if value is not None:
            return float(value)
    _fail("no_spot", f"No usable price field in Tradier quote for {symbol!r}: {quote}")
    raise RuntimeError  # unreachable


# ---------------------------------------------------------------------------
# Net GEX core (self-contained reimplementation)
# ---------------------------------------------------------------------------


def _net_gex_per_strike(
    chain: List[Dict[str, Any]],
    spot: float,
    offset: float,
) -> List[Dict[str, float]]:
    """Compute per-strike net dealer gamma exposure.

    Mirrors :func:`quant_analysis.analytics.visualization.compute_net_gamma_exposure`
    but is reimplemented here without pandas/plotly so the CLI does not
    depend on the analytics module.

    Sign convention: dealers are assumed long calls / short puts (the
    standard market-maker assumption used throughout the dashboard), so
    per-strike net GEX = call_GEX - put_GEX. Positive net GEX is the
    "dampening" regime; negative is the "amplifying" regime.
    """

    per_strike_calls: Dict[float, float] = {}
    per_strike_puts: Dict[float, float] = {}

    for opt in chain:
        try:
            strike = float(opt["strike"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (spot - offset <= strike <= spot + offset):
            continue

        greeks = opt.get("greeks") or {}
        gamma = greeks.get("gamma")
        if gamma is None:
            continue
        try:
            gamma_f = float(gamma)
        except (TypeError, ValueError):
            continue

        oi = opt.get("open_interest") or 0
        try:
            oi_i = int(oi)
        except (TypeError, ValueError):
            oi_i = 0

        multiplier = opt.get("contract_size") or 100
        try:
            mult_f = float(multiplier)
        except (TypeError, ValueError):
            mult_f = 100.0

        gex = gamma_f * oi_i * mult_f
        side = str(opt.get("option_type", "")).lower()
        if side == "call":
            per_strike_calls[strike] = per_strike_calls.get(strike, 0.0) + gex
        elif side == "put":
            per_strike_puts[strike] = per_strike_puts.get(strike, 0.0) + gex

    strikes = sorted(set(per_strike_calls) | set(per_strike_puts))
    rows = []
    for k in strikes:
        net = per_strike_calls.get(k, 0.0) - per_strike_puts.get(k, 0.0)
        rows.append(
            {
                "strike": k,
                "call_gex": per_strike_calls.get(k, 0.0),
                "put_gex": per_strike_puts.get(k, 0.0),
                "net_gex": net,
            }
        )
    return rows


def _zero_gamma_strike(rows: List[Dict[str, float]]) -> Optional[float]:
    """Linearly interpolate the strike at which cumulative net GEX = 0.

    Returns ``None`` when the cumulative series does not change sign in
    the observed strike window. The cumulative series is the running
    sum of per-strike net GEX, which is what the dealer-positioning
    literature uses to identify the gamma-flip level.
    """

    if len(rows) < 2:
        return None

    cumulative: List[Tuple[float, float]] = []
    running = 0.0
    for row in rows:
        running += float(row["net_gex"])
        cumulative.append((float(row["strike"]), running))

    for i in range(1, len(cumulative)):
        k0, c0 = cumulative[i - 1]
        k1, c1 = cumulative[i]
        if c0 == 0.0:
            return k0
        if c1 == 0.0:
            return k1
        if (c0 < 0.0 < c1) or (c0 > 0.0 > c1):
            # Linear interpolation between the two adjacent strikes.
            slope = (c1 - c0) / (k1 - k0) if k1 != k0 else 0.0
            if slope == 0.0:
                return (k0 + k1) / 2.0
            return k0 - c0 / slope
    return None


# ---------------------------------------------------------------------------
# IV-rank realised-vol proxy
# ---------------------------------------------------------------------------


def _realised_vol_rank(
    api: TradierAPI,
    symbol: str,
    lookback: int,
) -> Dict[str, Any]:
    """Compute an IV-rank-style score from realised vol of daily returns.

    The Tradier free tier does not expose historical option-chain IV,
    so we cannot compute a true ATM-IV rank without an additional
    paid feed. This proxy uses the underlying's daily close-to-close
    realised volatility:

    1. Fetch ``lookback + 21`` daily bars from Tradier's history endpoint.
    2. Compute a rolling 21-day annualised realised volatility series.
    3. Rank the latest 21d realised vol against the trailing
       ``lookback`` window, returning ``(rv - min) / (max - min) * 100``.

    The result is **not** an IV rank in the formal sense; it is a
    realised-vol percentile, useful as a stand-in for sizing decisions
    that key off vol regime. The JSON output flags this clearly via the
    ``method`` field so downstream code can decide whether to consume
    it. A future ticket should replace this with true ATM-IV history
    once a historical chain feed is wired in (see
    ``docs/openclaw/06_DATA_SOURCES.md``).
    """

    # Tradier daily history end inclusive; pull 50 extra calendar days
    # to safely cover ~lookback + 21 trading days plus weekends.
    end = _dt.date.today()
    start = end - _dt.timedelta(days=int(lookback * 1.6) + 60)
    bars = api.history(
        symbol,
        interval="daily",
        start=start.isoformat(),
        end=end.isoformat(),
    )
    if not bars or len(bars) < 22:
        _fail(
            "insufficient_history",
            f"Tradier returned {len(bars) if bars else 0} bars for "
            f"{symbol!r}; need at least 22 to compute realised vol.",
        )

    closes: List[float] = []
    for bar in bars:
        c = bar.get("close")
        if c is None:
            continue
        try:
            closes.append(float(c))
        except (TypeError, ValueError):
            continue

    if len(closes) < 22:
        _fail("insufficient_history", f"Only {len(closes)} usable closes.")

    # Daily log returns
    log_returns: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0 or cur <= 0:
            continue
        log_returns.append(math.log(cur / prev))

    # Rolling 21-day annualised stdev
    window = 21
    annualisation = math.sqrt(252.0)
    rolling: List[float] = []
    for i in range(window, len(log_returns) + 1):
        chunk = log_returns[i - window : i]
        mean = sum(chunk) / len(chunk)
        variance = sum((x - mean) ** 2 for x in chunk) / (len(chunk) - 1)
        rolling.append(math.sqrt(variance) * annualisation)

    if len(rolling) < 2:
        _fail("insufficient_history", "Not enough rolling-vol points to rank.")

    # Use the last `lookback` (or all available, whichever is smaller)
    # observations as the ranking window, including the latest point.
    window_series = rolling[-min(lookback, len(rolling)) :]
    latest = window_series[-1]
    lo = min(window_series)
    hi = max(window_series)
    if hi == lo:
        rank_pct = 50.0
    else:
        rank_pct = (latest - lo) / (hi - lo) * 100.0

    return {
        "method": "realised_vol_proxy",
        "method_note": (
            "Tradier free tier lacks historical implied-vol; this is a "
            "21d realised-vol percentile over the lookback window. Treat "
            "as a proxy, not a true IV rank."
        ),
        "lookback_observations": len(window_series),
        "latest_realised_vol_annualised": round(latest, 6),
        "min_realised_vol_annualised": round(lo, 6),
        "max_realised_vol_annualised": round(hi, 6),
        "rank_pct": round(rank_pct, 4),
    }


# ---------------------------------------------------------------------------
# Click app
# ---------------------------------------------------------------------------


_ENV_CHOICE = click.Choice(["sandbox", "production"], case_sensitive=False)


@click.group(help="Headless CLI for quant_analysis (JSON stdout).")
@click.option(
    "--env",
    "env_choice",
    type=_ENV_CHOICE,
    default=None,
    help="Override TRADIER_ENV. Defaults to the value set in the environment.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Emit DEBUG-level logs to stderr.",
)
@click.pass_context
def cli(ctx: click.Context, env_choice: Optional[str], verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    ctx.ensure_object(dict)
    ctx.obj["env_choice"] = env_choice


@cli.command("net-gex", help="Aggregate net dealer gamma exposure.")
@click.option("--symbol", required=True, help="Underlying ticker, e.g. SPY.")
@click.option(
    "--expiry",
    default=None,
    help="Expiration in YYYY-MM-DD. Defaults to the next listed expiration.",
)
@click.option(
    "--offset",
    type=float,
    default=25.0,
    show_default=True,
    help="Strike window half-width around spot, in price units.",
)
@click.pass_context
def net_gex_cmd(
    ctx: click.Context,
    symbol: str,
    expiry: Optional[str],
    offset: float,
) -> None:
    api, env = _get_api(ctx.obj["env_choice"])
    symbol = symbol.upper()
    try:
        chosen_expiry = _next_expiry(api, symbol, expiry)
        spot = _spot(api, symbol)
        chain = api.option_chain(symbol, chosen_expiry, greeks="true")
    except TradierAPIError as exc:
        _fail("tradier_api_error", str(exc))
        return

    rows = _net_gex_per_strike(chain, spot, offset)
    aggregated = sum(r["net_gex"] for r in rows)
    if rows:
        peak = max(rows, key=lambda r: r["net_gex"])
        trough = min(rows, key=lambda r: r["net_gex"])
        peak_strike = peak["strike"]
        trough_strike = trough["strike"]
    else:
        peak_strike = None
        trough_strike = None

    _emit(
        {
            "status": "ok",
            "command": "net-gex",
            "symbol": symbol,
            "environment": env,
            "expiry": chosen_expiry,
            "spot": spot,
            "offset": offset,
            "n_strikes_in_window": len(rows),
            "aggregated_net_gex": aggregated,
            "peak_net_gex_strike": peak_strike,
            "trough_net_gex_strike": trough_strike,
            "per_strike": rows,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
    )


@cli.command("zero-gamma", help="Estimate the zero-gamma flip strike.")
@click.option("--symbol", required=True, help="Underlying ticker, e.g. SPY.")
@click.option(
    "--expiry",
    default=None,
    help="Expiration in YYYY-MM-DD. Defaults to the next listed expiration.",
)
@click.option(
    "--offset",
    type=float,
    default=50.0,
    show_default=True,
    help="Strike window half-width around spot. Wider than net-gex by "
    "default since the zero-crossing may sit further from spot.",
)
@click.pass_context
def zero_gamma_cmd(
    ctx: click.Context,
    symbol: str,
    expiry: Optional[str],
    offset: float,
) -> None:
    api, env = _get_api(ctx.obj["env_choice"])
    symbol = symbol.upper()
    try:
        chosen_expiry = _next_expiry(api, symbol, expiry)
        spot = _spot(api, symbol)
        chain = api.option_chain(symbol, chosen_expiry, greeks="true")
    except TradierAPIError as exc:
        _fail("tradier_api_error", str(exc))
        return

    rows = _net_gex_per_strike(chain, spot, offset)
    flip = _zero_gamma_strike(rows)

    _emit(
        {
            "status": "ok",
            "command": "zero-gamma",
            "symbol": symbol,
            "environment": env,
            "expiry": chosen_expiry,
            "spot": spot,
            "offset": offset,
            "n_strikes_in_window": len(rows),
            "zero_gamma_strike": flip,
            "method": "linear_interp_on_cumulative_net_gex",
            "method_note": (
                "Cumulative sum of per-strike net GEX is computed in "
                "ascending strike order; the flip strike is the linear "
                "interpolant at the first sign change. Returns null when "
                "no sign change is present in the observed window."
            ),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
    )


@cli.command("iv-rank", help="Realised-vol-proxy rank over a lookback window.")
@click.option("--symbol", required=True, help="Underlying ticker, e.g. SPY.")
@click.option(
    "--lookback",
    type=int,
    default=252,
    show_default=True,
    help="Trading-day lookback window for the percentile rank.",
)
@click.pass_context
def iv_rank_cmd(ctx: click.Context, symbol: str, lookback: int) -> None:
    api, env = _get_api(ctx.obj["env_choice"])
    symbol = symbol.upper()
    if lookback < 22:
        _fail("invalid_lookback", "Lookback must be at least 22 trading days.")
        return

    try:
        result = _realised_vol_rank(api, symbol, lookback)
    except TradierAPIError as exc:
        _fail("tradier_api_error", str(exc))
        return

    _emit(
        {
            "status": "ok",
            "command": "iv-rank",
            "symbol": symbol,
            "environment": env,
            "lookback_requested": lookback,
            **result,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
    )


def main() -> None:
    """Module entry-point for ``python -m quant_analysis.cli``."""

    cli(obj={})


if __name__ == "__main__":
    main()
