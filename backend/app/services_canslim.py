"""Per-stock CAN SLIM scan: which stocks are flagged to buy, and why.

Scans the snapshot/watchlist stock universe (index ETFs excluded), scores
each name on the seven letters, gates buys on the market-direction state,
and persists fresh heavy-volume breakouts as ``stock_breakout`` signals in
the shared direction signal feed.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from quant_analysis.analytics.canslim import (
    build_stock_narrative,
    describe_breakout_signal,
    detect_breakout,
    evaluate_stock_canslim,
    merged_canslim_config,
    weighted_rs_score,
)
from quant_analysis.analytics.market_direction import STATE_LABELS, classify_market_state
from quant_analysis.config import CONFIG

from backend.app.cache import cache
from backend.app.services_direction import (
    _completed_bars,
    _fetch_bars,
    _market_open_now,
    direction_config,
)

logger = logging.getLogger(__name__)

TTL_FUNDAMENTALS = 24 * 3600
TTL_HOLDINGS = 24 * 3600
TTL_LEADERS = 900

_GATE_MESSAGES = {
    "confirmed_uptrend": "Confirmed uptrend - breakout buys are allowed under O'Neil's rules.",
    "uptrend_under_pressure": "Uptrend under pressure - distribution is building; size new buys with caution.",
    "rally_attempt": "Rally attempt in progress - wait for a follow-through day before new buys.",
    "correction": "Market in correction - O'Neil's rule is no new buys; build the watchlist instead.",
    "unavailable": "Broad market state unavailable.",
}

_READINESS_ORDER = {
    "buy_candidate": 0,
    "extended": 1,
    "near_pivot": 2,
    "wait_market": 3,
    "not_ready": 4,
    "insufficient_data": 5,
}


def canslim_config() -> Dict[str, Any]:
    raw = CONFIG.get("canslim", {}) or {}
    cfg = merged_canslim_config(raw)
    cfg["exclude"] = [str(s).upper() for s in raw.get("exclude", []) or []]
    cfg["max_symbols"] = int(raw.get("max_symbols", 160))
    cfg["max_constituents_per_etf"] = int(raw.get("max_constituents_per_etf", 12))
    cfg["fundamentals_top_n"] = int(raw.get("fundamentals_top_n", 25))
    cfg["fetch_workers"] = max(1, min(int(raw.get("fetch_workers", 6)), 16))
    cfg["include_watchlist_symbols"] = bool(raw.get("include_watchlist_symbols", True))
    cfg["fundamentals_pause_seconds"] = float(
        raw.get("fundamentals_pause_seconds", 0.4)
    )
    cfg["fallback_constituents"] = {
        str(k).upper(): [str(s).upper() for s in v or []]
        for k, v in (raw.get("fallback_constituents", {}) or {}).items()
    }
    return cfg


def get_etf_constituents(symbol: str, limit: int, fallback: List[str]) -> Dict[str, Any]:
    """Holdings for one ETF: live when reachable, configured list otherwise."""

    def compute() -> Dict[str, Any]:
        from quant_analysis.integrations.fundamentals import fetch_etf_holdings

        live = fetch_etf_holdings(symbol, limit)
        if live:
            return {"symbols": live, "source": "provider"}
        return {"symbols": list(fallback[:limit]), "source": "configured"}

    return cache.get_or_compute(
        f"canslim:holdings:{symbol}:{limit}", TTL_HOLDINGS, compute
    )


def _build_universe(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Candidates from every tracked ETF, attributed to their leading group.

    A stock held by several ETFs is attributed to the most specific group
    (a sector over the broad market) and, among sectors, to the strongest
    one by relative strength - O'Neil bought leaders of leading groups, so
    the group a name is credited to should be the one actually leading.
    """

    dcfg = direction_config()
    exclude = set(cfg["exclude"])

    # Rank the ETFs themselves so attribution can prefer the leading group.
    etf_rs: Dict[str, Optional[float]] = {}
    for entry in dcfg["indices"]:
        bars = _fetch_bars(entry["symbol"], dcfg["history_days"])
        etf_rs[entry["symbol"]] = (
            weighted_rs_score([float(b["close"]) for b in bars]) if bars else None
        )
    ranked_etfs = sorted(
        (s for s, v in etf_rs.items() if v is not None),
        key=lambda s: etf_rs[s],
        reverse=True,
    )
    etf_rank = {symbol: i + 1 for i, symbol in enumerate(ranked_etfs)}

    holdings_by_etf: Dict[str, Dict[str, Any]] = {}
    owners: Dict[str, List[Dict[str, Any]]] = {}
    provider_sources = 0
    configured_sources = 0
    empty_etfs: List[str] = []

    for entry in dcfg["indices"]:
        etf = entry["symbol"]
        result = get_etf_constituents(
            etf,
            cfg["max_constituents_per_etf"],
            cfg["fallback_constituents"].get(etf, []),
        )
        holdings_by_etf[etf] = result
        if not result["symbols"]:
            empty_etfs.append(etf)
            continue
        if result["source"] == "provider":
            provider_sources += 1
        else:
            configured_sources += 1
        for symbol in result["symbols"]:
            if symbol in exclude:
                continue
            owners.setdefault(symbol, []).append(
                {
                    "symbol": etf,
                    "label": entry["label"],
                    "group": entry["group"],
                    "rank": etf_rank.get(etf, 999),
                }
            )

    watchlist_only: List[str] = []
    if cfg["include_watchlist_symbols"]:
        from backend.app.services_watchlists import snapshot_symbols

        for symbol in snapshot_symbols():
            symbol = symbol.upper()
            if symbol in exclude:
                continue
            if symbol not in owners:
                owners[symbol] = []
                watchlist_only.append(symbol)

    attribution: Dict[str, Dict[str, Any]] = {}
    for symbol, etfs in owners.items():
        if not etfs:
            attribution[symbol] = {
                "sector_symbol": None,
                "sector_label": "Watchlist",
                "sector_rank": None,
            }
            continue
        sectors = [e for e in etfs if e["group"] == "Sector"] or etfs
        best = min(sectors, key=lambda e: e["rank"])
        attribution[symbol] = {
            "sector_symbol": best["symbol"],
            "sector_label": best["label"],
            "sector_rank": etf_rank.get(best["symbol"]),
        }

    # Order by group strength so the cap keeps leading-group names first.
    symbols = sorted(
        owners.keys(),
        key=lambda s: (attribution[s]["sector_rank"] or 999, s),
    )
    capped = symbols[: cfg["max_symbols"]]
    return {
        "symbols": capped,
        "attribution": attribution,
        "etf_rank": etf_rank,
        "dropped": len(symbols) - len(capped),
        "watchlist_only": watchlist_only,
        "holdings_source": (
            "provider"
            if provider_sources and not configured_sources
            else "configured"
            if configured_sources and not provider_sources
            else "mixed"
            if provider_sources or configured_sources
            else "unavailable"
        ),
        "etfs_without_holdings": empty_etfs,
    }


def _fundamentals_key(symbol: str) -> str:
    return f"canslim:fundamentals:{symbol}"


def _cached_fundamentals(symbol: str) -> Optional[Dict[str, Any]]:
    """Warmed fundamentals for a symbol, or None when not yet fetched.

    Reads the cache without triggering a fetch so the scan can use whatever
    the background warm-up already collected.
    """

    return cache.peek(_fundamentals_key(symbol)) or None


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    def compute() -> Dict[str, Any]:
        from quant_analysis.integrations.fundamentals import fetch_stock_fundamentals

        data = fetch_stock_fundamentals(symbol)
        _record_observation(symbol, data)
        return data

    return cache.get_or_compute(
        _fundamentals_key(symbol), TTL_FUNDAMENTALS, compute
    )


def _record_observation(symbol: str, data: Dict[str, Any]) -> None:
    """Persist a dated fundamentals row so trends accrue from our own data."""

    from backend.app import storage
    from quant_analysis.storage.snapshots import last_trading_date

    try:
        storage.record_fundamentals_observation(symbol, last_trading_date(), data)
    except Exception:
        logger.info("Fundamentals history write failed for %s", symbol)


def _trend_observations(symbols: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    from backend.app import storage

    try:
        return storage.get_fundamentals_trends(symbols)
    except Exception:
        logger.info("Fundamentals trend lookup failed")
        return {}


def warm_fundamentals_universe() -> Dict[str, Any]:
    """Fetch fundamentals for every candidate, paced, in the background.

    Run off the scheduler rather than a web request: once warmed, the
    24-hour cache means every scanned candidate carries full CAN SLIM
    letters instead of only the technical shortlist. Each fetch also
    appends a dated observation, which is what makes sponsorship and
    growth trends possible at all.
    """

    cfg = canslim_config()
    universe = _build_universe(cfg)
    fetched = 0
    failed = 0
    skipped = 0
    for i, symbol in enumerate(universe["symbols"]):
        if _cached_fundamentals(symbol) is not None:
            skipped += 1
            continue
        try:
            get_fundamentals(symbol)
            fetched += 1
        except Exception:
            failed += 1
            logger.info("Fundamentals warm-up failed for %s", symbol)
        if i < len(universe["symbols"]) - 1 and cfg["fundamentals_pause_seconds"]:
            time.sleep(cfg["fundamentals_pause_seconds"])
    logger.info(
        "Fundamentals warm-up: %d fetched, %d already cached, %d failed",
        fetched,
        skipped,
        failed,
    )
    return {
        "universe": len(universe["symbols"]),
        "fetched": fetched,
        "cached": skipped,
        "failed": failed,
    }


def _market_state() -> str:
    dcfg = direction_config()
    bars = _fetch_bars(dcfg["benchmark"], dcfg["history_days"])
    if not bars:
        return "unavailable"
    return classify_market_state(bars, dcfg)["state"]


def _leader_item(
    symbol: str,
    bars: List[Dict[str, Any]],
    fundamentals: Dict[str, Any],
    rs_percentile: Optional[float],
    universe_size: Optional[int],
    market_state: str,
    cfg: Dict[str, Any],
    observations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    result = evaluate_stock_canslim(
        bars,
        fundamentals,
        rs_percentile,
        universe_size,
        market_state,
        cfg,
        observations=observations,
    )
    closes = [float(b["close"]) for b in bars]
    change_pct = (
        (closes[-1] / closes[-2] - 1.0) * 100.0
        if len(closes) > 1 and closes[-2]
        else None
    )
    label = fundamentals.get("name") or symbol
    return {
        "symbol": symbol,
        "name": fundamentals.get("name"),
        "readiness": result["readiness"],
        "readiness_label": result["readiness_label"],
        "score": result["score"],
        "scorecard": result["rows"],
        "rs_percentile": result["rs_percentile"],
        "off_high_pct": result["off_high_pct"],
        "quarterly_eps_growth_pct": _round(fundamentals.get("quarterly_eps_growth_pct")),
        "quarterly_revenue_growth_pct": _round(
            fundamentals.get("quarterly_revenue_growth_pct")
        ),
        "annual_eps_growth_pct": _round(fundamentals.get("annual_eps_growth_pct")),
        "roe_pct": _round(fundamentals.get("roe_pct")),
        "institutional_pct": _round(fundamentals.get("institutional_pct")),
        "breakout": result["breakout"],
        "base": result.get("base"),
        "eps_acceleration": result.get("eps_acceleration"),
        "sales_acceleration": result.get("sales_acceleration"),
        "sponsorship_trend": result.get("sponsorship_trend"),
        "entry": result.get("entry"),
        "last_close": round(closes[-1], 2) if closes else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "narrative": build_stock_narrative(label, result, market_state),
        "missing": list(fundamentals.get("missing", [])),
        "_result": result,
    }


def _round(value: Any) -> Optional[float]:
    return round(float(value), 1) if isinstance(value, (int, float)) else None


def _prescreen_score(
    bars: List[Dict[str, Any]], rs_percentile: Optional[float], cfg: Dict[str, Any]
) -> float:
    """Cheap technical merit used to pick who gets a fundamentals fetch.

    Fetching company financials for a 150-name universe would be slow and
    rude to the provider, so the funnel screens on price and volume first -
    the same order O'Neil worked in - and verifies fundamentals on the
    short list.
    """

    score = rs_percentile or 0.0
    breakout = detect_breakout(bars, cfg)
    if breakout:
        score += 40.0
    closes = [float(b["close"]) for b in bars]
    window = closes[-252:]
    if window:
        off_high = (closes[-1] / max(window) - 1.0) * 100.0
        if off_high >= -cfg["new_high_near_pct"]:
            score += 15.0
    return score


_SKIPPED_FUNDAMENTALS = {
    "quarterly_eps_growth_pct": None,
    "quarterly_revenue_growth_pct": None,
    "annual_eps_growth_pct": None,
    "roe_pct": None,
    "institutional_pct": None,
    "fetch_skipped": True,
    "missing": [],
}


def _scan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    universe = _build_universe(cfg)
    symbols = universe["symbols"]
    dcfg = direction_config()
    market_state = _market_state()

    # Stage 1: bars for the whole universe, fetched concurrently. Cached
    # per symbol, so only a cold scan pays the full cost.
    bars_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    unavailable: List[str] = []
    with ThreadPoolExecutor(max_workers=cfg["fetch_workers"]) as pool:
        futures = {
            pool.submit(_fetch_bars, symbol, dcfg["history_days"]): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                bars = future.result()
            except Exception:
                logger.info("Bar fetch failed for %s", symbol)
                bars = []
            if len(bars) < 120:
                unavailable.append(symbol)
                continue
            bars_by_symbol[symbol] = bars

    rs_raw = {
        symbol: weighted_rs_score([float(b["close"]) for b in bars])
        for symbol, bars in bars_by_symbol.items()
    }
    ranked = sorted(
        (s for s, v in rs_raw.items() if v is not None), key=lambda s: rs_raw[s]
    )
    n_ranked = len(ranked)
    percentiles = {
        symbol: (idx / (n_ranked - 1) * 100.0 if n_ranked > 1 else 50.0)
        for idx, symbol in enumerate(ranked)
    }

    # Stage 2: fundamentals only for the strongest technical candidates.
    shortlist = sorted(
        bars_by_symbol,
        key=lambda s: _prescreen_score(bars_by_symbol[s], percentiles.get(s), cfg),
        reverse=True,
    )[: cfg["fundamentals_top_n"]]

    # Anything already warmed by the background job is free, so the
    # shortlist only bounds *new* network fetches: in steady state every
    # candidate carries full fundamentals.
    fundamentals_by_symbol: Dict[str, Dict[str, Any]] = {}
    for symbol in bars_by_symbol:
        cached = _cached_fundamentals(symbol)
        if cached is not None:
            fundamentals_by_symbol[symbol] = cached

    to_fetch = [s for s in shortlist if s not in fundamentals_by_symbol]
    for i, symbol in enumerate(to_fetch):
        try:
            fundamentals_by_symbol[symbol] = get_fundamentals(symbol)
        except Exception:
            logger.info("Fundamentals fetch failed for %s", symbol)
            continue
        if i < len(to_fetch) - 1 and cfg["fundamentals_pause_seconds"]:
            time.sleep(cfg["fundamentals_pause_seconds"])

    observations = _trend_observations(list(bars_by_symbol))

    items = []
    for symbol, bars in bars_by_symbol.items():
        fundamentals = fundamentals_by_symbol.get(
            symbol, {**_SKIPPED_FUNDAMENTALS, "symbol": symbol}
        )
        item = _leader_item(
            symbol,
            bars,
            fundamentals,
            percentiles.get(symbol),
            n_ranked,
            market_state,
            cfg,
            observations=observations.get(symbol, []),
        )
        item["fundamentals_fetched"] = symbol in fundamentals_by_symbol
        item.update(universe["attribution"].get(symbol, {}))
        items.append(item)

    items.sort(
        key=lambda i: (
            _READINESS_ORDER.get(i["readiness"], 9),
            -(i["rs_percentile"] or 0),
        )
    )

    as_of = None
    if bars_by_symbol:
        as_of = max(bars[-1]["date"] for bars in bars_by_symbol.values())

    sectors = sorted(
        {
            (i.get("sector_symbol") or "", i.get("sector_label") or "Watchlist")
            for i in items
        },
        key=lambda pair: pair[1],
    )

    return {
        "as_of": as_of,
        "provisional": _market_open_now(),
        "market_state": market_state,
        "market_state_label": STATE_LABELS.get(market_state, "Unavailable"),
        "gate_open": market_state in ("confirmed_uptrend", "uptrend_under_pressure"),
        "gate_message": _GATE_MESSAGES.get(market_state, _GATE_MESSAGES["unavailable"]),
        "items": items,
        "universe": symbols,
        "universe_size": len(symbols),
        "scanned": len(bars_by_symbol),
        "fundamentals_scanned": len(fundamentals_by_symbol),
        "holdings_source": universe["holdings_source"],
        "etfs_without_holdings": universe["etfs_without_holdings"],
        "dropped_for_capacity": universe["dropped"],
        "sectors": [
            {"symbol": s or None, "label": label} for s, label in sectors
        ],
        "excluded": list(cfg["exclude"]),
        "unavailable": unavailable,
        "stop_loss_pct": cfg["stop_loss_pct"],
        "methodology": (
            "Seven-letter CAN SLIM scan across the holdings of every tracked market and sector ETF, "
            "so candidates come from all market sections rather than one watchlist. Price and volume are "
            f"screened first for the whole universe; company fundamentals are then fetched for the top "
            f"{cfg['fundamentals_top_n']} technical candidates and the rest are marked technical-only. "
            f"C: quarterly EPS {cfg['quarterly_growth_met_pct']:.0f}%+ YoY. A: annual EPS "
            f"{cfg['annual_growth_met_pct']:.0f}%+/yr and ROE {cfg['roe_met_pct']:.0f}%+. N: 52-week-high "
            f"breakout on {cfg['breakout_volume_ratio']}x average volume. S: up/down volume above 1.00. "
            f"L: relative strength {cfg['rs_met_percentile']:.0f}th percentile+ (12-month, recent quarter "
            "double-weighted) within the scanned universe. I: institutional ownership with buying power left. "
            "M: the follow-through-day market state; no buy flags while the market is in a correction. "
            "Each stock is credited to its strongest holding group, since O'Neil bought leaders of leading "
            "groups. Fundamentals are best-effort Yahoo data - unavailable letters are labeled, never guessed."
        ),
    }


def _cached_scan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cache.get_or_compute("canslim:leaders", TTL_LEADERS, lambda: _scan(cfg))


def get_leaders() -> Dict[str, Any]:
    cfg = canslim_config()
    payload = _cached_scan(cfg)
    try:
        _maybe_evaluate_stock_signals()
    except Exception:
        logger.exception("Lazy stock-signal evaluation failed")
    return {
        **payload,
        "items": [{k: v for k, v in i.items() if k != "_result"} for i in payload["items"]],
    }


def get_stock_detail(symbol: str) -> Optional[Dict[str, Any]]:
    cfg = canslim_config()
    symbol = symbol.upper()
    if symbol in cfg["exclude"]:
        return None
    dcfg = direction_config()
    bars = _fetch_bars(symbol, dcfg["history_days"])
    if len(bars) < 120:
        return None
    market_state = _market_state()
    fundamentals = get_fundamentals(symbol)
    item = _leader_item(symbol, bars, fundamentals, None, None, market_state, cfg)
    item.pop("_result", None)
    item["fundamentals_fetched"] = True
    # Reuse the cached scan's attribution when the symbol was in it, so a
    # detail view names the same leading group as the table row.
    try:
        cached = cache.get_or_compute("canslim:leaders", TTL_LEADERS, lambda: _scan(cfg))
        match = next((i for i in cached["items"] if i["symbol"] == symbol), None)
        if match:
            for key in ("sector_symbol", "sector_label", "sector_rank"):
                item[key] = match.get(key)
    except Exception:
        logger.info("Attribution lookup failed for %s", symbol)
    return item


_STOCK_EVAL_LOCK = None
_LAST_STOCK_EVAL_SESSION: Optional[str] = None


def _maybe_evaluate_stock_signals() -> None:
    """Evaluate once per completed session from whichever caller gets here first."""

    global _STOCK_EVAL_LOCK, _LAST_STOCK_EVAL_SESSION
    from threading import Lock

    if _STOCK_EVAL_LOCK is None:
        _STOCK_EVAL_LOCK = Lock()
    dcfg = direction_config()
    benchmark_bars = _completed_bars(_fetch_bars(dcfg["benchmark"], dcfg["history_days"]))
    if not benchmark_bars:
        return
    session = benchmark_bars[-1]["date"]
    with _STOCK_EVAL_LOCK:
        if _LAST_STOCK_EVAL_SESSION == session:
            return
        _LAST_STOCK_EVAL_SESSION = session
    evaluate_stock_signals()


def evaluate_stock_signals() -> Dict[str, Any]:
    """Persist fresh buy-candidate breakouts for the last completed session.

    Idempotent via the direction_signals unique constraint; delivery reuses
    the direction channels and only fires for newly inserted rows.
    """

    from backend.app import storage
    from backend.app.config import get_settings
    from backend.app.services_direction import _deliver_signals

    cfg = canslim_config()
    payload = _cached_scan(cfg)
    inserted: List[Dict[str, Any]] = []
    for item in payload["items"]:
        result = item.get("_result")
        if not result or item["readiness"] != "buy_candidate":
            continue
        bars = _completed_bars(
            _fetch_bars(item["symbol"], direction_config()["history_days"])
        )
        breakout = detect_breakout(bars, result["config"], within=1)
        # Only the session that started the advance is signal-worthy; a
        # stock mid-run resolves to an older pivot and must not re-alert.
        if not breakout or breakout.get("sessions_ago", 0) != 0:
            continue
        label = item.get("name") or item["symbol"]
        completed_result = {**result, "breakout": breakout}
        text = describe_breakout_signal(label, completed_result)
        if not text:
            continue
        row = {
            "symbol": item["symbol"],
            "label": label,
            "signal_type": "stock_breakout",
            "signal_date": breakout["date"],
            "state": payload["market_state"],
            "title": text["title"],
            "message": text["message"],
            "payload": breakout,
        }
        try:
            if storage.insert_direction_signal(row):
                inserted.append(row)
        except Exception:
            logger.exception("Failed to persist breakout signal for %s", item["symbol"])

    if inserted:
        _deliver_signals(inserted, get_settings())
    return {
        "evaluated": len(payload["items"]),
        "inserted": len(inserted),
        "as_of": payload["as_of"],
    }
