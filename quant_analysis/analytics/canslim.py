"""Per-stock CAN SLIM evaluation with O'Neil's published thresholds.

Pure functions: callers supply daily OHLCV bars, a fundamentals dict from
``quant_analysis.integrations.fundamentals``, a relative-strength percentile
computed across the scan universe, and the broad-market state from the
market-direction engine. Every threshold is configurable and surfaced.

The letters:
- C: latest quarterly EPS growth YoY, 25%+ met (O'Neil looked for 20-25%+),
  with quarterly sales growth as supporting context.
- A: annual EPS growth (multi-year average) plus 17%+ return on equity.
- N: new price highs - a fresh 52-week-high breakout on heavy volume is the
  buy trigger; near-pivot is watch territory. The qualitative half (new
  products, management) is not computable and is said so.
- S: supply and demand - volume on up days vs down days, float context, and
  the breakout day's volume vs its 50-day average.
- L: leader or laggard - weighted 12-month relative strength percentile
  within the scanned universe (recent quarter double-weighted).
- I: institutional sponsorship percent (13F-derived, lagged, labeled).
- M: market direction from the follow-through-day engine; O'Neil's rule is
  no new buys while the market is in a correction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from quant_analysis.analytics.market_direction import (
    STATE_CONFIRMED,
    STATE_CORRECTION,
    STATE_LABELS,
    STATE_PRESSURE,
    STATE_RALLY,
    STATE_UNCONFIRMED,
    compute_ema,
    pct_return,
)

DEFAULTS: Dict[str, Any] = {
    "quarterly_growth_met_pct": 25.0,
    "quarterly_growth_borderline_pct": 15.0,
    "annual_growth_met_pct": 25.0,
    "annual_growth_borderline_pct": 15.0,
    "roe_met_pct": 17.0,
    "new_high_near_pct": 5.0,
    "new_high_borderline_pct": 15.0,
    "breakout_volume_ratio": 1.4,
    "breakout_recent_sessions": 5,
    "updown_volume_met": 1.0,
    "updown_volume_borderline": 0.85,
    "rs_met_percentile": 80.0,
    "rs_borderline_percentile": 50.0,
    "inst_min_pct": 20.0,
    "inst_low_pct": 5.0,
    "inst_max_pct": 95.0,
    "min_score_buy": 4,
    "min_score_watch": 4,
    "stop_loss_pct": 8.0,
    # O'Neil's never-chase rule: past this much above the pivot, a 7-8% stop
    # sits inside the range of a normal pullback.
    "max_extension_pct": 5.0,
    # Base quality (the consolidation a breakout emerges from). O'Neil's
    # proper bases run at least five to seven weeks and correct less than
    # about a third; shallower, longer bases are the higher-quality ones.
    "base_min_sessions": 25,
    "base_max_depth_pct": 35.0,
    "base_ideal_depth_pct": 25.0,
    # Sponsorship trend from our own accrued observations.
    "sponsorship_trend_min_points": 2,
}

READINESS_LABELS = {
    "buy_candidate": "Qualified breakout setup",
    "extended": "Extended - do not chase",
    "near_pivot": "Watch - near 52-week high",
    "wait_market": "Qualified - wait for the market",
    "not_ready": "Not ready",
    "insufficient_data": "Insufficient data",
}


def _finite_number(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def merged_canslim_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    for key, value in (config or {}).items():
        if key in cfg and value is not None:
            cfg[key] = value
    return cfg


def weighted_rs_score(closes: List[float]) -> Optional[float]:
    """IBD-style 12-month relative strength raw score.

    2x the 3-month return plus the 6-, 9-, and 12-month returns; recent
    action is double-weighted the way O'Neil's RS rating emphasizes it.
    Requires a full 252-session comparison so recent listings are not given a
    synthetic 12-month rating from repeated short-horizon returns.
    """

    if len(closes) <= 252:
        return None
    r63 = pct_return(closes, 63)
    r126 = pct_return(closes, 126)
    r189 = pct_return(closes, 189)
    r252 = pct_return(closes, 252)
    if any(value is None for value in (r63, r126, r189, r252)):
        return None
    return 2.0 * r63 + r126 + r189 + r252


def detect_breakout(
    bars: List[Dict[str, Any]], cfg: Dict[str, Any], within: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """The pivot of the current 52-week-high advance, if there is one.

    A breakout session closes above the highest close of the prior 252
    sessions with volume at least ``breakout_volume_ratio`` times its own
    50-day average. Only the last ``within`` sessions are scanned (default
    from config) so stale breakouts do not read as fresh flags.

    The reported date, price, and volume ratio all belong to the same heavy-
    volume trigger session. This keeps the entry and risk math auditable.
    """

    within = within or int(cfg["breakout_recent_sessions"])
    n = len(bars)
    if n < 253:
        return None
    closes = [float(b["close"]) for b in bars]
    volumes = [float(b.get("volume") or 0) for b in bars]

    def prior_high_at(i: int) -> float:
        return max(closes[max(0, i - 252):i])

    def avg_vol_at(i: int) -> float:
        start = max(0, i - 50)
        return sum(volumes[start:i]) / max(1, i - start)

    for offset in range(0, min(within, n - 60)):
        i = n - 1 - offset
        avg_vol = avg_vol_at(i)
        if not (
            closes[i] > prior_high_at(i)
            and avg_vol > 0
            and volumes[i] >= cfg["breakout_volume_ratio"] * avg_vol
        ):
            continue
        return {
            "date": bars[i]["date"],
            "price": round(closes[i], 2),
            "prior_high": round(prior_high_at(i), 2),
            "volume_ratio": round(volumes[i] / avg_vol, 2),
            "sessions_ago": n - 1 - i,
            "stop_price": round(
                closes[i] * (1.0 - cfg["stop_loss_pct"] / 100.0), 2
            ),
        }
    return None


def compute_base_quality(
    bars: List[Dict[str, Any]], pivot_index: int, cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Length and depth of the consolidation a breakout emerged from.

    The base runs from the peak that preceded it to the pivot session. A
    proper base is long enough to have shaken out weak holders and shallow
    enough that the advance is not a recovery from collapse - the reason
    O'Neil bought breakouts from bases rather than any new high.
    """

    if pivot_index <= 0 or pivot_index >= len(bars):
        return None
    closes = [float(b["close"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    prior = closes[:pivot_index]
    if len(prior) < 10:
        return None

    prior_high = max(prior)
    peak_index = max(range(len(prior)), key=lambda i: prior[i])
    sessions = pivot_index - peak_index
    if sessions < 2:
        return None
    trough = min(lows[peak_index:pivot_index])
    depth_pct = (trough / prior_high - 1.0) * 100.0

    long_enough = sessions >= cfg["base_min_sessions"]
    shallow_enough = depth_pct >= -cfg["base_max_depth_pct"]
    if long_enough and depth_pct >= -cfg["base_ideal_depth_pct"]:
        quality = "proper"
    elif long_enough and shallow_enough:
        quality = "acceptable"
    elif not long_enough:
        quality = "short"
    else:
        quality = "deep"
    return {
        "sessions": sessions,
        "weeks": round(sessions / 5.0, 1),
        "depth_pct": round(depth_pct, 1),
        "quality": quality,
    }


def _trend_direction(values: List[Optional[float]], tolerance: float = 0.0) -> str:
    """Classify an ordered series (oldest first) as rising/falling/flat."""

    clean = [
        number
        for value in values
        if (number := _finite_number(value)) is not None
    ]
    if len(clean) < 2:
        return "unknown"
    delta = clean[-1] - clean[0]
    if delta > tolerance:
        return "rising"
    if delta < -tolerance:
        return "falling"
    return "flat"


def growth_acceleration(series: List[Optional[float]]) -> Dict[str, Any]:
    """Is quarterly growth accelerating? ``series`` is newest first.

    O'Neil wanted the growth rate itself improving, not merely positive;
    two comparable quarters are the minimum to say anything at all.
    """

    clean = [
        number
        for value in series
        if (number := _finite_number(value)) is not None
    ][:3]
    if len(clean) < 2:
        return {"direction": "unknown", "points": len(clean), "detail": None}
    ordered = list(reversed(clean))  # oldest first
    direction = "accelerating"
    for a, b in zip(ordered, ordered[1:]):
        if b < a:
            direction = None
            break
    if direction is None:
        direction = "decelerating"
        for a, b in zip(ordered, ordered[1:]):
            if b > a:
                direction = "mixed"
                break
    return {
        "direction": direction,
        "points": len(clean),
        "detail": " → ".join(f"{v:+.0f}%" for v in ordered),
    }


def sponsorship_trend(
    observations: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Direction of institutional ownership from our own dated history.

    Providers report only a lagged current percentage, so an observed trend
    only exists once this project has watched the number over time.
    """

    points = [
        {**observation, "institutional_pct": number}
        for observation in observations
        if (number := _finite_number(observation.get("institutional_pct")))
        is not None
    ]
    if len(points) < cfg["sponsorship_trend_min_points"]:
        return {"direction": "accruing", "points": len(points), "change_pct": None}
    ordered = sorted(points, key=lambda o: str(o.get("as_of", "")))
    first = float(ordered[0]["institutional_pct"])
    last = float(ordered[-1]["institutional_pct"])
    change = last - first
    counts = [
        o.get("institutional_holder_count")
        for o in ordered
        if o.get("institutional_holder_count") is not None
    ]
    return {
        "direction": _trend_direction([first, last], tolerance=0.25),
        "points": len(ordered),
        "change_pct": round(change, 2),
        "from_date": str(ordered[0].get("as_of")),
        "holder_count_direction": _trend_direction([float(c) for c in counts])
        if len(counts) >= 2
        else "unknown",
    }


def _row(letter: str, name: str, status: str, value: str, detail: str) -> Dict[str, str]:
    return {"letter": letter, "name": name, "status": status, "value": value, "detail": detail}


def _updown_volume(bars: List[Dict[str, Any]]) -> Optional[float]:
    closes = [float(b["close"]) for b in bars]
    up = down = 0.0
    for i in range(max(1, len(bars) - 50), len(bars)):
        vol = float(bars[i].get("volume") or 0)
        if closes[i] > closes[i - 1]:
            up += vol
        elif closes[i] < closes[i - 1]:
            down += vol
    if up + down <= 0:
        return None
    return up / down if down else float("inf")


def evaluate_stock_canslim(
    bars: List[Dict[str, Any]],
    fundamentals: Dict[str, Any],
    rs_percentile: Optional[float],
    universe_size: Optional[int],
    market_state: str,
    config: Optional[Dict[str, Any]] = None,
    observations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Full seven-letter evaluation plus a buy-readiness verdict."""

    cfg = merged_canslim_config(config)
    closes = [float(b["close"]) for b in bars]
    rows: List[Dict[str, str]] = []
    # A large universe is screened technically first, so most names never
    # get a fundamentals fetch. Say that plainly instead of implying the
    # data was tried and missing.
    skipped = bool(fundamentals.get("fetch_skipped"))
    no_fetch = (
        "Not fetched - this name is outside the fundamentals shortlist; open it to pull its financials."
    )

    # C - current quarterly earnings
    qtr = _finite_number(fundamentals.get("quarterly_eps_growth_pct"))
    sales = _finite_number(fundamentals.get("quarterly_revenue_growth_pct"))
    eps_accel = growth_acceleration(fundamentals.get("quarterly_eps_growth_series") or [])
    sales_accel = growth_acceleration(
        fundamentals.get("quarterly_revenue_growth_series") or []
    )
    qtr_values = fundamentals.get("quarterly_eps_values") or []
    if qtr is None and len(qtr_values) >= 5:
        # Earnings exist but the growth rate is undefined because the
        # company lost money. That is a failed criterion, not missing data:
        # O'Neil required real and rising profits.
        latest, year_ago = qtr_values[0], qtr_values[4]
        if latest <= 0:
            detail = (
                f"Latest quarter lost money ({latest:,.2f} per share); O'Neil bought profitable, "
                "accelerating earnings, so this fails outright."
            )
            value = "loss"
        else:
            detail = (
                f"Returned to profit ({latest:,.2f} per share) from a loss a year ago, so a growth rate "
                "is undefined. Promising, but it does not satisfy the criterion as written."
            )
            value = "loss → profit"
        rows.append(_row("C", "Current quarterly earnings", "not_met", value, detail))
    elif qtr is None:
        rows.append(_row("C", "Current quarterly earnings", "unavailable", "n/a",
                         no_fetch if skipped else
                         "Fewer than five quarters of comparable earnings are available for this company."))
    else:
        status = (
            "met" if qtr >= cfg["quarterly_growth_met_pct"]
            else "borderline" if qtr >= cfg["quarterly_growth_borderline_pct"]
            else "not_met"
        )
        # Accelerating growth upgrades a borderline quarter; decelerating
        # growth downgrades an otherwise-passing one. O'Neil watched the
        # rate of change, not just the level.
        if status == "borderline" and eps_accel["direction"] == "accelerating":
            status = "met"
        elif status == "met" and eps_accel["direction"] == "decelerating":
            status = "borderline"

        sales_part = f"; quarterly sales {sales:+.0f}%" if sales is not None else ""
        accel_bits = []
        if eps_accel["direction"] not in ("unknown", "mixed"):
            accel_bits.append(f"EPS growth {eps_accel['direction']} ({eps_accel['detail']})")
        if sales_accel["direction"] not in ("unknown", "mixed"):
            accel_bits.append(
                f"sales growth {sales_accel['direction']} ({sales_accel['detail']})"
            )
        accel_part = f" Trend: {'; '.join(accel_bits)}." if accel_bits else ""
        rows.append(_row(
            "C", "Current quarterly earnings", status, f"EPS {qtr:+.0f}% YoY",
            f"Latest quarterly EPS {qtr:+.0f}% vs a year ago (O'Neil wanted {cfg['quarterly_growth_met_pct']:.0f}%+){sales_part}.{accel_part}",
        ))

    # A - annual earnings growth + ROE
    annual = _finite_number(fundamentals.get("annual_eps_growth_pct"))
    roe = _finite_number(fundamentals.get("roe_pct"))
    annual_values = fundamentals.get("annual_eps_values") or []
    annual_loss = bool(annual_values) and any(value <= 0 for value in annual_values[:5])
    if annual is None and annual_loss:
        roe_part = f" ROE {roe:.0f}%." if roe is not None else ""
        rows.append(_row(
            "A", "Annual earnings growth", "not_met", "loss years",
            f"Annual earnings were negative in the recent record, so a multi-year growth rate is "
            f"undefined and the three-to-five-year track record O'Neil wanted does not exist.{roe_part}",
        ))
    elif annual is None:
        roe_part = f" ROE is {roe:.0f}%." if roe is not None else ""
        rows.append(_row("A", "Annual earnings growth", "unavailable", "n/a",
                         no_fetch if skipped
                         else f"At least three comparable years of positive annual EPS are required.{roe_part}"))
    else:
        growth_ok = annual is not None and annual >= cfg["annual_growth_met_pct"]
        growth_near = annual is not None and annual >= cfg["annual_growth_borderline_pct"]
        roe_ok = roe is not None and roe >= cfg["roe_met_pct"]
        if growth_ok and roe_ok:
            status = "met"
        elif growth_ok or roe_ok or growth_near:
            status = "borderline"
        else:
            status = "not_met"
        annual_part = f"{annual:+.0f}%/yr" if annual is not None else "n/a"
        roe_part = f"ROE {roe:.0f}%" if roe is not None else "ROE n/a"
        years = fundamentals.get("annual_growth_years")
        years_part = f" over {years} year(s)" if years else ""
        rows.append(_row(
            "A", "Annual earnings growth", status, f"{annual_part} · {roe_part}",
            f"Annual EPS growth {annual_part}{years_part} (target {cfg['annual_growth_met_pct']:.0f}%+), {roe_part} (target {cfg['roe_met_pct']:.0f}%+).",
        ))

    # N - new highs / breakout
    breakout = detect_breakout(bars, cfg)
    window = closes[-252:] if len(closes) >= 252 else []
    high_52w = max(window) if window else None
    if not high_52w:
        rows.append(_row("N", "New highs (breakout)", "unavailable", "n/a", "No price history."))
        off_high = None
    else:
        off_high = (closes[-1] / high_52w - 1.0) * 100.0
        if breakout:
            pivot_index = len(bars) - 1 - int(breakout.get("sessions_ago") or 0)
            base = compute_base_quality(bars, pivot_index, cfg)
            status = "met"
            value = f"breakout {breakout['date']} on {breakout['volume_ratio']}x vol"
            detail = (
                f"Cleared the prior 52-week high ({breakout['prior_high']}) at {breakout['price']} on "
                f"{breakout['volume_ratio']}x its 50-day average volume - the heavy-volume trigger O'Neil required."
            )
            if base:
                detail += (
                    f" Base: {base['weeks']} weeks, {base['depth_pct']:.0f}% deep ({base['quality']})."
                )
                # A breakout from a too-short or too-deep base is a weaker
                # setup than the letter alone suggests.
                if base["quality"] in ("short", "deep"):
                    status = "borderline"
                    detail += (
                        " O'Neil wanted at least five to seven weeks of consolidation correcting less "
                        "than about a third; this base does not meet that."
                    )
            else:
                status = "borderline"
                detail += (
                    " No measurable consolidation preceded the new high, so this is not a proper-base breakout."
                )
            detail += (
                " The qualitative 'new' (product, management) is not computable; check the news feed."
            )
        elif off_high >= -cfg["new_high_near_pct"]:
            status = "borderline"
            value = f"{off_high:.1f}% off 52w high"
            detail = "At the pivot zone but no heavy-volume breakout yet - watch, do not chase."
        elif off_high >= -cfg["new_high_borderline_pct"]:
            status = "not_met"
            value = f"{off_high:.1f}% off 52w high"
            detail = "Below the pivot zone. O'Neil bought strength near new highs, not discounts."
        else:
            status = "not_met"
            value = f"{off_high:.1f}% off 52w high"
            detail = "Far from a new high; a proper base must form first."
        rows.append(_row("N", "New highs (breakout)", status, value, detail))

    # S - supply and demand
    ratio = _updown_volume(bars)
    float_shares = _finite_number(fundamentals.get("float_shares"))
    if ratio is None:
        rows.append(_row("S", "Supply and demand (volume)", "unavailable", "n/a",
                         "Volume data unavailable."))
    else:
        status = (
            "met" if ratio > cfg["updown_volume_met"]
            else "borderline" if ratio >= cfg["updown_volume_borderline"]
            else "not_met"
        )
        shown = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        float_part = (
            f" Float {float_shares / 1e6:,.0f}M shares."
            if isinstance(float_shares, (int, float))
            else ""
        )
        rows.append(_row(
            "S", "Supply and demand (volume)", status, f"up/down vol {shown}",
            f"Up-day volume vs down-day volume over 50 sessions is {shown}; above 1.00 means demand absorbs supply.{float_part}",
        ))

    # L - leader or laggard
    if rs_percentile is None:
        rows.append(_row("L", "Leader or laggard", "unavailable", "n/a",
                         "Needs enough price history for the 12-month relative-strength rating."))
    else:
        status = (
            "met" if rs_percentile >= cfg["rs_met_percentile"]
            else "borderline" if rs_percentile >= cfg["rs_borderline_percentile"]
            else "not_met"
        )
        size_part = f" of {universe_size} scanned" if universe_size else ""
        rows.append(_row(
            "L", "Leader or laggard", status, f"RS {rs_percentile:.0f}th pct",
            f"Weighted 12-month relative strength ranks in the {rs_percentile:.0f}th percentile{size_part}. O'Neil bought the leaders of a group, never the cheap laggards.",
        ))

    # I - institutional sponsorship
    inst = _finite_number(fundamentals.get("institutional_pct"))
    trend = sponsorship_trend(observations or [], cfg)
    if inst is None:
        rows.append(_row("I", "Institutional sponsorship", "unavailable", "n/a",
                         no_fetch if skipped else
                         "Institutional ownership could not be fetched (13F data is quarterly and lagged)."))
    else:
        if inst > 100.0:
            # Yahoo reports holdings against float rather than shares
            # outstanding, so heavily shorted or low-float names can print
            # above 100%. Say so instead of implying impossible ownership.
            status = "borderline"
            detail = (
                f"Reported ownership is {inst:.0f}% of float - above 100%, which happens with low-float "
                "or heavily shorted names. Treat the level as unreliable and watch the direction instead."
            )
        elif cfg["inst_min_pct"] <= inst <= cfg["inst_max_pct"]:
            status = "borderline"
            detail = "The reported ownership level shows sponsorship, but level alone does not prove that sponsorship is increasing."
        elif inst > cfg["inst_max_pct"]:
            status = "borderline"
            detail = "Heavily owned - O'Neil warned that over-owned names have little buying power left."
        elif inst >= cfg["inst_low_pct"]:
            status = "borderline"
            detail = "Some sponsorship, still under-discovered."
        else:
            status = "not_met"
            detail = "Almost no institutional ownership - quality funds have not validated the name."

        # Direction matters more than level: O'Neil wanted funds
        # accumulating, not merely present.
        if trend["direction"] == "rising":
            trend_part = (
                f" Reported ownership is rising ({trend['change_pct']:+.1f} pts since {trend['from_date']}). "
                "This lagged provider series is consistent with increasing sponsorship but does not identify current-session buying."
            )
            if status == "borderline" and inst < cfg["inst_max_pct"]:
                status = "met"
        elif trend["direction"] == "falling":
            trend_part = (
                f" Reported ownership is falling ({trend['change_pct']:+.1f} pts since {trend['from_date']}). "
                "This lagged series is consistent with declining sponsorship, not live fund selling."
            )
            if status != "not_met":
                status = "not_met"
        elif trend["direction"] == "flat":
            trend_part = f" Ownership is flat since {trend['from_date']}."
        else:
            trend_part = (
                " Ownership trend is still accruing - this project records the figure daily, so the"
                " accumulation direction becomes available once a few observations exist."
            )
        value = f"{inst:.0f}% held"
        if trend["direction"] == "rising":
            value += " ↑"
        elif trend["direction"] == "falling":
            value += " ↓"
        rows.append(_row(
            "I", "Institutional sponsorship", status, value,
            f"{detail}{trend_part} (13F-derived, lags by up to a quarter.)",
        ))

    # M - market direction
    if market_state == STATE_CONFIRMED:
        status, detail = "met", "The market is in a confirmed uptrend - the environment O'Neil required before buying."
    elif market_state == STATE_PRESSURE:
        status, detail = "borderline", "Uptrend under pressure - distribution is building; new buys need extra caution."
    elif market_state == STATE_RALLY:
        status, detail = "borderline", "Rally attempt in progress - wait for a follow-through day before full positions."
    elif market_state == STATE_CORRECTION:
        status, detail = "not_met", "Market in correction - O'Neil's rule: three of four stocks follow the market, so no new buys."
    elif market_state == STATE_UNCONFIRMED:
        status, detail = "borderline", "The available history began in an uptrend, but no confirming follow-through day was observed in-window."
    else:
        status, detail = "unavailable", "Broad market state unavailable."
    rows.append(_row(
        "M", "Market direction", status, STATE_LABELS.get(market_state, "n/a"), detail
    ))

    met = sum(1 for r in rows if r["status"] == "met")
    scored = sum(1 for r in rows if r["status"] != "unavailable")
    gate_open = market_state in (STATE_CONFIRMED, STATE_PRESSURE)
    # A stock stretched far above its 20-day EMA is extended from any sane
    # entry, even when it sits at a new high, so it is never "near pivot".
    ema20 = compute_ema(closes, 20)[-1] if len(closes) >= 20 else None
    stretched = bool(
        ema20 and (closes[-1] / ema20 - 1.0) * 100.0 > cfg["max_extension_pct"]
    )
    near_pivot = (
        off_high is not None
        and off_high >= -cfg["new_high_near_pct"]
        and not stretched
    )

    # Extension from the pivot decides whether a qualified breakout is still
    # buyable. Buying 10% past the pivot with a 7-8% stop is how a correct
    # signal turns into a stop-out on an ordinary pullback.
    entry: Optional[Dict[str, Any]] = None
    extension_pct = None
    if breakout and closes:
        extension_pct = (closes[-1] / float(breakout["price"]) - 1.0) * 100.0
        buy_limit = float(breakout["price"]) * (1.0 + cfg["max_extension_pct"] / 100.0)
        within = 0.0 <= extension_pct <= cfg["max_extension_pct"]
        entry_status = (
            "buyable" if within else "below_pivot" if extension_pct < 0 else "extended"
        )
        purchase_stop = closes[-1] * (1.0 - cfg["stop_loss_pct"] / 100.0)
        entry = {
            "status": entry_status,
            "pivot": breakout["price"],
            "buy_limit": round(buy_limit, 2),
            "stop_price": round(purchase_stop, 2),
            "extension_pct": round(extension_pct, 1),
            "detail": (
                f"{extension_pct:+.1f}% from the {breakout['price']} trigger; the buy range runs to "
                f"{buy_limit:.2f}. An {cfg['stop_loss_pct']:.0f}% stop from a purchase here is {purchase_stop:.2f}."
                if within
                else (
                    f"Price is {abs(extension_pct):.1f}% below the {breakout['price']} breakout trigger, so the breakout has not held. "
                    "It is not a current buy setup."
                    if extension_pct < 0
                else (
                    f"Already {extension_pct:+.1f}% past the {breakout['price']} pivot - beyond O'Neil's "
                    f"{cfg['max_extension_pct']:.0f}% chase limit. Wait for a new base or a constructive pullback."
                )
                )
            ),
        }

    c_row = next((row for row in rows if row["letter"] == "C"), None)
    a_row = next((row for row in rows if row["letter"] == "A"), None)
    n_row = next((row for row in rows if row["letter"] == "N"), None)
    l_row = next((row for row in rows if row["letter"] == "L"), None)
    fundamentals_ready = bool(
        not skipped
        and c_row
        and a_row
        and c_row["status"] != "unavailable"
        and a_row["status"] != "unavailable"
    )
    fundamentals_qualified = bool(
        fundamentals_ready
        and c_row["status"] in {"met", "borderline"}
        and a_row["status"] in {"met", "borderline"}
    )
    leader_qualified = bool(l_row and l_row["status"] == "met")
    breakout_qualified = bool(breakout and n_row and n_row["status"] == "met")

    if scored < 4 or not fundamentals_ready:
        readiness = "insufficient_data"
    elif (
        breakout_qualified
        and gate_open
        and fundamentals_qualified
        and leader_qualified
        and met >= cfg["min_score_buy"]
    ):
        if entry and entry["status"] == "buyable":
            readiness = "buy_candidate"
        elif entry and entry["status"] == "extended":
            readiness = "extended"
        else:
            readiness = "not_ready"
    elif (
        gate_open
        and near_pivot
        and fundamentals_qualified
        and leader_qualified
        and met >= cfg["min_score_watch"]
    ):
        readiness = "near_pivot"
    elif (
        fundamentals_qualified
        and leader_qualified
        and (breakout_qualified or near_pivot)
        and met >= cfg["min_score_watch"]
        and not gate_open
    ):
        readiness = "wait_market"
    else:
        readiness = "not_ready"

    base = (
        compute_base_quality(
            bars, len(bars) - 1 - int(breakout.get("sessions_ago") or 0), cfg
        )
        if breakout
        else None
    )
    return {
        "rows": rows,
        "score": {"met": met, "scored": scored, "total": len(rows)},
        "readiness": readiness,
        "readiness_label": READINESS_LABELS[readiness],
        "breakout": breakout,
        "base": base,
        "eps_acceleration": eps_accel["direction"],
        "sales_acceleration": sales_accel["direction"],
        "sponsorship_trend": trend["direction"],
        "entry": entry,
        "off_high_pct": round(off_high, 1) if off_high is not None else None,
        "rs_percentile": round(rs_percentile, 0) if rs_percentile is not None else None,
        "stop_loss_pct": cfg["stop_loss_pct"],
        "config": cfg,
    }


def build_stock_narrative(
    label: str, result: Dict[str, Any], market_state: str
) -> List[str]:
    lines: List[str] = []
    readiness = result["readiness"]
    score = result["score"]
    breakout = result.get("breakout")
    cfg = result["config"]

    entry = result.get("entry")
    if readiness == "buy_candidate" and breakout:
        lines.append(
            f"{label} is a qualified CAN SLIM breakout setup: it broke out to a new 52-week high on "
            f"{breakout['date']} at {breakout['price']} on {breakout['volume_ratio']}x average volume, "
            f"with {score['met']} of {score['scored']} scored criteria met and the market gate open."
        )
        if entry:
            lines.append(
                f"Entry discipline: buy within {cfg['max_extension_pct']:.0f}% of the trigger (up to "
                f"{entry['buy_limit']}), currently {entry['extension_pct']:+.1f}% from it. The stop belongs "
                f"{cfg['stop_loss_pct']:.0f}% below the actual purchase price; at the current snapshot that is "
                f"{entry['stop_price']}."
            )
        else:
            lines.append(
                f"O'Neil's risk rule: cut the loss at {cfg['stop_loss_pct']:.0f}% below the actual fill, no exceptions."
            )
    elif readiness == "extended" and breakout and entry:
        lines.append(
            f"{label} qualifies on {score['met']} of {score['scored']} criteria and broke out on "
            f"{breakout['date']}, but it is now {entry['extension_pct']:+.1f}% past the {breakout['price']} "
            f"pivot - beyond the {cfg['max_extension_pct']:.0f}% chase limit."
        )
        lines.append(
            f"Buying here would require a {cfg['stop_loss_pct']:.0f}% stop near {entry['stop_price']}, inside "
            "the range of an ordinary pullback toward the trigger. Wait for a new base or constructive pullback."
        )
    elif readiness == "near_pivot":
        lines.append(
            f"{label} sits near its pivot ({result['off_high_pct']}% off the 52-week high) with "
            f"{score['met']} of {score['scored']} criteria met. Watch for a heavy-volume breakout; do not chase without one."
        )
    elif readiness == "wait_market":
        lines.append(
            f"{label} qualifies on {score['met']} of {score['scored']} criteria, but the market is not in an uptrend. "
            "O'Neil's rule: three of four stocks follow the market - build the watchlist, buy after a follow-through day."
        )
    elif readiness == "insufficient_data":
        lines.append(
            f"Too few criteria could be scored for {label} to call it either way; fundamental data is unavailable."
        )
    else:
        lines.append(
            f"{label} does not currently qualify: {score['met']} of {score['scored']} criteria met."
        )
    return lines


def describe_breakout_signal(
    label: str, result: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """Title and message for a persisted stock-breakout signal."""

    breakout = result.get("breakout")
    if not breakout:
        return None
    score = result["score"]
    c_row = next((r for r in result["rows"] if r["letter"] == "C"), None)
    l_row = next((r for r in result["rows"] if r["letter"] == "L"), None)
    extras = []
    if c_row and c_row["status"] != "unavailable":
        extras.append(c_row["value"])
    if l_row and l_row["status"] != "unavailable":
        extras.append(l_row["value"])
    extra = f" ({', '.join(extras)})" if extras else ""
    return {
        "title": f"{label}: 52-week-high breakout on {breakout['volume_ratio']}x volume",
        "message": (
            f"{label} cleared its prior 52-week high ({breakout['prior_high']}) at {breakout['price']} on "
            f"{breakout['volume_ratio']}x its 50-day average volume, with {score['met']} of {score['scored']} CAN SLIM "
            f"criteria met{extra}. If bought at the {breakout['price']} trigger, O'Neil's discipline puts the "
            f"{result['stop_loss_pct']:.0f}% stop at {breakout['stop_price']}; use the actual fill for any later purchase. "
            "Leading-stock breakouts after a follow-through "
            "day are the strongest confirmation that the market bottom is real."
        ),
    }
