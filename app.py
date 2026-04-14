import re
import calendar
from collections import Counter
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from html import escape, unescape
from textwrap import shorten, dedent
from quant_analysis.analytics.visualization import (
    build_gamma_gap_plot,
    compute_gamma_gap_metrics,
    describe_gamma_gap,
    generate_binomial_tree,
    interpret_net_gex,
    plot_binomial_tree,
    plot_put_call_ratios,
    plot_volume_spikes_stacked,
)
from quant_analysis.services.ai_analysis import openai_query, render_model_selection
from quant_analysis.services.market_data import (
    get_expirations,
    get_option_chain,
    get_stock_quote,
    compute_put_call_ratios,
    compute_unusual_spikes,
    load_options_data,
    fetch_and_filter_rss,
    get_liquidity_metrics,
    get_futures_quotes,
    get_bid_to_cover,
    get_bond_yield_info,
    get_vix_info,
    fetch_net_gex_for_expiration,
)
from quant_analysis.storage.db import (
    init_db,
    load_analyses,
    load_gamma_gap_history,
    save_gamma_gap_results,
)


METRIC_COLOR_THEMES = {
    "gamma exposure": {
        "label": "Gamma Exposure",
        "sequence": ["#38bdf8", "#0ea5e9", "#0284c7", "#7dd3fc", "#22d3ee", "#164e63"],
        "single": "#38bdf8",
    },
    "delta exposure": {
        "label": "Delta Exposure",
        "sequence": ["#f97316", "#fb923c", "#facc15", "#f59e0b", "#fbbf24", "#c2410c"],
        "single": "#f97316",
    },
    "open interest": {
        "label": "Open Interest",
        "sequence": ["#22c55e", "#16a34a", "#4ade80", "#86efac", "#15803d", "#065f46"],
        "single": "#22c55e",
    },
    "volume": {
        "label": "Volume",
        "sequence": ["#a855f7", "#c084fc", "#d946ef", "#f472b6", "#7c3aed", "#4c1d95"],
        "single": "#a855f7",
    },
}


DEFAULT_WATCHLIST = [
    "SPY",
    "SPX",
    "AAPL",
    "GOOGL",
    "PLTR",
    "TEM",
    "AMD",
    "NVDA",
    "TSLA",
    "VST",
    "META",
    "SMCI",
    "SOFI",
    "TSM",
    "AVGO",
    "MU",
    "HOOD",
    "BULL",
    "BE",
    "INTC",
    "QQQ",
]

BULLISH_TERMS = {
    "upgrade",
    "beats",
    "beat",
    "raised guidance",
    "surge",
    "record",
    "bullish",
    "rally",
    "optimism",
    "growth",
}

BEARISH_TERMS = {
    "downgrade",
    "miss",
    "cuts guidance",
    "slump",
    "selloff",
    "lawsuit",
    "bearish",
    "warning",
    "slowdown",
    "decline",
}


SIGNAL_LEGEND = [
    (
        "Supportive",
        "Tailwind — positioning and tape both leaning with the idea, so scaling in is safer.",
    ),
    (
        "Neutral",
        "Balanced — mixed cues suggest pacing entries and waiting for another pillar to join.",
    ),
    (
        "Adverse",
        "Headwind — opposing flow/sentiment means the setup is fighting the tape right now.",
    ),
]


def _initialise_symbol_state() -> None:
    """Ensure ticker-related session state survives reruns during rebase conflict fixes."""

    default_symbol = DEFAULT_WATCHLIST[0]
    state = st.session_state
    state.setdefault("active_ticker", default_symbol)
    state.setdefault("manual_ticker", state["active_ticker"])
    state.setdefault(
        "watchlist_choice",
        state["active_ticker"] if state["active_ticker"] in DEFAULT_WATCHLIST else default_symbol,
    )
    if state["watchlist_choice"] not in DEFAULT_WATCHLIST:
        state["watchlist_choice"] = default_symbol


def _format_expiration_option(expiration: str) -> str:
    try:
        dt = datetime.strptime(expiration, "%Y-%m-%d")
        return dt.strftime("%b %d, %Y (%a)")
    except Exception:
        return expiration


def _request_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _render_expiration_calendar(
    exp_pairs: list[tuple[str, datetime.date]],
    ticker: str,
    ui,
) -> list[str]:
    exp_pairs = sorted(exp_pairs, key=lambda item: item[1])
    if not exp_pairs:
        return []

    exp_values = [exp for exp, _ in exp_pairs]
    exp_dates = [d for _, d in exp_pairs]
    exp_lookup = {d: exp for exp, d in exp_pairs}
    first_three = exp_values[:3]

    if st.session_state.get("selected_exp_ticker") != ticker:
        st.session_state["selected_expirations_list"] = first_three
        st.session_state["selected_exp_ticker"] = ticker
        st.session_state["exp_cal_month"] = exp_dates[0].replace(day=1).isoformat()

    selected_buffer = list(st.session_state.get("selected_expirations_list", []))
    selected_buffer = [exp for exp in selected_buffer if exp in exp_values]
    st.session_state["selected_expirations_list"] = selected_buffer

    month_token = st.session_state.get("exp_cal_month", exp_dates[0].replace(day=1).isoformat())
    try:
        month_anchor = datetime.strptime(month_token, "%Y-%m-%d").date().replace(day=1)
    except Exception:
        month_anchor = exp_dates[0].replace(day=1)

    min_month = exp_dates[0].replace(day=1)
    max_month = exp_dates[-1].replace(day=1)

    nav_prev, nav_title, nav_next = ui.columns([1, 2, 1])
    with nav_prev:
        prev_month = (month_anchor.replace(day=1) - timedelta(days=1)).replace(day=1)
        if st.button("◀", key="exp_prev_month", use_container_width=True, disabled=month_anchor <= min_month):
            st.session_state["exp_cal_month"] = prev_month.isoformat()
            _request_rerun()
    with nav_title:
        st.markdown(
            f"<div style='text-align:center;padding-top:0.4rem;'><strong>{month_anchor.strftime('%B %Y')}</strong></div>",
            unsafe_allow_html=True,
        )
    with nav_next:
        month_days = calendar.monthrange(month_anchor.year, month_anchor.month)[1]
        next_month = (month_anchor.replace(day=month_days) + timedelta(days=1)).replace(day=1)
        if st.button("▶", key="exp_next_month", use_container_width=True, disabled=month_anchor >= max_month):
            st.session_state["exp_cal_month"] = next_month.isoformat()
            _request_rerun()

    weekday_cols = ui.columns(7)
    for col, label in zip(weekday_cols, ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]):
        with col:
            st.caption(label)

    first_weekday, month_days = calendar.monthrange(month_anchor.year, month_anchor.month)
    start_offset = first_weekday  # Monday=0
    grid_start = month_anchor - timedelta(days=start_offset)
    selected_set = set(selected_buffer)

    for week_idx in range(6):
        week_cols = ui.columns(7)
        for day_idx, col in enumerate(week_cols):
            day = grid_start + timedelta(days=week_idx * 7 + day_idx)
            with col:
                if day.month != month_anchor.month:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    continue
                exp_val = exp_lookup.get(day)
                if exp_val is None:
                    st.button(str(day.day), key=f"exp_day_{ticker}_{day.isoformat()}", disabled=True, use_container_width=True)
                else:
                    is_selected = exp_val in selected_set
                    label = f"{day.day}*" if is_selected else str(day.day)
                    if st.button(
                        label,
                        key=f"exp_day_{ticker}_{day.isoformat()}",
                        use_container_width=True,
                        type="primary",
                    ):
                        if exp_val in selected_set:
                            selected_set.remove(exp_val)
                        else:
                            selected_set.add(exp_val)
                        ordered = [exp for exp in exp_values if exp in selected_set]
                        st.session_state["selected_expirations_list"] = ordered
                        _request_rerun()

    selected_exps = ui.multiselect(
        "Selected expiration dates",
        options=exp_values,
        key="selected_expirations_list",
        format_func=_format_expiration_option,
        help="Calendar clicks populate this list; charts and algos use only these dates.",
    )
    ui.caption(f"Using {len(selected_exps)} selected expirations.")
    return selected_exps


def _recommended_strike_offset(ticker: str, spot: Optional[float]) -> int:
    symbol = (ticker or "").upper()
    if symbol == "SPX":
        return 500
    if spot is None or spot <= 0:
        return 35
    scaled = int(round((spot * 0.07) / 5.0) * 5)
    return max(15, min(300, scaled))


def _strike_slider_max(ticker: str, recommended_offset: int) -> int:
    symbol = (ticker or "").upper()
    if symbol == "SPX":
        return 1500
    return max(300, recommended_offset * 2)


def ensure_news_cache(limit_per_feed: int = 25) -> list[dict]:
    """Fetch and cache the filtered RSS feed once per app run."""

    cache_key = "cached_articles"
    if cache_key not in st.session_state:
        try:
            st.session_state[cache_key] = fetch_and_filter_rss(limit_per_feed=limit_per_feed)
        except Exception:
            st.session_state[cache_key] = []
    return st.session_state[cache_key]


def evaluate_gamma_signal(metrics: Optional[dict]) -> dict:
    """Translate gamma gap metrics into a discrete signal."""

    title = "Dealer Magnet"
    if not metrics:
        return {
            "title": title,
            "status": "Weak",
            "score": 0,
            "explanation": "No positive gamma magnet identified near spot.",
        }

    score = metrics.get("score", 0.0)
    positive = metrics.get("positive_zone", False)
    distance = metrics.get("distance", 0.0)
    magnet = metrics.get("magnet_strike")

    if magnet is None:
        magnet_text = "the nearby strike"
    else:
        magnet_text = f"{magnet:.1f}"

    if score >= 70 and positive:
        status = "Supportive"
        level = 2
        explanation = (
            f"Score {score:.0f}/120 with spot inside a positive gamma pocket → dealers lean toward {magnet_text}."
        )
    elif score >= 40:
        status = "Neutral"
        level = 1
        explanation = (
            f"Score {score:.0f}/120; magnet near {magnet_text} is {distance:+.2f} away so hedging pull is modest."
        )
    else:
        status = "Adverse"
        level = 0
        tail = "negative" if not positive else "weak"
        explanation = (
            f"Score {score:.0f}/120 with {tail} local gamma – magnet tug toward {magnet_text} is unreliable."
        )

    return {
        "title": title,
        "status": status,
        "score": level,
        "explanation": explanation,
    }


def evaluate_flow_signal(vol_ratio: float, oi_ratio: float, liquidity: Optional[dict]) -> dict:
    """Assess whether tape flow and liquidity confirm the trade idea."""

    title = "Flow & Liquidity"
    liq = liquidity or {}
    volume = liq.get("volume")
    spread = liq.get("bid_ask_spread_pct")
    depth = liq.get("order_book_depth")

    call_volume = vol_ratio < 1
    call_oi = oi_ratio < 1
    tight_spread = spread is not None and spread <= 0.015
    deep_book = depth is not None and depth >= 500

    positives = sum([call_volume, call_oi, tight_spread, deep_book])

    if positives >= 3:
        status = "Supportive"
        level = 2
        explanation = (
            f"Call flow dominating (V:{vol_ratio:.2f} · OI:{oi_ratio:.2f}) with tight spreads"
            + (f" {spread*100:.2f}%" if spread is not None else "")
            + " and healthy depth."
        )
    elif positives == 2:
        status = "Neutral"
        level = 1
        explanation = (
            f"Mixed confirmation — call flow {'strong' if call_volume or call_oi else 'weak'}"
            f" and liquidity metrics {('ok' if tight_spread or deep_book else 'soft')}"
            + (f" (spread {spread*100:.2f}%)" if spread is not None else "")
            + "."
        )
    else:
        status = "Adverse"
        level = 0
        explanation_parts = [
            f"Put/Call vol {vol_ratio:.2f}",
            f"OI {oi_ratio:.2f}",
        ]
        if spread is not None:
            explanation_parts.append(f"spread {spread*100:.2f}%")
        if depth is not None:
            explanation_parts.append(f"depth {int(depth):,}")
        explanation = "Flow headwinds: " + " · ".join(explanation_parts)

    return {
        "title": title,
        "status": status,
        "score": level,
        "explanation": explanation,
    }


def evaluate_sentiment_signal(ticker: str, articles: list[dict], vix: Optional[dict]) -> dict:
    """Combine curated headlines and VIX regime into a sentiment pulse."""

    title = "Macro & Sentiment"
    vix = vix or {"1d_return": 0.0, "spot": float("nan")}
    vix_chg = vix.get("1d_return") or 0.0
    ticker_lower = (ticker or "").lower()
    relevant = [a for a in articles if ticker_lower and ticker_lower in a.get("title", "").lower()]
    if len(relevant) < 5:
        relevant = articles[:5]

    counts = Counter()
    for art in relevant:
        title = art.get("title", "").lower()
        bull = any(term in title for term in BULLISH_TERMS)
        bear = any(term in title for term in BEARISH_TERMS)
        if bull and not bear:
            counts["bull"] += 1
        elif bear and not bull:
            counts["bear"] += 1

    bull = counts.get("bull", 0)
    bear = counts.get("bear", 0)
    spread = bull - bear

    if spread > 0 and vix_chg <= 1.0:
        status = "Supportive"
        level = 2
        explanation = (
            f"Headlines skew bullish ({bull}:{bear}) with VIX {vix_chg:+.1f}% → risk appetite intact."
        )
    elif abs(spread) <= 1 and abs(vix_chg) <= 2.5:
        status = "Neutral"
        level = 1
        explanation = (
            f"News mixed ({bull}:{bear}) and VIX {vix_chg:+.1f}% — neutral tone."
        )
    else:
        status = "Adverse"
        level = 0
        explanation = (
            f"Defensive skew ({bull}:{bear}) or volatility rising (VIX {vix_chg:+.1f}%)."
        )

    return {
        "title": "Macro & Sentiment",
        "status": status,
        "score": level,
        "explanation": explanation,
    }


def render_signal_card(signal: dict):
    """Render a concise status tile for the holistic dashboard."""

    palette = {
        "Supportive": "#22c55e",
        "Neutral": "#facc15",
        "Adverse": "#f97316",
        "Weak": "#94a3b8",
    }
    status = signal.get("status", "Neutral")
    color = palette.get(status, "#38bdf8")
    title = signal.get("title", "Signal")
    explanation = signal.get("explanation", "")

    st.markdown(
        f"""
        <div class='soft-card'>
            <div class='metric-footnote'>{title}</div>
            <h3 style='color:{color};margin-top:0.2rem;'>{status}</h3>
            <p class='metric-footnote' style='margin-top:0.75rem;'>{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_legend():
    """Display a small legend that explains the Supportive/Neutral/Adverse tiers."""
    for status, description in SIGNAL_LEGEND:
        st.markdown(f"- **{status}** — {description}")


def inject_global_styles():
    st.markdown(
        """
        <style>
            html, body, [data-testid="stAppViewContainer"] {
                background: radial-gradient(circle at 20% 20%, #1e293b 0%, #0f172a 40%, #020617 100%) !important;
                color: #e2e8f0;
            }

            .stApp {
                background: transparent;
                color: #e2e8f0;
            }

            [data-testid="stDecoration"], [data-testid="stToolbar"] {
                background: transparent !important;
            }

            [data-testid="stHeader"] {
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 64, 175, 0.75));
                border-bottom: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 10px 30px rgba(2, 6, 23, 0.45);
                backdrop-filter: blur(16px);
            }

            [data-testid="stSidebar"] {
                background: rgba(15, 23, 42, 0.72) !important;
                backdrop-filter: blur(20px);
            }

            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
                max-width: 1400px;
            }

            .stTabs [role="tablist"] button {
                border-radius: 12px;
                background: rgba(15, 23, 42, 0.45);
                border: 0;
                color: #cbd5f5;
                padding: 0.75rem 1.3rem;
                margin-right: 0.65rem;
                transition: all 0.2s ease-in-out;
            }

            .stTabs [role="tablist"] button:hover {
                filter: brightness(1.15);
            }

            .stTabs [role="tablist"] button[aria-selected="true"] {
                background: linear-gradient(135deg, #6366f1, #0ea5e9);
                color: #ffffff;
                box-shadow: 0 12px 30px rgba(14, 165, 233, 0.25);
            }

            .metric-card {
                background: rgba(15, 23, 42, 0.6);
                border-radius: 18px;
                padding: 1.2rem 1.5rem;
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 18px 38px rgba(2, 6, 23, 0.35);
                height: 100%;
            }

            .metric-card h3 {
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #94a3b8;
                margin-bottom: 0.4rem;
            }

            .metric-card p {
                font-size: 1.65rem;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.3rem;
            }

            .metric-delta {
                font-size: 0.9rem;
                color: #38bdf8;
                margin-bottom: 0.25rem;
            }

            .metric-footnote {
                font-size: 0.75rem;
                color: #cbd5f5;
                opacity: 0.85;
                margin: 0;
            }

            .legend-card {
                background: rgba(15, 23, 42, 0.55);
                border-radius: 14px;
                padding: 0.9rem 1.1rem;
                border: 1px solid rgba(148, 163, 184, 0.18);
                margin-top: 0.75rem;
            }

            .legend-row {
                display: flex;
                align-items: center;
                gap: 0.6rem;
                font-size: 0.8rem;
                color: #cbd5f5;
                margin-bottom: 0.45rem;
            }

            .legend-row:last-child {
                margin-bottom: 0;
            }

            .legend-chip {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.18rem 0.7rem;
                border-radius: 999px;
                font-weight: 600;
                font-size: 0.75rem;
                border: 1px solid rgba(148, 163, 184, 0.45);
                background: rgba(30, 64, 175, 0.16);
            }

            .soft-card {
                background: rgba(15, 23, 42, 0.58);
                border-radius: 18px;
                padding: 1.5rem;
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 18px 38px rgba(2, 6, 23, 0.35);
            }

            .soft-card h4 {
                color: #cbd5f5;
                margin-bottom: 0.5rem;
            }

            .hero-chart {
                position: relative;
                border-radius: 22px;
                padding: 1.75rem 1.6rem 1.4rem;
                background: linear-gradient(135deg, rgba(30, 64, 175, 0.35), rgba(8, 145, 178, 0.18));
                border: 1px solid rgba(125, 211, 252, 0.28);
                box-shadow: 0 28px 55px rgba(13, 148, 136, 0.28);
                overflow: hidden;
            }

            .hero-chart::after {
                content: "";
                position: absolute;
                inset: 0;
                pointer-events: none;
                border-radius: 22px;
                background: radial-gradient(circle at 85% 20%, rgba(56, 189, 248, 0.22), transparent 55%);
            }

            .hero-chart__title {
                position: relative;
                font-size: 1.35rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
                color: #f8fafc;
            }

            .hero-chart__caption {
                position: relative;
                color: rgba(226, 232, 240, 0.82);
                font-size: 0.9rem;
                margin-bottom: 1rem;
            }

            .article-card {
                background: rgba(15, 23, 42, 0.6);
                border-radius: 16px;
                padding: 1.1rem 1.2rem;
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 15px 32px rgba(2, 6, 23, 0.32);
                height: 100%;
            }

            .article-card h5 {
                font-size: 1.05rem;
                color: #f8fafc;
                margin-bottom: 0.5rem;
            }

            .article-card p {
                font-size: 0.85rem;
                color: #cbd5f5;
                line-height: 1.5;
            }

            .article-card a {
                color: #38bdf8;
            }

            .stExpander {
                background: rgba(15, 23, 42, 0.55);
                border-radius: 14px;
                border: 1px solid rgba(148, 163, 184, 0.18);
                overflow: hidden;
            }

            .stExpander > div:first-child {
                background: rgba(30, 41, 59, 0.55);
                color: #f8fafc;
            }

            .stExpander .streamlit-expanderContent {
                background: rgba(2, 6, 23, 0.3);
            }

            .stTable, .stDataFrame {
                background: transparent !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, *, delta: str | None = None, footnote: str | None = None):
    delta_html = (
        f'<div class="metric-delta">{escape(delta)}</div>' if delta else ""
    )
    foot_html = (
        f'<div class="metric-footnote">{escape(footnote)}</div>' if footnote else ""
    )
    card_html = dedent(
        f"""\
<div class="metric-card">
  <h3>{escape(title)}</h3>
  <p>{escape(value)}</p>
  {delta_html}
  {foot_html}
</div>
"""
    )
    st.markdown(card_html, unsafe_allow_html=True)


def chunked(items, size):
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def format_blurb(text: str, *, limit: int = 200) -> str:
    if not text:
        return ""
    cleaned = unescape(re.sub(r"<.*?>", "", text)).replace("\n", " ").strip()
    return shorten(cleaned, width=limit, placeholder="…")


def _normalise_option_focus(option_focus: str) -> str:
    option_focus = option_focus.lower()
    if option_focus in {"both", "combined", "all"}:
        return "both"
    if option_focus in {"call", "calls"}:
        return "call"
    if option_focus in {"put", "puts"}:
        return "put"
    return "both"


def _display_expiration_label(label: str, dte: int) -> str:
    return f"{label} · {dte} DTE"


def _ensure_position_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee exposure, OI, and volume columns exist before chart prep."""

    work_df = df.copy()

    # Normalise option type for consistent sign handling
    if "option_type" in work_df.columns:
        work_df["option_type"] = work_df["option_type"].astype(str).str.lower()

    # Contract size defaults to 100 when not supplied
    if "contract_size" not in work_df.columns:
        work_df["contract_size"] = 100
    else:
        work_df["contract_size"] = (
            work_df["contract_size"].fillna(100).replace(0, 100)
        )

    # Ensure open interest/volume exist for aggregation views
    if "open_interest" not in work_df.columns:
        alt_open_interest = work_df.get("openInterest")
        if alt_open_interest is not None:
            work_df["open_interest"] = alt_open_interest.fillna(0)
        else:
            work_df["open_interest"] = 0.0
    else:
        work_df["open_interest"] = work_df["open_interest"].fillna(0)

    if "volume" not in work_df.columns:
        alt_volume = work_df.get("trade_volume")
        if alt_volume is not None:
            work_df["volume"] = alt_volume.fillna(0)
        else:
            work_df["volume"] = 0.0
    else:
        work_df["volume"] = work_df["volume"].fillna(0)

    # Derive exposures when the upstream payload does not pre-compute them
    gamma_raw = work_df.get("gamma")
    if gamma_raw is None:
        gamma_raw = pd.Series(0.0, index=work_df.index)
        gamma_available = False
    else:
        gamma_raw = gamma_raw.fillna(0.0)
        gamma_available = not gamma_raw.eq(0).all()

    delta_raw = work_df.get("delta")
    if delta_raw is None:
        delta_raw = pd.Series(0.0, index=work_df.index)
        delta_available = False
    else:
        delta_raw = delta_raw.fillna(0.0)
        delta_available = not delta_raw.eq(0).all()

    open_interest = work_df["open_interest"].astype(float)
    contract_size = work_df["contract_size"].astype(float)

    gamma_exposure = gamma_raw * open_interest * contract_size
    delta_exposure = delta_raw * open_interest * contract_size

    if "option_type" in work_df.columns:
        put_mask = work_df["option_type"] == "put"
        if put_mask.any():
            gamma_exposure.loc[put_mask] *= -1

    existing_gamma = work_df.get("GammaExposure")
    if existing_gamma is None:
        work_df["GammaExposure"] = gamma_exposure
    else:
        existing_gamma = existing_gamma.fillna(0.0)
        needs_gamma_fix = False
        if gamma_available:
            if (gamma_exposure < 0).any() and not (existing_gamma < 0).any():
                needs_gamma_fix = True
            elif (gamma_exposure > 0).any() and not (existing_gamma > 0).any():
                needs_gamma_fix = True
        if needs_gamma_fix:
            work_df["GammaExposure"] = gamma_exposure
        else:
            work_df["GammaExposure"] = existing_gamma

    existing_delta = work_df.get("DeltaExposure")
    if existing_delta is None:
        work_df["DeltaExposure"] = delta_exposure
    else:
        existing_delta = existing_delta.fillna(0.0)
        needs_delta_fix = False
        if delta_available:
            if (delta_exposure < 0).any() and not (existing_delta < 0).any():
                needs_delta_fix = True
            if (delta_exposure > 0).any() and not (existing_delta > 0).any():
                needs_delta_fix = True
        if needs_delta_fix:
            work_df["DeltaExposure"] = delta_exposure
        else:
            work_df["DeltaExposure"] = existing_delta

    return work_df


def prepare_strike_metric(
    df_raw,
    view_mode,
    option_focus="both",
    breakout_expirations=True,
    signed_view=False,
):
    df = _ensure_position_columns(df_raw)
    df["DTE"] = df.get("DTE", 0).fillna(0).astype(int)
    df["expiration_label"] = df.get("expiration_label", "").astype(str)
    df["expiration_display"] = df.apply(
        lambda row: _display_expiration_label(row["expiration_label"], row["DTE"]), axis=1
    )

    metric_key = view_mode.lower()
    focus = _normalise_option_focus(option_focus)

    metric_map = {
        "gamma exposure": "GammaExposure",
        "delta exposure": "DeltaExposure",
        "open interest": "open_interest",
        "volume": "volume",
    }

    if metric_key not in metric_map or metric_key not in METRIC_COLOR_THEMES:
        raise ValueError(f"Unsupported metric view: {view_mode}")

    value_col = metric_map[metric_key]
    theme = METRIC_COLOR_THEMES[metric_key]
    label = theme["label"]

    work_df = df.copy()
    if focus != "both" and "option_type" in work_df.columns:
        work_df = work_df[work_df["option_type"] == focus]
    elif focus == "both" and "option_type" in work_df.columns:
        work_df = work_df[work_df["option_type"].isin(["call", "put"])]

    if value_col not in work_df.columns:
        work_df[value_col] = 0.0

    split_call_put = focus == "both" and value_col in {"open_interest", "volume"}

    if value_col in {"GammaExposure", "DeltaExposure"}:
        work_df["base_value"] = work_df[value_col] / 1e6
        work_df["display_value"] = (
            work_df["base_value"] if signed_view else work_df["base_value"].abs()
        )
        value_label = f"{label} ($M)" if signed_view else f"{label} (|$M|)"
    else:
        work_df["base_value"] = work_df[value_col]
        if split_call_put and "option_type" in work_df.columns:
            side_sign = work_df["option_type"].map({"call": 1.0, "put": -1.0}).fillna(0.0)
            work_df["display_value"] = work_df["base_value"] * side_sign
            value_label = f"{label} (Calls + / Puts -)"
        else:
            work_df["display_value"] = work_df["base_value"]
            value_label = label

    if breakout_expirations:
        group_cols = ["strike", "expiration_label", "DTE", "expiration_display"]
        if split_call_put and "option_type" in work_df.columns:
            group_cols.append("option_type")
        grouped = (
            work_df.groupby(group_cols)
            .agg(Value=("display_value", "sum"), RawValue=("base_value", "sum"))
            .reset_index()
        )

        if grouped.empty:
            empty = grouped.rename(columns={"strike": "Strike"})
            return empty[["Strike", "Value"]], value_label, label, theme

        grouped = (
            grouped.rename(
                columns={
                    "strike": "Strike",
                    "expiration_display": "Expiration",
                }
            )
            .sort_values(["DTE", "Strike"])
        )
        if "option_type" in grouped.columns:
            grouped["OptionSide"] = grouped["option_type"].str.title()
        return (
            grouped[
                [
                    col
                    for col in [
                        "Strike",
                        "Expiration",
                        "DTE",
                        "expiration_label",
                        "OptionSide",
                        "Value",
                        "RawValue",
                    ]
                    if col in grouped.columns
                ]
            ],
            value_label,
            label,
            theme,
        )

    grouped = (
        work_df.groupby("strike")
        .agg(
            RawValue=("base_value", "sum"),
            Magnitude=("display_value", "sum"),
        )
        .reset_index()
        .rename(columns={"strike": "Strike"})
    )

    if signed_view:
        grouped["Value"] = grouped["RawValue"]
    else:
        grouped["Value"] = grouped["Magnitude"]

    return grouped[["Strike", "Value", "RawValue"]], value_label, label, theme


def prepare_expiration_metric(
    df_raw,
    view_mode,
    option_focus="both",
    signed_view=False,
):
    df = _ensure_position_columns(df_raw)
    df["DTE"] = df.get("DTE", 0).fillna(0).astype(int)
    df["expiration_label"] = df.get("expiration_label", "").astype(str)
    df["expiration_display"] = df.apply(
        lambda row: _display_expiration_label(row["expiration_label"], row["DTE"]), axis=1
    )

    metric_key = view_mode.lower()
    focus = _normalise_option_focus(option_focus)

    metric_map = {
        "gamma exposure": "GammaExposure",
        "delta exposure": "DeltaExposure",
        "open interest": "open_interest",
        "volume": "volume",
    }

    if metric_key not in metric_map or metric_key not in METRIC_COLOR_THEMES:
        raise ValueError(f"Unsupported metric view: {view_mode}")

    value_col = metric_map[metric_key]
    theme = METRIC_COLOR_THEMES[metric_key]
    label = theme["label"]

    work_df = df.copy()
    if focus != "both" and "option_type" in work_df.columns:
        work_df = work_df[work_df["option_type"] == focus]
    elif focus == "both" and "option_type" in work_df.columns:
        work_df = work_df[work_df["option_type"].isin(["call", "put"])]

    if value_col not in work_df.columns:
        work_df[value_col] = 0.0

    split_call_put = focus == "both" and value_col in {"open_interest", "volume"}

    if value_col in {"GammaExposure", "DeltaExposure"}:
        work_df["base_value"] = work_df[value_col] / 1e6
        work_df["display_value"] = (
            work_df["base_value"] if signed_view else work_df["base_value"].abs()
        )
        value_label = f"{label} ($M)" if signed_view else f"{label} (|$M|)"
    else:
        work_df["base_value"] = work_df[value_col]
        if split_call_put and "option_type" in work_df.columns:
            side_sign = work_df["option_type"].map({"call": 1.0, "put": -1.0}).fillna(0.0)
            work_df["display_value"] = work_df["base_value"] * side_sign
            value_label = f"{label} (Calls + / Puts -)"
        else:
            work_df["display_value"] = work_df["base_value"]
            value_label = label

    group_cols = ["expiration_date", "expiration_label", "DTE", "expiration_display"]
    if split_call_put and "option_type" in work_df.columns:
        group_cols.append("option_type")
    grouped = (
        work_df.groupby(group_cols)
        .agg(Value=("display_value", "sum"), RawValue=("base_value", "sum"))
        .reset_index()
    )

    if grouped.empty:
        empty = grouped.rename(columns={"expiration_display": "Expiration"})
        return (
            empty[[col for col in ["Expiration", "Value"] if col in empty.columns]],
            value_label,
            label,
            theme,
        )

    chart_df = grouped.sort_values("expiration_date").rename(
        columns={"expiration_display": "Expiration"}
    )
    if "option_type" in chart_df.columns:
        chart_df["OptionSide"] = chart_df["option_type"].str.title()
    chart_df = chart_df[
        [
            col
            for col in ["Expiration", "Value", "RawValue", "DTE", "expiration_label", "OptionSide"]
            if col in chart_df.columns
        ]
    ]

    return chart_df, value_label, label, theme


init_db()

articles: list[dict] = ensure_news_cache()

# ---------------- Streamlit Config ----------------
st.set_page_config(layout="wide", page_title="Options Analytics Dashboard")

inject_global_styles()
st.title("📊 Options Analytics Dashboard")

# --- Sidebar Inputs ---
_initialise_symbol_state()

watch_choice = st.sidebar.selectbox(
    "Watchlist symbols",
    options=DEFAULT_WATCHLIST,
    key="watchlist_choice",
)

if watch_choice != st.session_state["active_ticker"]:
    st.session_state["active_ticker"] = watch_choice
    st.session_state["manual_ticker"] = watch_choice

manual_symbol_input = st.sidebar.text_input(
    "Or type a symbol",
    key="manual_ticker",
)
manual_symbol = manual_symbol_input.strip().upper()

if manual_symbol:
    if manual_symbol != st.session_state["manual_ticker"]:
        st.session_state["manual_ticker"] = manual_symbol
    if manual_symbol != st.session_state["active_ticker"]:
        st.session_state["active_ticker"] = manual_symbol
else:
    st.session_state["manual_ticker"] = ""
    st.session_state["active_ticker"] = watch_choice

ticker = st.session_state["active_ticker"].upper()
expirations = []
if ticker:
    try:
        expirations = get_expirations(
            ticker,
            st.secrets.get("TRADIER_TOKEN"),
            include_all_roots=True
        )
    except Exception:
        st.sidebar.error("Error fetching expirations.")
exp_pairs = []
for exp in expirations:
    try:
        exp_pairs.append((exp, datetime.strptime(exp, "%Y-%m-%d").date()))
    except Exception:
        continue

picker_col, picker_info_col = st.columns([1, 2], gap="small")
with picker_col:
    with st.popover("🗓 Pick Expirations", use_container_width=True):
        selected_exps = _render_expiration_calendar(exp_pairs, ticker, ui=st)
with picker_info_col:
    valid_expirations = {exp for exp, _ in exp_pairs}
    selected_state = list(st.session_state.get("selected_expirations_list", []))
    selected_state = [exp for exp in selected_state if exp in valid_expirations]
    selected_count = len(selected_state)
    st.markdown("**Expiration Selection**")
    st.caption(f"{selected_count} dates selected. Charts and algos use this list.")

selected_exps = selected_state
spot = None
if ticker:
    try:
        spot = get_stock_quote(ticker,  st.secrets.get("TRADIER_TOKEN"))
        st.sidebar.markdown(f"**Spot Price:** ***{spot:.2f}***")
    except Exception:
        st.sidebar.error("Error fetching spot price.")

recommended_offset = _recommended_strike_offset(ticker, spot)
offset_max = _strike_slider_max(ticker, recommended_offset)
if st.session_state.get("strike_offset_ticker") != ticker:
    st.session_state["strike_offset"] = recommended_offset
    st.session_state["strike_offset_ticker"] = ticker
current_offset = int(st.session_state.get("strike_offset", recommended_offset))
current_offset = max(1, min(offset_max, current_offset))
st.session_state["strike_offset"] = current_offset
offset = st.sidebar.slider(
    "Strike Range ±",
    min_value=1,
    max_value=offset_max,
    key="strike_offset",
)
if spot is not None:
    st.sidebar.caption(f"Default for {ticker}: ±{recommended_offset} (~7% of spot).")
elif ticker == "SPX":
    st.sidebar.caption("Default for SPX: ±500.")

enable_ai = st.sidebar.checkbox("Enable AI Analysis", value=True)

# --- Tabs ---
tab_names = [
    "Overview Metrics",
    "Options Positioning",
    "Gamma Gap Radar",
    "Binomial Tree",
    "Market Sentiment",
    "Market News"
]
if enable_ai:
    tab_names.append("AI Analysis")
tab_names.append("Economic Calendar")
tabs = st.tabs(tab_names)
tab1 = tabs[0]
tab2 = tabs[1]
gap_tab = tabs[2]
binom_tab = tabs[3]
sentiment_tab = tabs[4]
news_tab = tabs[5]
if enable_ai:
    ai_tab = tabs[6]
    calender_tab = tabs[7]
else:
    ai_tab = None
    calender_tab = tabs[6]

# --- Tab 1: Overview Metrics ---
with tab1:
    st.header("📈 Overview Metrics")
    tradier_token = st.secrets.get("TRADIER_TOKEN")
    if ticker and selected_exps and spot is not None:
        exp0 = selected_exps[0]
        try:
            chain0 = get_option_chain(
                ticker, exp0, tradier_token, include_all_roots=True
            )
        except Exception:
            st.error(f"Failed to fetch options for {exp0}")
            st.stop()

        df0 = pd.DataFrame(chain0)
        if "greeks" in df0.columns:
            greeks_df0 = pd.json_normalize(df0.pop("greeks"))
            df0 = pd.concat([df0, greeks_df0], axis=1)
        df0 = df0[(df0.strike >= spot - offset) & (df0.strike <= spot + offset)]

        exp_dt = datetime.strptime(exp0, "%Y-%m-%d")
        df0["expiration_date"] = pd.to_datetime(df0.get("expiration_date", exp_dt))
        df0["expiration_label"] = exp_dt.strftime("%b %d (%a)")
        df0["DTE"] = max((exp_dt.date() - datetime.utcnow().date()).days, 0)
        if "option_type" in df0.columns:
            df0["option_type"] = df0["option_type"].astype(str).str.lower()

        df_net = (
            pd.DataFrame([
                {
                    "Strike": opt["strike"],
                    "GEX": opt.get("gamma", 0) * opt.get("open_interest", 0) * opt.get("contract_size", 100),
                }
                for opt in df0.to_dict("records")
            ])
            .groupby("Strike").sum().reset_index().sort_values("Strike")
        )

        if df_net.empty:
            st.info("No gamma exposure data returned for the selected settings.")
            st.stop()

        gamma_metrics = compute_gamma_gap_metrics(df_net, spot, offset=offset)
        total_oi = df0.get("open_interest", pd.Series(dtype=float)).sum()
        magnet_row = df_net.loc[df_net["GEX"].idxmax()]
        magnet_strike = magnet_row["Strike"]
        magnet_val = magnet_row["GEX"]
        gamma_flip_mask = df_net["GEX"].shift().mul(df_net["GEX"]).lt(0)
        gamma_flips = df_net.loc[gamma_flip_mask, "Strike"].round(1).tolist()
        gamma_flip_text = ", ".join(map(str, gamma_flips)) if gamma_flips else "No flip in range"

        snapshot_container = st.container()
        snapshot_container.markdown("#### Market snapshot")
        c1, c2, c3 = snapshot_container.columns(3)
        with c1:
            delta = f"{spot - magnet_strike:+.2f} vs peak GEX"
            metric_card("Spot Price", f"${spot:.2f}", delta=delta, footnote=f"{ticker} | Exp {exp0}")
        with c2:
            metric_card(
                "Peak Net GEX",
                f"{magnet_val/1e6:.2f}M",
                footnote=f"Anchors near strike {magnet_strike:.0f}",
            )
        with c3:
            metric_card("Gamma Flip Zones", gamma_flip_text, footnote="Where dealer hedging flips sign")

        st.markdown("---")

        chart_col, insight_col = st.columns([3, 2], gap="large")
        with chart_col:
            st.subheader("Dealer positioning lens")
            chart_metric = st.radio(
                "Visualize strike dynamics",
                options=["Gamma Exposure", "Delta Exposure", "Open Interest", "Volume"],
                horizontal=True,
            )
            option_focus = st.radio(
                "Option side",
                options=["Combined", "Calls", "Puts"],
                horizontal=True,
            )
            chart_df, value_label, label, theme = prepare_strike_metric(
                df0,
                chart_metric,
                option_focus,
                breakout_expirations=False,
                signed_view=True,
            )

            if chart_df.empty:
                st.warning("No data available for this view.")
            else:
                exposures = {"Gamma Exposure", "Delta Exposure"}
                value_fmt = "+,.2f" if chart_metric in exposures else ",.0f"
                fig_gex = px.bar(
                    chart_df,
                    x="Value",
                    y="Strike",
                    orientation="h",
                    labels={"Value": value_label, "Strike": "Strike"},
                    height=620,
                    title=f"{label} (Exp {exp0})\n(±{offset} strikes around {spot:.1f})",
                )
                fig_gex.update_traces(
                    marker_color=theme["single"],
                    hovertemplate=(
                        "<b>Strike %{y}</b><br>"
                        + ("Value %{x:" + value_fmt + "}M" if chart_metric in exposures else "Value %{x:" + value_fmt + "}")
                        + "<extra></extra>"
                    ),
                )
                fig_gex.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.25)",
                    margin=dict(l=40, r=40, t=90, b=40),
                )
                xaxis_kwargs = dict(title=value_label, tickfont=dict(color="#e2e8f0"))
                if chart_metric in exposures:
                    xaxis_kwargs["tickformat"] = "+,.2f"
                    fig_gex.add_vline(x=0, line_dash="dash", line_color="#64748b")
                fig_gex.update_xaxes(**xaxis_kwargs)
                fig_gex.update_yaxes(tickfont=dict(size=14, color="#e2e8f0"))
                st.plotly_chart(fig_gex, use_container_width=True)
                st.caption("Flip between exposure, OI, and volume to see how hedging and liquidity align.")

        with insight_col:
            st.subheader("What the lens is highlighting")
            with st.container():
                st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
                st.markdown(
                    """
                    - **Net Gamma** indicates how aggressively dealers need to hedge. Positive values usually dampen price swings.
                    - **Gamma Exposure splits** (calls vs puts) surface directional imbalances in dealer positioning.
                    - **Open Interest & Volume views** spotlight where traders concentrate liquidity and fresh flow.
                    """
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
                st.markdown("**Automated read-through**")
                insights = "\n".join(f"- {line}" for line in interpret_net_gex(df_net, spot))
                st.markdown(insights)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        calls = (
            df0[df0.option_type == "call"][['strike', 'mid_iv']]
            .rename(columns={'mid_iv': 'iv_call'})
        )
        puts = (
            df0[df0.option_type == "put"][['strike', 'mid_iv']]
            .rename(columns={'mid_iv': 'iv_put'})
        )
        iv_skew_df = pd.merge(calls, puts, on='strike')
        iv_skew_df['IV Skew'] = iv_skew_df['iv_put'] - iv_skew_df['iv_call']
        iv_skew_df = iv_skew_df.sort_values('strike').reset_index(drop=True)

        st.markdown("#### IV skew lens")
        if iv_skew_df.empty:
            st.info("Skew requires overlapping call and put strikes within the selected range.")
            with st.expander("How to interpret skew", expanded=True):
                st.markdown(
                    dedent(
                        """
                        - **Positive skew** means puts are richer than calls, signalling demand for downside insurance or dealer short gamma.
                        - **Negative skew** shows calls pricing above puts, often after squeeze dynamics or call overwriting flows.
                        - Track how the slope shifts with spot — a steepening skew into lower strikes hints at intensifying crash hedging.
                        - Overlay with volume/OI spikes to confirm whether skew moves are driven by fresh trades or mark-to-market moves.
                        """
                    )
                )
            with st.expander("More skew diagnostics"):
                st.markdown(
                    dedent(
                        """
                        - Compare skew changes with the dealer gamma flip zones to gauge how hedging pressure may evolve.
                        - Watch ATM skew versus realized volatility to spot overpriced crash protection.
                        - Use the strike slider to inspect how skew reprices as spot migrates intraday.
                        """
                    )
                )
        else:
            chart_col, meta_col = st.columns([3, 2], gap="large")
            with chart_col:
                fig_skew = px.line(
                    iv_skew_df,
                    x='strike',
                    y='IV Skew',
                    markers=True,
                    title=f"Put minus Call IV\n(±{offset} around {spot:.1f})",
                    labels={'strike': 'Strike', 'IV Skew': 'IV Skew'},
                    template="plotly_dark",
                    height=360,
                )
                fig_skew.update_traces(
                    line_color="#38bdf8",
                    line_width=3,
                    marker=dict(size=7, color="#c084fc", line=dict(color="#0f172a", width=0.6)),
                    hovertemplate="Strike %{x:.0f}<br>Skew %{y:+.2%}<extra></extra>",
                )
                fig_skew.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.25)",
                    margin=dict(l=30, r=20, t=80, b=40),
                )
                fig_skew.update_yaxes(
                    title="IV Skew",
                    tickfont=dict(size=13, color="#e2e8f0"),
                    tickformat="+.1%",
                    zeroline=False,
                    gridcolor="rgba(148,163,184,0.18)",
                )
                fig_skew.update_xaxes(
                    title="Strike",
                    tickfont=dict(color="#e2e8f0"),
                    gridcolor="rgba(148,163,184,0.12)",
                )
                fig_skew.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                st.plotly_chart(fig_skew, use_container_width=True)

            with meta_col:
                atm_idx = (iv_skew_df['strike'] - spot).abs().idxmin()
                atm_row = iv_skew_df.loc[atm_idx]
                peak_row = iv_skew_df.loc[iv_skew_df['IV Skew'].idxmax()]
                trough_row = iv_skew_df.loc[iv_skew_df['IV Skew'].idxmin()]

                top_metrics = st.columns(2)
                with top_metrics[0]:
                    metric_card(
                        "ATM Skew",
                        f"{atm_row['IV Skew']*100:+.2f}%",
                        footnote=f"Strike {atm_row['strike']:.0f}",
                    )
                with top_metrics[1]:
                    metric_card(
                        "ATM Put IV",
                        f"{atm_row['iv_put']*100:.2f}%",
                        footnote=f"Call IV {atm_row['iv_call']*100:.2f}%",
                    )

                bottom_metrics = st.columns(2)
                with bottom_metrics[0]:
                    metric_card(
                        "Peak Skew",
                        f"{peak_row['IV Skew']*100:+.2f}%",
                        footnote=f"Strike {peak_row['strike']:.0f}",
                    )
                with bottom_metrics[1]:
                    metric_card(
                        "Trough Skew",
                        f"{trough_row['IV Skew']*100:+.2f}%",
                        footnote=f"Strike {trough_row['strike']:.0f}",
                    )

                with st.expander("How to interpret skew", expanded=True):
                    st.markdown(
                        dedent(
                            """
                            - **Positive skew** means puts are richer than calls, signalling demand for downside insurance or dealer short gamma.
                            - **Negative skew** shows calls pricing above puts, often after squeeze dynamics or call overwriting flows.
                            - Track how the slope shifts with spot — a steepening skew into lower strikes hints at intensifying crash hedging.
                            - Overlay with volume/OI spikes to confirm whether skew moves are driven by fresh trades or mark-to-market moves.
                            """
                        )
                    )
                with st.expander("More skew diagnostics"):
                    st.markdown(
                        dedent(
                            """
                            - Compare skew changes with the dealer gamma flip zones to gauge how hedging pressure may evolve.
                            - Watch ATM skew versus realized volatility to spot overpriced crash protection.
                            - Use the strike slider to inspect how skew reprices as spot migrates intraday.
                            """
                        )
                    )

        vol_ratio, oi_ratio = compute_put_call_ratios(df0)
        st.markdown("#### Flow diagnostics")
        diag_cols = st.columns(3)
        with diag_cols[0]:
            metric_card("Put/Call Volume", f"{vol_ratio:.2f}", footnote=">1 suggests defensive flow")
        with diag_cols[1]:
            metric_card("Put/Call Open Interest", f"{oi_ratio:.2f}", footnote=">1 = more downside hedges outstanding")
        with diag_cols[2]:
            metric_card("Total OI", f"{int(total_oi):,}", footnote="Contracts within selected strikes")

        fig = plot_put_call_ratios(vol_ratio, oi_ratio)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.25)",
        )
        st.plotly_chart(fig, use_container_width=True)

        liq_metrics = None
        try:
            liq_metrics = get_liquidity_metrics(ticker, tradier_token)
            st.markdown("#### Liquidity snapshot")
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                metric_card("Trading Volume", f"{liq_metrics['volume']:,}")
            if liq_metrics.get("bid_ask_spread_pct") is not None:
                hist = liq_metrics.get("avg_spread_pct")
                delta = None
                if hist is not None and hist:
                    delta = f"{(liq_metrics['bid_ask_spread_pct']/hist-1)*100:+.1f}% vs avg"
                with lc2:
                    metric_card(
                        "Bid-Ask Spread",
                        f"{liq_metrics['bid_ask_spread_pct']*100:.2f}%",
                        delta=delta,
                        footnote="Tighter spreads = easier executions",
                    )
            else:
                with lc2:
                    metric_card("Bid-Ask Spread", "N/A")
            if liq_metrics.get("order_book_depth") is not None:
                with lc3:
                    metric_card("Order Book Depth", f"{liq_metrics['order_book_depth']:,}")
            else:
                with lc3:
                    metric_card("Order Book Depth", "N/A")
            st.caption("Lower volume, wider spreads and shallow depth typically signal **low liquidity**.")
        except Exception as e:
            st.warning(f"Liquidity metrics unavailable: {e}")

        try:
            vix_data = get_vix_info()
        except Exception:
            vix_data = {"spot": float("nan"), "1d_return": 0.0, "5d_return": 0.0}

        st.markdown("#### Holistic trade posture")
        st.caption(
            "Supportive = tailwind, Neutral = sideways, Adverse = headwind. Wait for at least two tailwinds before sizing up."
        )
        render_signal_legend()
        articles = st.session_state.get("cached_articles", articles)
        gamma_signal = evaluate_gamma_signal(gamma_metrics)
        flow_signal = evaluate_flow_signal(vol_ratio, oi_ratio, liq_metrics)
        sentiment_signal = evaluate_sentiment_signal(ticker, articles, vix_data)
        signals = [gamma_signal, flow_signal, sentiment_signal]
        hol_cols = st.columns(3)
        for col, sig in zip(hol_cols, signals):
            with col:
                render_signal_card(sig)

        st.session_state["latest_vix"] = vix_data
        supportive = sum(sig["score"] == 2 for sig in signals)
        neutral = sum(sig["score"] == 1 for sig in signals)
        adverse = [sig for sig in signals if sig["score"] == 0]
        supportive_names = ", ".join(sig["title"] for sig in signals if sig["score"] == 2)
        neutral_names = ", ".join(sig["title"] for sig in signals if sig["score"] == 1)
        adverse_names = ", ".join(sig["title"] for sig in adverse)

        st.caption(
            f"Snapshot → {supportive} supportive · {neutral} neutral · {len(adverse)} adverse pillars."
        )

        if supportive == len(signals):
            verdict_icon = "✅"
            verdict_text = (
                f"All three pillars aligned ({supportive_names}) — favour staged entries toward the dealer magnet."
            )
        elif adverse:
            verdict_icon = "⚠️"
            verdict_text = (
                f"{adverse_names} showing headwinds — let those pillars improve before sizing LEAPS."
            )
        elif supportive == 0:
            verdict_icon = "ℹ️"
            neutral_hint = neutral_names or "Neutral reads only"
            verdict_text = (
                f"{neutral_hint} — stand aside or keep risk tiny until flow improves."
            )
        else:
            verdict_icon = "ℹ️"
            verdict_text = (
                f"{supportive_names} supportive while {neutral_names or 'remaining pillars'} neutral — scale in gradually and monitor the neutral pillar."
            )

        st.markdown(
            f"<div class='soft-card'><strong>{verdict_icon} Quant take:</strong> {verdict_text}</div>",
            unsafe_allow_html=True,
        )

        spikes_df = compute_unusual_spikes(df0)
        st.markdown("#### Unusual flow radar")
        st.dataframe(spikes_df, use_container_width=True, hide_index=True)
        fig_spikes = plot_volume_spikes_stacked(spikes_df, offset=offset, spot=spot)
        fig_spikes.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.25)",
        )
        st.plotly_chart(fig_spikes, use_container_width=True)

        st.caption(
            "Volume/OI spikes paired with gamma positioning help confirm whether fresh flow is following or fighting dealer hedging."
        )
    else:
        st.info("Select ticker, expirations, and ensure spot price loaded.")
# --- Tab 2: Options Positioning ---
with tab2:
    st.header("🎯 Options Positioning")
    if ticker and selected_exps and spot is not None:
        token = st.secrets.get("TRADIER_TOKEN")
        df = load_options_data(ticker, selected_exps, token)
        if df.empty:
            st.info("No options data available.")
        else:
            df = df.copy()
            df = df[(df.strike >= spot - offset) & (df.strike <= spot + offset)]
            if df.empty:
                st.info("No contracts within the selected strike window.")
            else:
                df["DTE"] = df["DTE"].fillna(0).astype(int)
                df["GammaExposure"] = df["GammaExposure"].fillna(0.0)
                df["DeltaExposure"] = df["DeltaExposure"].fillna(0.0)
                df["open_interest"] = df["open_interest"].fillna(0)
                df["volume"] = df["volume"].fillna(0)
                df["expiration_label"] = df["expiration_date"].dt.strftime("%b %d (%a)")
                if "option_type" in df.columns:
                    df["option_type"] = df["option_type"].astype(str).str.lower()

                dte_available = sorted(df["DTE"].unique())
                if len(dte_available) > 5:
                    focus_dte = dte_available[:5]
                    df = df[df["DTE"].isin(focus_dte)]

                if df.empty:
                    st.info("Nearest expirations did not return any contracts in range.")
                else:
                    df["abs_gamma"] = df["GammaExposure"].abs()
                    df["abs_delta"] = df["DeltaExposure"].abs()

                    for key, default in {
                        "pos_lens_mode": "Strike lens",
                        "pos_chart_metric": "Gamma Exposure",
                        "pos_option_focus": "Combined",
                    }.items():
                        st.session_state.setdefault(key, default)

                    if "option_type" in df.columns:
                        call_peak = (
                            df[df["option_type"] == "call"]
                            .groupby(["strike", "expiration_label", "DTE"])["abs_gamma"]
                            .sum()
                        )
                        put_peak = (
                            df[df["option_type"] == "put"]
                            .groupby(["strike", "expiration_label", "DTE"])["abs_gamma"]
                            .sum()
                        )
                    else:
                        call_peak = pd.Series(dtype=float)
                        put_peak = pd.Series(dtype=float)
                    exp_liquidity = (
                        df.groupby(["expiration_label", "DTE"])
                        .agg(
                            total_oi=("open_interest", "sum"),
                            total_volume=("volume", "sum"),
                            total_gamma_abs=("abs_gamma", "sum"),
                        )
                        .reset_index()
                    )

                    st.markdown("#### Positioning pulse")
                    met1, met2, met3 = st.columns(3)
                    with met1:
                        if not call_peak.empty:
                            strike, label_call, dte_call = call_peak.idxmax()
                            call_val = call_peak.max() / 1e6
                            metric_card(
                                "Call Gamma Peak",
                                f"{call_val:.2f}M",
                                footnote=f"@ {strike:.0f} · {_display_expiration_label(label_call, int(dte_call))}",
                            )
                        else:
                            metric_card("Call Gamma Peak", "N/A", footnote="No call strikes in view")
                    with met2:
                        if not put_peak.empty:
                            strike, label_put, dte_put = put_peak.idxmax()
                            put_val = put_peak.max() / 1e6
                            metric_card(
                                "Put Gamma Peak",
                                f"{put_val:.2f}M",
                                footnote=f"@ {strike:.0f} · {_display_expiration_label(label_put, int(dte_put))}",
                            )
                        else:
                            metric_card("Put Gamma Peak", "N/A", footnote="No put strikes in view")
                    with met3:
                        if not exp_liquidity.empty:
                            liq_row = exp_liquidity.loc[exp_liquidity["total_oi"].idxmax()]
                            exp_note = _display_expiration_label(
                                liq_row["expiration_label"], int(liq_row["DTE"])
                            )
                            metric_card(
                                "Most Crowded Expiration",
                                f"{int(liq_row['total_oi']):,}",
                                footnote=f"Vol {int(liq_row['total_volume']):,} · |Γ| {liq_row['total_gamma_abs']/1e6:.2f}M",
                            )
                        else:
                            metric_card("Most Crowded Expiration", "N/A", footnote="Insufficient contracts")

                    st.markdown("---")

                    chart_col, insight_col = st.columns([5, 2], gap="large")
                    with chart_col:
                        st.markdown("<div class='hero-chart'>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 class='hero-chart__title'>📈 Positioning drilldown</h3>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            "<p class='hero-chart__caption'>Explore how dealer positioning stacks across strikes or expirations with side-specific views.</p>",
                            unsafe_allow_html=True,
                        )
                        lens_options = ["Strike lens", "Expiration lens"]
                        metric_options = [
                            "Gamma Exposure",
                            "Delta Exposure",
                            "Open Interest",
                            "Volume",
                        ]
                        focus_options = ["Combined", "Calls", "Puts"]

                        with st.form("positioning_controls"):
                            lens_choice = st.radio(
                                "Slice positioning by",
                                options=lens_options,
                                horizontal=True,
                                index=lens_options.index(st.session_state["pos_lens_mode"]),
                            )
                            metric_choice = st.radio(
                                "Focus metric",
                                options=metric_options,
                                horizontal=True,
                                index=metric_options.index(st.session_state["pos_chart_metric"]),
                            )
                            focus_choice = st.radio(
                                "Option side",
                                options=focus_options,
                                horizontal=True,
                                index=focus_options.index(st.session_state["pos_option_focus"]),
                                help="Combined overlays both sides; OI/Volume show calls as + and puts as - for quick skew reads.",
                            )
                            submit_controls = st.form_submit_button(
                                "Update positioning view",
                                use_container_width=True,
                            )

                        if submit_controls:
                            st.session_state["pos_lens_mode"] = lens_choice
                            st.session_state["pos_chart_metric"] = metric_choice
                            st.session_state["pos_option_focus"] = focus_choice

                        lens_mode = st.session_state["pos_lens_mode"]
                        chart_metric = st.session_state["pos_chart_metric"]
                        option_focus = st.session_state["pos_option_focus"]

                        value_fmt = ".2f" if chart_metric in {"Gamma Exposure", "Delta Exposure"} else ",.0f"
                        signed_fmt = "+,.2f" if chart_metric in {"Gamma Exposure", "Delta Exposure"} else "+,.0f"
                        if lens_mode == "Strike lens":
                            signed_view = chart_metric in {"Gamma Exposure", "Delta Exposure"}
                            chart_df, value_label, label, theme = prepare_strike_metric(
                                df,
                                chart_metric,
                                option_focus,
                                signed_view=signed_view,
                            )
                            if chart_df.empty:
                                st.warning("No data available for this view.")
                            else:
                                custom_cols = ["Expiration", "DTE", "RawValue"]
                                has_side = "OptionSide" in chart_df.columns
                                if has_side:
                                    custom_cols.append("OptionSide")
                                custom = chart_df[custom_cols].values
                                color_field = "OptionSide" if has_side else "Expiration"
                                color_sequence = ["#34d399", "#fb7185"] if has_side else theme["sequence"]
                                fig = px.bar(
                                    chart_df,
                                    x="Value",
                                    y="Strike",
                                    orientation="h",
                                    color=color_field,
                                    labels={"Value": value_label, "Strike": "Strike"},
                                    height=700,
                                    color_discrete_sequence=color_sequence,
                                    title=f"{label} across strikes\n(±{offset} around {spot:.1f})",
                                )
                                fig.update_traces(
                                    marker_line_width=0.4,
                                    marker_line_color="rgba(15,23,42,0.8)",
                                    customdata=custom,
                                    hovertemplate=(
                                        "<b>Strike %{y}</b><br>"
                                        + "Value %{x:" + value_fmt + "}"
                                        + "<br>Expiration %{customdata[0]}"
                                        + "<br>DTE %{customdata[1]}"
                                        + "<br>Signed %{customdata[2]:" + signed_fmt + "}"
                                        + ("<br>Side %{customdata[3]}" if has_side else "")
                                        + "<extra></extra>"
                                    ),
                                )
                                fig.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(15,23,42,0.25)",
                                    margin=dict(l=40, r=40, t=90, b=40),
                                    legend_title_text=("Option Side" if has_side else "Expiration · DTE"),
                                    legend=dict(
                                        bgcolor="rgba(15,23,42,0.7)",
                                        bordercolor="rgba(148, 163, 184, 0.4)",
                                        borderwidth=1,
                                        font=dict(color="#e2e8f0"),
                                    ),
                                )
                                fig.update_xaxes(
                                    title=value_label,
                                    tickfont=dict(color="#e2e8f0"),
                                    zeroline=True,
                                    zerolinecolor="#94a3b8",
                                )
                                fig.update_yaxes(tickfont=dict(size=14, color="#e2e8f0"))
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            signed_view = chart_metric in {"Gamma Exposure", "Delta Exposure"}
                            chart_df, value_label, label, theme = prepare_expiration_metric(
                                df,
                                chart_metric,
                                option_focus,
                                signed_view=signed_view,
                            )
                            if chart_df.empty:
                                st.warning("No data available for this view.")
                            else:
                                custom_cols = ["DTE", "RawValue"]
                                has_side = "OptionSide" in chart_df.columns
                                if has_side:
                                    custom_cols.append("OptionSide")
                                custom = chart_df[custom_cols].values
                                color_field = "OptionSide" if has_side else "Expiration"
                                color_sequence = ["#34d399", "#fb7185"] if has_side else theme["sequence"]
                                fig = px.bar(
                                    chart_df,
                                    x="Expiration",
                                    y="Value",
                                    color=color_field,
                                    labels={"Value": value_label, "Expiration": "Expiration"},
                                    height=700,
                                    color_discrete_sequence=color_sequence,
                                    title=f"{label} across expirations",
                                )
                                fig.update_traces(
                                    marker_line_width=0.4,
                                    marker_line_color="rgba(15,23,42,0.8)",
                                    customdata=custom,
                                    hovertemplate=(
                                        "<b>%{x}</b><br>"
                                        + "Value %{y:" + value_fmt + "}"
                                        + "<br>DTE %{customdata[0]}"
                                        + "<br>Signed %{customdata[1]:" + signed_fmt + "}"
                                        + ("<br>Side %{customdata[2]}" if has_side else "")
                                        + "<extra></extra>"
                                    ),
                                )
                                fig.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(15,23,42,0.25)",
                                    margin=dict(l=40, r=40, t=90, b=40),
                                    showlegend=has_side,
                                )
                                fig.update_xaxes(tickfont=dict(color="#e2e8f0"))
                                fig.update_yaxes(
                                    title=value_label,
                                    tickfont=dict(color="#e2e8f0"),
                                    zeroline=True,
                                    zerolinecolor="#94a3b8",
                                )
                                st.plotly_chart(fig, use_container_width=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                    with insight_col:
                        st.subheader("Key positioning takeaways")
                        gamma_split = (
                            df.groupby("option_type")["abs_gamma"].sum() / 1e6
                            if "option_type" in df.columns
                            else pd.Series(dtype=float)
                        )
                        call_mag = float(gamma_split.get("call", 0.0))
                        put_mag = float(gamma_split.get("put", 0.0))
                        bullets = []
                        if not call_peak.empty:
                            strike, label_call, dte_call = call_peak.idxmax()
                            bullets.append(
                                f"- **Call concentration:** {strike:.0f} holds {call_peak.max()/1e6:.2f}M |Γ| ({_display_expiration_label(label_call, int(dte_call))})."
                            )
                        if not put_peak.empty:
                            strike, label_put, dte_put = put_peak.idxmax()
                            bullets.append(
                                f"- **Put concentration:** {strike:.0f} carries {put_peak.max()/1e6:.2f}M |Γ| ({_display_expiration_label(label_put, int(dte_put))})."
                            )
                        if not exp_liquidity.empty:
                            liq_row = exp_liquidity.loc[exp_liquidity["total_gamma_abs"].idxmax()]
                            bullets.append(
                                f"- **Expiry with heft:** {_display_expiration_label(liq_row['expiration_label'], int(liq_row['DTE']))} tops |Γ| at {liq_row['total_gamma_abs']/1e6:.2f}M with OI {int(liq_row['total_oi']):,}."
                            )
                        if call_mag or put_mag:
                            bullets.append(
                                f"- **Gamma balance:** Calls house {call_mag:.2f}M |Γ| vs puts {put_mag:.2f}M |Γ|; toggle sides to inspect imbalance."
                            )
                        bullets.append(
                            "- **Tip:** Use Combined to spot call/put skew at a glance, then isolate Calls or Puts to validate the driver."
                        )
                        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
                        st.markdown("\n".join(bullets))
                        st.markdown("</div>", unsafe_allow_html=True)

                        exp_detail = (
                            df.assign(
                                gamma_abs=df["GammaExposure"].abs() / 1e6,
                                delta_abs=df["DeltaExposure"].abs() / 1e6,
                            )
                            .groupby(["expiration_label", "DTE", "option_type"])
                            .agg(
                                gamma_abs=("gamma_abs", "sum"),
                                delta_abs=("delta_abs", "sum"),
                                open_interest=("open_interest", "sum"),
                                volume=("volume", "sum"),
                            )
                            .reset_index()
                        )

                        if not exp_detail.empty:
                            pivot = exp_detail.pivot_table(
                                index=["expiration_label", "DTE"],
                                columns="option_type",
                                values=["gamma_abs", "delta_abs", "open_interest", "volume"],
                                aggfunc="sum",
                                fill_value=0,
                            ).reset_index()

                            pivot.columns = [
                                "Expiration" if col == "expiration_label" else
                                "DTE" if col == "DTE" else
                                f"Call |Γ| (M)" if col == ("gamma_abs", "call") else
                                f"Put |Γ| (M)" if col == ("gamma_abs", "put") else
                                f"Call |Δ| (M)" if col == ("delta_abs", "call") else
                                f"Put |Δ| (M)" if col == ("delta_abs", "put") else
                                f"Call OI" if col == ("open_interest", "call") else
                                f"Put OI" if col == ("open_interest", "put") else
                                f"Call Volume" if col == ("volume", "call") else
                                f"Put Volume" if col == ("volume", "put") else str(col)
                                for col in pivot.columns
                            ]

                            ordered_cols = [
                                "Expiration",
                                "DTE",
                                "Call |Γ| (M)",
                                "Put |Γ| (M)",
                                "Call |Δ| (M)",
                                "Put |Δ| (M)",
                                "Call OI",
                                "Put OI",
                                "Call Volume",
                                "Put Volume",
                            ]
                            display_df = pivot[[col for col in ordered_cols if col in pivot.columns]]
                            st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
                            st.markdown("**Expiration rundown**")
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Select ticker and expirations to view positioning.")

# --- Tab 3: Gamma Gap Radar ---
with gap_tab:
    st.header("🧲 Gamma Gap Radar")
    st.caption(
        "Scan hot tickers for positive gamma magnets where spot is likely to mean-revert."
    )

    tradier_token = st.secrets.get("TRADIER_TOKEN")
    default_hot_list = ", ".join(DEFAULT_WATCHLIST)
    col_hot, col_dte, col_max = st.columns([3, 2, 1])
    hot_input = col_hot.text_input(
        "Hot tickers (comma separated)",
        value=st.session_state.get("gamma_gap_hot", default_hot_list),
        help="List tickers you want to evaluate for gamma-driven gap fills.",
    )
    st.session_state["gamma_gap_hot"] = hot_input

    dte_min_max = col_dte.slider(
        "DTE window",
        min_value=0,
        max_value=45,
        value=st.session_state.get("gamma_gap_dte", (0, 7)),
        help="Filter expirations to a DTE range so we focus on imminent dealer hedging flows.",
    )
    st.session_state["gamma_gap_dte"] = dte_min_max

    max_expirations = int(
        col_max.number_input(
            "Exp per ticker",
            min_value=1,
            max_value=4,
            value=int(st.session_state.get("gamma_gap_exp", 2)),
            help="Limit the number of expirations scanned per ticker (nearest DTE first).",
        )
    )
    st.session_state["gamma_gap_exp"] = max_expirations

    offset_scan = st.slider(
        "Strike range (±)",
        min_value=5,
        max_value=150,
        value=int(st.session_state.get("gamma_gap_offset", offset)),
        help="Gamma magnets are calculated using strikes within spot ± this range.",
    )
    st.session_state["gamma_gap_offset"] = offset_scan

    run_scan = st.button("Run Gamma Gap Scan", type="primary")

    if run_scan:
        if not tradier_token:
            st.error("Tradier token missing – unable to fetch option data.")
        else:
            tickers_to_scan = [sym.strip().upper() for sym in hot_input.split(",") if sym.strip()]
            tickers_to_scan = sorted(set(tickers_to_scan))
            if not tickers_to_scan:
                st.warning("Please provide at least one ticker symbol.")
            else:
                with st.spinner("Analysing gamma profiles..."):
                    analysis_records: list[dict] = []
                    db_rows: list[dict] = []
                    errors: list[str] = []
                    for symbol in tickers_to_scan:
                        try:
                            spot_px = get_stock_quote(symbol, tradier_token)
                        except Exception as exc:
                            errors.append(f"{symbol}: quote error ({exc})")
                            continue

                        if spot_px in (None, 0):
                            errors.append(f"{symbol}: invalid spot price")
                            continue

                        try:
                            expirations = get_expirations(
                                symbol,
                                tradier_token,
                                include_all_roots=True,
                            )
                        except Exception as exc:
                            errors.append(f"{symbol}: expirations error ({exc})")
                            continue

                        if not expirations:
                            errors.append(f"{symbol}: no expirations returned")
                            continue

                        dte_window = (min(dte_min_max), max(dte_min_max))
                        scoped_exps: list[tuple[str, int]] = []
                        for exp in expirations:
                            try:
                                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                            except ValueError:
                                continue
                            dte = (exp_date - datetime.utcnow().date()).days
                            if dte_window[0] <= dte <= dte_window[1]:
                                scoped_exps.append((exp, dte))
                        if not scoped_exps:
                            errors.append(
                                f"{symbol}: no expirations within {dte_window[0]}-{dte_window[1]} DTE"
                            )
                            continue

                        scoped_exps.sort(key=lambda x: x[1])
                        for exp, dte in scoped_exps[:max_expirations]:
                            df_net = fetch_net_gex_for_expiration(
                                symbol,
                                exp,
                                tradier_token,
                                spot_px,
                                offset=offset_scan,
                            )
                            if df_net.empty:
                                continue

                            metrics = compute_gamma_gap_metrics(df_net, spot_px, offset=offset_scan)
                            if not metrics:
                                continue

                            commentary = describe_gamma_gap(metrics)
                            record = {
                                "ticker": symbol,
                                "expiration": exp,
                                "dte": dte,
                                "spot": spot_px,
                                "settings": {
                                    "offset": offset_scan,
                                    "dte_window": dte_window,
                                },
                                **metrics,
                                "direction": "Up toward magnet" if metrics["distance"] > 0 else "Down toward magnet",
                                "df_net": df_net,
                                "commentary": commentary,
                            }
                            analysis_records.append(record)
                            row_payload = {k: v for k, v in record.items() if k not in {"df_net", "commentary"}}
                            db_rows.append(row_payload)

                    if errors:
                        st.warning("\n".join(errors))

                st.session_state["gamma_gap_results"] = analysis_records
                if db_rows:
                    try:
                        save_gamma_gap_results(db_rows)
                    except Exception as exc:
                        st.warning(f"Unable to log gamma gap results: {exc}")

    gamma_gap_results: list[dict] = st.session_state.get("gamma_gap_results", [])
    if gamma_gap_results:
        table_records = []
        for rec in gamma_gap_results:
            table_records.append(
                {
                    "Ticker": rec["ticker"],
                    "Expiration": rec["expiration"],
                    "DTE": rec["dte"],
                    "Spot": rec["spot"],
                    "Magnet": rec["magnet_strike"],
                    "Distance": rec["distance"],
                    "Score": rec["score"],
                    "Positive γ?": "Yes" if rec["positive_zone"] else "No",
                    "Direction": rec["direction"],
                }
            )
        table_df = pd.DataFrame(table_records)
        table_df = table_df.sort_values("Score", ascending=False)
        st.markdown("#### Gap fill leaderboard")
        st.dataframe(
            table_df.style.format(
                {
                    "Spot": "{:.2f}",
                    "Magnet": "{:.2f}",
                    "Distance": "{:+.2f}",
                    "Score": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        key_options = [f"{rec['ticker']} · {rec['expiration']}" for rec in gamma_gap_results]
        selection = st.selectbox(
            "Inspect details",
            options=key_options,
            help="Pick a ticker/expiry combo to review the gamma profile and commentary.",
        )

        if selection:
            sel_idx = key_options.index(selection)
            chosen = gamma_gap_results[sel_idx]
            detail_col, chart_col = st.columns([2, 3], gap="large")
            with detail_col:
                st.markdown("#### Magnet context")
                st.markdown(chosen["commentary"], unsafe_allow_html=False)
                st.metric(
                    "Gap-fill score",
                    f"{chosen['score']:.1f}/120",
                    help="Composite score factoring magnet strength, distance, and local gamma sign.",
                )
                st.metric(
                    "Gamma gradient",
                    f"{chosen['spot_gradient']:+.2f}",
                    help="Slope of net gamma at spot; steeper gradients imply stronger pull toward the magnet.",
                )
                st.markdown(
                    f"**Settings:** ±{chosen['settings']['offset']} strikes · DTE window {chosen['settings']['dte_window'][0]}–{chosen['settings']['dte_window'][1]}"
                )

            with chart_col:
                st.markdown("#### Net gamma profile")
                fig_gap = build_gamma_gap_plot(chosen["df_net"], chosen["spot"], chosen["magnet_strike"])
                st.plotly_chart(fig_gap, use_container_width=True)

    else:
        st.info("Run the scanner to populate magnet candidates.")

    history = load_gamma_gap_history(limit=10)
    if history:
        with st.expander("Recent gamma gap snapshots"):
            hist_df = pd.DataFrame(history)
            hist_df["positive_zone"] = hist_df["positive_zone"].map({1: "Yes", 0: "No"})
            st.dataframe(
                hist_df[[
                    "ts",
                    "ticker",
                    "expiration",
                    "dte",
                    "spot",
                    "magnet_strike",
                    "distance",
                    "score",
                    "positive_zone",
                ]].rename(
                    columns={
                        "ts": "Timestamp",
                        "ticker": "Ticker",
                        "expiration": "Expiration",
                        "dte": "DTE",
                        "spot": "Spot",
                        "magnet_strike": "Magnet",
                        "distance": "Distance",
                        "score": "Score",
                        "positive_zone": "Positive γ?",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

# --- Tab 4: Binomial Tree ---
with binom_tab:
    st.header("🧮 Binomial Tree")
    if ticker and expirations and spot is not None:
        st.caption("Calibrate pricing trees with live rates, implied vols, and configurable step sizes.")
        exp = st.selectbox("Expiration", expirations)
        token = st.secrets.get("TRADIER_TOKEN")
        try:
            chain = get_option_chain(ticker, exp, token, include_all_roots=True)
        except Exception:
            st.error("Failed to fetch options chain")
            chain = []

        strikes = sorted({float(opt.get("strike", 0)) for opt in chain if opt.get("strike")})
        default_strike = min(strikes, key=lambda x: abs(x - spot)) if strikes else spot

        days_to_exp_default = max(
            1,
            (
                datetime.strptime(exp, "%Y-%m-%d").date()
                - datetime.utcnow().date()
            ).days,
        )

        rf_info = get_bond_yield_info("^TNX")
        base_rf = (rf_info.get("spot") or 0) / 100

        iv_market = None
        for opt in chain:
            if float(opt.get("strike", 0)) == float(default_strike) and opt.get("option_type") == "call":
                iv_market = opt.get("greeks", {}).get("iv") or opt.get("greeks", {}).get("mid_iv")
                if iv_market:
                    break
        if iv_market:
            iv_market = float(iv_market)
        else:
            iv_market = 0.2

        with st.form("binomial_tree_form"):
            left, right = st.columns((2, 1))
            with left:
                strike = st.number_input(
                    "Strike",
                    value=float(default_strike),
                    format="%.2f",
                    help="Node payoffs will be evaluated against this strike.",
                )
                opt_side = st.selectbox("Option Type", ["call", "put"], help="Choose which payoff to project.")
                steps = st.slider(
                    "Steps",
                    min_value=2,
                    max_value=50,
                    value=8,
                    help="Increase steps for a smoother tree at the cost of computation.",
                )
            with right:
                days_to_exp = st.number_input(
                    "Days to expiration",
                    min_value=1,
                    value=int(days_to_exp_default),
                    help="Override if you want to stress a custom horizon.",
                )
                rf_input = st.number_input(
                    "Risk-free rate (%)",
                    value=float(base_rf * 100 if base_rf else 4.50),
                    step=0.05,
                    format="%.2f",
                )
                iv_input = st.number_input(
                    "Volatility (IV %)",
                    value=float(iv_market * 100),
                    step=0.5,
                    format="%.2f",
                )

            build_tree = st.form_submit_button("Build tree", use_container_width=True)

        if build_tree:
            T = days_to_exp / 365
            if T <= 0:
                st.error("Days to expiration must be at least 1 to build the tree.")
            else:
                tree_df = generate_binomial_tree(
                    spot,
                    float(strike),
                    T,
                    float(rf_input) / 100,
                    float(iv_input) / 100,
                    int(steps),
                    opt_side,
                )
                fig = plot_binomial_tree(tree_df)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(tree_df, use_container_width=True)
                st.caption(
                    "Use higher step counts for smoother convergence toward Black-Scholes theoretical values."
                )
    else:
        st.info("Enter ticker and expiration to build a tree.")

# --- Tab 5: Market Sentiment ---
with sentiment_tab:
    st.header("🌅 Market Sentiment & Futures")
    st.caption("Cross-asset gauges to frame the macro backdrop and liquidity tone.")

    futs = get_futures_quotes(("ES=F", "NQ=F", "YM=F", "RTY=F", "CL=F", "GC=F"))
    highlight_meta = {
        "ES=F": "S&P 500 futures",
        "NQ=F": "Nasdaq 100 futures",
        "CL=F": "WTI crude",
    }
    if futs:
        top_cols = st.columns(len(highlight_meta))
        for (symbol, note), col in zip(highlight_meta.items(), top_cols):
            data = futs.get(symbol)
            if not data or data.get("last") is None:
                continue
            change = data.get("change_pct")
            delta = f"{change:+.2f}% vs open" if change is not None else None
            label = symbol.split("=")[0]
            with col:
                metric_card(
                    label,
                    f"{data['last']:.2f}",
                    delta=delta,
                    footnote=note,
                )

        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown("**Global futures board**")
        df_fut = pd.DataFrame(
            [
                {
                    "Contract": symbol.replace("=F", ""),
                    "Last": payload.get("last"),
                    "Change %": payload.get("change_pct"),
                }
                for symbol, payload in futs.items()
            ]
        ).sort_values("Contract")
        st.dataframe(df_fut, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    try:
        ten = get_bond_yield_info("^TNX")
    except Exception:
        ten = {"spot": float("nan"), "1d_return": 0.0}
    vix = st.session_state.get("latest_vix")
    if not vix:
        try:
            vix = get_vix_info()
        except Exception:
            vix = {"spot": float("nan"), "1d_return": 0.0, "5d_return": 0.0}
    b2c = get_bid_to_cover(api_key=st.secrets.get("FRED_API_KEY"))

    metric_cols = st.columns(3)
    with metric_cols[0]:
        metric_card(
            "10Y Treasury Yield",
            f"{ten['spot']:.2f}%",
            delta=f"{ten['1d_return']:+.2f}% 1d",
            footnote="Benchmark rate for discounting equity cash flows",
        )
    with metric_cols[1]:
        metric_card(
            "VIX Volatility Index",
            f"{vix['spot']:.2f}",
            delta=f"{vix['1d_return']:+.2f}% 1d",
            footnote=f"5d: {vix['5d_return']:+.2f}%",
        )
    with metric_cols[2]:
        if b2c.get("value") is not None:
            metric_card(
                "10Y Auction Bid-to-Cover",
                f"{b2c['value']:.2f}",
                footnote=">2.5 typically signals strong demand",
            )
        else:
            metric_card("10Y Auction Bid-to-Cover", "N/A", footnote="FRED data unavailable")

# --- Tab 6: Market News ---
with news_tab:
    st.header("📰 Market & Sentiment News")
    st.caption("Condensed macro, volatility and flow stories curated from your watchlists.")
    if st.button("Refresh headlines", key="refresh_headlines"):
        try:
            st.session_state["cached_articles"] = fetch_and_filter_rss(limit_per_feed=30)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error refreshing news: {e}")

    articles = st.session_state.get("cached_articles", [])

    if articles:
        for row in chunked(articles[:12], 3):
            cols = st.columns(len(row))
            for col, art in zip(cols, row):
                if not art:
                    continue
                summary = format_blurb(art.get("summary") or art.get("description") or "")
                title = escape(art.get("title", ""))
                source = escape(art.get("source", ""))
                date = escape(art.get("date", ""))
                link = art.get("link", "")
                col.markdown(
                    f"""
                    <div class="article-card">
                        <h5><a href="{link}" target="_blank">{title}</a></h5>
                        <p>{summary}</p>
                        <div class="metric-footnote">{source} · {date}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("Headlines filtered for options, volatility, and macro catalysts.")
    else:
        st.info("No recent articles matching your filters just yet.")

with calender_tab:
    st.header("📅 Economic Calendar")
    st.caption("Upcoming high-impact releases to complement the news stream.")
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.components.v1.html(
        """
        <iframe src="https://sslecal2.investing.com?columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&importance=2,3&features=datepicker,timezone,filters&countries=5&calType=week&timeZone=8&lang=1"
         style="width:100%;min-height:850px;border:0;" allowtransparency="true"></iframe>
        <div class="poweredBy" style="font-family: Arial, Helvetica, sans-serif;">
          <span style="font-size: 11px;color: #cbd5f5;text-decoration: none;">
            Real Time Economic Calendar provided by
            <a href="https://www.investing.com/" rel="nofollow" target="_blank" style="font-size: 11px;color: #38bdf8; font-weight: bold;" class="underline_link">Investing.com</a>.
          </span>
        </div>
        """,
        height=900,
        scrolling=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 7: AI Analysis ---
if enable_ai and ai_tab:
    with ai_tab:
        st.header("🤖 AI Analysis")
        st.write(
            "Prepare the AI request to review the data packet, estimated token usage, and confirm before sending."
        )
        if "want_ai" not in st.session_state:
            st.session_state.want_ai = False

        openai_creds, selected_model = render_model_selection(ticker, selected_exps)

        if st.button("Prepare AI Analysis"):
            st.session_state.want_ai = True

        if st.session_state.want_ai:
            if st.button("Cancel AI Preparation", key="cancel_ai_preparation"):
                st.session_state.want_ai = False
            else:
                openai_query(
                    df,
                    iv_skew_df,
                    vol_ratio,
                    oi_ratio,
                    articles,
                    spot,
                    offset,
                    ticker,
                    selected_exps,
                    selected_model,
                    openai_creds,
                )

        st.markdown("---")
        st.header("📚 Past AI Analyses")
        hist = load_analyses(limit=10)

        # format however you like, e.g. 24h
        for rec in hist:
            ts_utc = datetime.fromisoformat(rec["ts"]).replace(tzinfo=ZoneInfo("UTC"))
            ts_la  = ts_utc.astimezone(ZoneInfo("America/Los_Angeles"))
            label = ts_la.strftime("%Y-%m-%d %H:%M %Z")
            with st.expander(f"{label} — {rec['ticker']}"):
                st.markdown("**Payload:**")
                # st.json(rec["payload"])
                st.markdown(rec["token_count"])
                st.markdown("**Response:**")
                st.markdown(rec["response"])
