import re
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from zoneinfo import ZoneInfo
from html import escape, unescape
from textwrap import shorten, dedent
from helpers import (
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
)
from utils import (
    plot_put_call_ratios,
    plot_volume_spikes_stacked,
    interpret_net_gex,
    generate_binomial_tree,
    plot_binomial_tree,
)
from quant import openai_query
from db import init_db, load_analyses


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
    gamma_series = work_df.get("gamma", pd.Series(0, index=work_df.index)).fillna(0)
    delta_series = work_df.get("delta", pd.Series(0, index=work_df.index)).fillna(0)

    if "GammaExposure" not in work_df.columns:
        work_df["GammaExposure"] = (
            gamma_series * work_df["open_interest"] * work_df["contract_size"]
        )
    else:
        work_df["GammaExposure"] = work_df["GammaExposure"].fillna(0)

    if "DeltaExposure" not in work_df.columns:
        work_df["DeltaExposure"] = (
            delta_series * work_df["open_interest"] * work_df["contract_size"]
        )
    else:
        work_df["DeltaExposure"] = work_df["DeltaExposure"].fillna(0)

    if "option_type" in work_df.columns:
        put_mask = work_df["option_type"] == "put"
        work_df.loc[put_mask, ["GammaExposure", "DeltaExposure"]] *= -1

    return work_df


def prepare_strike_metric(df_raw, view_mode, option_focus="both"):
    df = _ensure_position_columns(df_raw)
    df["DTE"] = df.get("DTE", 0).fillna(0).astype(int)
    df["expiration_label"] = df.get("expiration_label", "").astype(str)
    df["expiration_display"] = df.apply(
        lambda row: _display_expiration_label(row["expiration_label"], row["DTE"]), axis=1
    )

    metric_key = view_mode.lower()
    focus = _normalise_option_focus(option_focus)

    metric_map = {
        "gamma exposure": ("GammaExposure", "Gamma Exposure", px.colors.sequential.Blues),
        "delta exposure": ("DeltaExposure", "Delta Exposure", px.colors.sequential.Oranges),
        "open interest": ("open_interest", "Open Interest", px.colors.sequential.Teal),
        "volume": ("volume", "Volume", px.colors.sequential.Purples),
    }

    if metric_key not in metric_map:
        raise ValueError(f"Unsupported metric view: {view_mode}")

    value_col, label, color_scale = metric_map[metric_key]

    work_df = df.copy()
    if focus != "both" and "option_type" in work_df.columns:
        work_df = work_df[work_df["option_type"] == focus]
    elif focus == "both" and "option_type" in work_df.columns:
        work_df = work_df[work_df["option_type"].isin(["call", "put"])]

    if value_col not in work_df.columns:
        work_df[value_col] = 0.0

    if value_col in {"GammaExposure", "DeltaExposure"}:
        work_df["abs_value"] = work_df[value_col].abs() / 1e6
        work_df["raw_value"] = work_df[value_col] / 1e6
        value_label = f"{label} (|$M|)"
    else:
        work_df["abs_value"] = work_df[value_col]
        work_df["raw_value"] = work_df[value_col]
        value_label = label

    group_cols = ["strike", "expiration_label", "DTE", "expiration_display"]
    grouped = (
        work_df.groupby(group_cols)
        .agg(Value=("abs_value", "sum"), RawValue=("raw_value", "sum"))
        .reset_index()
    )

    if grouped.empty:
        return grouped.rename(columns={"strike": "Strike"})[["Strike", "Value"]], value_label, label, color_scale

    grouped = (
        grouped.rename(
            columns={
                "strike": "Strike",
                "expiration_display": "Expiration",
            }
        )
        .sort_values(["DTE", "Strike"])
    )
    return grouped[["Strike", "Expiration", "DTE", "expiration_label", "Value", "RawValue"]], value_label, label, color_scale


def prepare_expiration_metric(df_raw, view_mode, option_focus="both"):
    df = _ensure_position_columns(df_raw)
    df["DTE"] = df.get("DTE", 0).fillna(0).astype(int)
    df["expiration_label"] = df.get("expiration_label", "").astype(str)
    df["expiration_display"] = df.apply(
        lambda row: _display_expiration_label(row["expiration_label"], row["DTE"]), axis=1
    )

    metric_key = view_mode.lower()
    focus = _normalise_option_focus(option_focus)

    metric_map = {
        "gamma exposure": ("GammaExposure", "Gamma Exposure", px.colors.sequential.Blues),
        "delta exposure": ("DeltaExposure", "Delta Exposure", px.colors.sequential.Oranges),
        "open interest": ("open_interest", "Open Interest", px.colors.sequential.Teal),
        "volume": ("volume", "Volume", px.colors.sequential.Purples),
    }

    if metric_key not in metric_map:
        raise ValueError(f"Unsupported metric view: {view_mode}")

    value_col, label, color_scale = metric_map[metric_key]

    work_df = df.copy()
    if focus != "both" and "option_type" in work_df.columns:
        work_df = work_df[work_df["option_type"] == focus]
    elif focus == "both" and "option_type" in work_df.columns:
        work_df = work_df[work_df["option_type"].isin(["call", "put"])]

    if value_col not in work_df.columns:
        work_df[value_col] = 0.0

    if value_col in {"GammaExposure", "DeltaExposure"}:
        work_df["abs_value"] = work_df[value_col].abs() / 1e6
        work_df["raw_value"] = work_df[value_col] / 1e6
        value_label = f"{label} (|$M|)"
    else:
        work_df["abs_value"] = work_df[value_col]
        work_df["raw_value"] = work_df[value_col]
        value_label = label

    group_cols = ["expiration_date", "expiration_label", "DTE", "expiration_display"]
    grouped = (
        work_df.groupby(group_cols)
        .agg(Value=("abs_value", "sum"), RawValue=("raw_value", "sum"))
        .reset_index()
    )

    if grouped.empty:
        empty = grouped.rename(columns={"expiration_display": "Expiration"})
        return empty[[col for col in ["Expiration", "Value"] if col in empty.columns]], value_label, label, color_scale

    chart_df = grouped.sort_values("expiration_date").rename(
        columns={"expiration_display": "Expiration"}
    )[["Expiration", "Value", "RawValue", "DTE", "expiration_label"]]

    return chart_df, value_label, label, color_scale


init_db()

articles: list[dict] = []

# ---------------- Streamlit Config ----------------
st.set_page_config(layout="wide", page_title="Options Analytics Dashboard")

inject_global_styles()
st.title("📊 Options Analytics Dashboard")

# --- Sidebar Inputs ---
ticker = st.sidebar.text_input("Ticker", "SPY").upper()
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
selected_exps = st.sidebar.multiselect(
    "Select Expiration Dates",
    options=expirations,
    default=expirations[:3]
)
offset = st.sidebar.slider("Strike Range ±", min_value=1, max_value=300, value=35)
spot = None
if ticker:
    try:
        spot = get_stock_quote(ticker,  st.secrets.get("TRADIER_TOKEN"))
        st.sidebar.markdown(f"**Spot Price:** ***{spot:.2f}***")
    except Exception:
        st.sidebar.error("Error fetching spot price.")

enable_ai = st.sidebar.checkbox("Enable AI Analysis", value=True)

# --- Tabs ---
tab_names = [
    "Overview Metrics",
    "Options Positioning",
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
binom_tab = tabs[2]
sentiment_tab = tabs[3]
news_tab = tabs[4]
if enable_ai:
    ai_tab = tabs[5]
    calender_tab = tabs[6]
else:
    ai_tab = None
    calender_tab = tabs[5]

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
            chart_df, value_label, label, color_scale = prepare_strike_metric(
                df0,
                chart_metric,
                option_focus,
            )

            if chart_df.empty:
                st.warning("No data available for this view.")
            else:
                value_fmt = ".2f" if chart_metric in {"Gamma Exposure", "Delta Exposure"} else ",.0f"
                signed_fmt = "+,.2f" if chart_metric in {"Gamma Exposure", "Delta Exposure"} else "+,.0f"
                custom = chart_df[["Expiration", "DTE", "RawValue"]].values
                fig_gex = px.bar(
                    chart_df,
                    x="Value",
                    y="Strike",
                    orientation="h",
                    color="Expiration",
                    labels={"Value": value_label, "Strike": "Strike"},
                    height=620,
                    color_discrete_sequence=color_scale,
                    title=f"{label} (Exp {exp0})\n(±{offset} strikes around {spot:.1f})",
                )
                fig_gex.update_traces(
                    customdata=custom,
                    hovertemplate=(
                        "<b>Strike %{y}</b><br>"
                        + "Value %{x:" + value_fmt + "}"
                        + "<br>Expiration %{customdata[0]}"
                        + "<br>DTE %{customdata[1]}"
                        + "<br>Signed %{customdata[2]:" + signed_fmt + "}"
                        + "<extra></extra>"
                    ),
                )
                fig_gex.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.25)",
                    margin=dict(l=40, r=40, t=90, b=40),
                    legend_title_text="Expiration · DTE",
                )
                fig_gex.update_xaxes(title=value_label, tickfont=dict(color="#e2e8f0"))
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

        calls = df0[df0.option_type == "call"][['strike', 'mid_iv']].rename(columns={'mid_iv': 'iv_call'})
        puts = df0[df0.option_type == "put"][['strike', 'mid_iv']].rename(columns={'mid_iv': 'iv_put'})
        iv_skew_df = pd.merge(calls, puts, on='strike')
        iv_skew_df['IV Skew'] = iv_skew_df['iv_put'] - iv_skew_df['iv_call']
        fig_skew = px.line(
            iv_skew_df,
            x='strike', y='IV Skew',
            markers=True,
            title=f"IV Skew (Put IV - Call IV)\n(±{offset} around {spot:.1f})",
            labels={'strike': 'Strike', 'IV Skew': 'IV Skew'},
            template="plotly_dark",
            height=520
        )
        fig_skew.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.25)",
            margin=dict(l=40, r=40, t=80, b=40)
        )
        fig_skew.update_yaxes(tickfont=dict(size=14, color="#e2e8f0"))
        fig_skew.update_xaxes(tickfont=dict(color="#e2e8f0"))
        fig_skew.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        st.plotly_chart(fig_skew, use_container_width=True)
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown(
            """
            **IV skew interpretation:**
            - Positive skew means puts are richer than calls, signalling demand for downside insurance or dealer short gamma.
            - Negative skew shows calls pricing above puts, often after squeeze dynamics or call overwriting flows.
            - Watch how the slope shifts with spot — a steepening skew into lower strikes hints at intensifying crash hedging.
            - Overlay with volume/OI spikes to confirm whether skew moves are driven by fresh trades or mark-to-market moves.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

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

        try:
            liq = get_liquidity_metrics(ticker, tradier_token)
            st.markdown("#### Liquidity snapshot")
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                metric_card("Trading Volume", f"{liq['volume']:,}")
            if liq.get("bid_ask_spread_pct") is not None:
                hist = liq.get("avg_spread_pct")
                delta = None
                if hist is not None and hist:
                    delta = f"{(liq['bid_ask_spread_pct']/hist-1)*100:+.1f}% vs avg"
                with lc2:
                    metric_card(
                        "Bid-Ask Spread",
                        f"{liq['bid_ask_spread_pct']*100:.2f}%",
                        delta=delta,
                        footnote="Tighter spreads = easier executions",
                    )
            else:
                with lc2:
                    metric_card("Bid-Ask Spread", "N/A")
            if liq.get("order_book_depth") is not None:
                with lc3:
                    metric_card("Order Book Depth", f"{liq['order_book_depth']:,}")
            else:
                with lc3:
                    metric_card("Order Book Depth", "N/A")
            st.caption("Lower volume, wider spreads and shallow depth typically signal **low liquidity**.")
        except Exception as e:
            st.warning(f"Liquidity metrics unavailable: {e}")

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

                    chart_col, insight_col = st.columns([3, 2], gap="large")
                    with chart_col:
                        st.subheader("Positioning drilldown")
                        lens_mode = st.radio(
                            "Slice positioning by",
                            options=["Strike lens", "Expiration lens"],
                            horizontal=True,
                        )
                        chart_metric = st.radio(
                            "Focus metric",
                            options=["Gamma Exposure", "Delta Exposure", "Open Interest", "Volume"],
                            horizontal=True,
                        )
                        option_focus = st.radio(
                            "Option side",
                            options=["Combined", "Calls", "Puts"],
                            horizontal=True,
                            help="Combined sums call & put magnitudes; switch sides to isolate flows.",
                        )

                        value_fmt = ".2f" if chart_metric in {"Gamma Exposure", "Delta Exposure"} else ",.0f"
                        signed_fmt = "+,.2f" if chart_metric in {"Gamma Exposure", "Delta Exposure"} else "+,.0f"
                        if lens_mode == "Strike lens":
                            chart_df, value_label, label, color_scale = prepare_strike_metric(
                                df,
                                chart_metric,
                                option_focus,
                            )
                            if chart_df.empty:
                                st.warning("No data available for this view.")
                            else:
                                custom = chart_df[["Expiration", "DTE", "RawValue"]].values
                                fig = px.bar(
                                    chart_df,
                                    x="Value",
                                    y="Strike",
                                    orientation="h",
                                    color="Expiration",
                                    labels={"Value": value_label, "Strike": "Strike"},
                                    height=650,
                                    color_discrete_sequence=color_scale,
                                    title=f"{label} across strikes\n(±{offset} around {spot:.1f})",
                                )
                                fig.update_traces(
                                    customdata=custom,
                                    hovertemplate=(
                                        "<b>Strike %{y}</b><br>"
                                        + "Value %{x:" + value_fmt + "}"
                                        + "<br>Expiration %{customdata[0]}"
                                        + "<br>DTE %{customdata[1]}"
                                        + "<br>Signed %{customdata[2]:" + signed_fmt + "}"
                                        + "<extra></extra>"
                                    ),
                                )
                                fig.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(15,23,42,0.25)",
                                    margin=dict(l=40, r=40, t=90, b=40),
                                    legend_title_text="Expiration · DTE",
                                )
                                fig.update_xaxes(title=value_label, tickfont=dict(color="#e2e8f0"))
                                fig.update_yaxes(tickfont=dict(size=14, color="#e2e8f0"))
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            chart_df, value_label, label, color_scale = prepare_expiration_metric(
                                df,
                                chart_metric,
                                option_focus,
                            )
                            if chart_df.empty:
                                st.warning("No data available for this view.")
                            else:
                                custom = chart_df[["DTE", "RawValue"]].values
                                fig = px.bar(
                                    chart_df,
                                    x="Expiration",
                                    y="Value",
                                    color="Expiration",
                                    labels={"Value": value_label, "Expiration": "Expiration"},
                                    height=650,
                                    color_discrete_sequence=color_scale,
                                    title=f"{label} across expirations",
                                )
                                fig.update_traces(
                                    customdata=custom,
                                    hovertemplate=(
                                        "<b>%{x}</b><br>"
                                        + "Value %{y:" + value_fmt + "}"
                                        + "<br>DTE %{customdata[0]}"
                                        + "<br>Signed %{customdata[1]:" + signed_fmt + "}"
                                        + "<extra></extra>"
                                    ),
                                )
                                fig.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(15,23,42,0.25)",
                                    margin=dict(l=40, r=40, t=90, b=40),
                                    showlegend=False,
                                )
                                fig.update_xaxes(tickfont=dict(color="#e2e8f0"))
                                fig.update_yaxes(title=value_label, tickfont=dict(color="#e2e8f0"))
                                st.plotly_chart(fig, use_container_width=True)

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
                            "- **Tip:** Use Combined for magnitude, then drill into Calls/Puts to validate directional bets."
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

# --- Tab 3: Binomial Tree ---
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

# --- Tab 3: Market Sentiment ---
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

# --- Tab 5: Market News ---
with news_tab:
    st.header("📰 Market & Sentiment News")
    st.caption("Condensed macro, volatility and flow stories curated from your watchlists.")
    articles = []
    try:
        articles = fetch_and_filter_rss(limit_per_feed=30)
    except Exception as e:
        st.error(f"Error fetching news: {e}")

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

# --- Tab 6: AI Analysis ---
if enable_ai and ai_tab:
    with ai_tab:
        st.header("🤖 AI Analysis")
        st.write(
            "Prepare the AI request to review the data packet, estimated token usage, and confirm before sending."
        )
        if "want_ai" not in st.session_state:
            st.session_state.want_ai = False

        if st.button("Prepare AI Analysis"):
            st.session_state.want_ai = True

        if st.session_state.want_ai:
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
            )

        if st.session_state.want_ai and st.button("Cancel AI Preparation"):
            st.session_state.want_ai = False
            st.session_state.pop("ai_model_confirmed", None)
            st.session_state.pop("ai_selected_model", None)

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
