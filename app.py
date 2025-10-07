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
    plot_exposure,
    plot_price_and_delta_projection,
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


def prepare_strike_metric(df_raw, df_net, view_mode, option_focus="both"):
    df = df_raw.copy()
    if "contract_size" not in df.columns:
        df["contract_size"] = 100
    df["contract_size"] = df["contract_size"].fillna(100)
    if "open_interest" not in df.columns:
        df["open_interest"] = 0
    else:
        df["open_interest"] = df["open_interest"].fillna(0)
    if "volume" not in df.columns:
        df["volume"] = 0
    else:
        df["volume"] = df["volume"].fillna(0)
    if "gamma" not in df.columns:
        df["gamma"] = 0.0
    df["gamma"] = df["gamma"].fillna(0.0)
    df["gamma_exposure"] = (
        df["gamma"] * df["open_interest"] * df["contract_size"]
    )

    view_mode = view_mode.lower()
    option_focus = option_focus.lower()

    if view_mode == "net gamma exposure":
        chart_df = df_net.rename(columns={"GEX": "Value"}).copy()
        chart_df["Metric"] = "Net Gamma"
        label = "Net Gamma Exposure"
        title = "Net Gamma Exposure"
        color_scale = px.colors.diverging.Tealrose
    elif view_mode == "gamma exposure — calls":
        calls = (
            df[df["option_type"].str.lower() == "call"]
            .groupby("strike")["gamma_exposure"].sum()
            .reset_index()
        )
        chart_df = calls.rename(columns={"gamma_exposure": "Value", "strike": "Strike"})
        label = "Gamma Exposure"
        title = "Call Gamma Exposure"
        color_scale = px.colors.sequential.Blues
    elif view_mode == "gamma exposure — puts":
        puts = (
            df[df["option_type"].str.lower() == "put"]
            .groupby("strike")["gamma_exposure"].sum()
            .reset_index()
        )
        chart_df = puts.rename(columns={"gamma_exposure": "Value", "strike": "Strike"})
        label = "Gamma Exposure"
        title = "Put Gamma Exposure"
        color_scale = px.colors.sequential.Oranges
    else:
        metric_col = "open_interest" if "open" in view_mode else "volume"
        if option_focus in {"calls", "puts"}:
            df = df[df["option_type"].str.lower() == option_focus[:-1] if option_focus.endswith("s") else option_focus]
        grouped = df.groupby("strike")[metric_col].sum().reset_index()
        chart_df = grouped.rename(columns={metric_col: "Value", "strike": "Strike"})
        label = metric_col.replace("_", " ").title()
        title = label
        color_scale = px.colors.sequential.Teal

    chart_df = chart_df.sort_values("Strike")
    return chart_df, label, title, color_scale


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
            view_options = [
                "Net Gamma Exposure",
                "Gamma Exposure — Calls",
                "Gamma Exposure — Puts",
                "Open Interest",
                "Volume",
            ]
            chart_view = st.radio(
                "Visualize strike dynamics",
                options=view_options,
                horizontal=True,
            )
            option_focus = "Both"
            if chart_view in {"Open Interest", "Volume"}:
                option_focus = st.radio(
                    "Option focus",
                    options=["Both", "Calls", "Puts"],
                    horizontal=True,
                    help="Toggle to isolate liquidity by option side",
                )
            chart_df, value_label, title_label, color_scale = prepare_strike_metric(
                df0,
                df_net,
                chart_view,
                option_focus,
            )

            if chart_df.empty:
                st.warning("No data available for this view.")
            else:
                fig_gex = px.bar(
                    chart_df,
                    x="Value",
                    y="Strike",
                    orientation="h",
                    color="Value",
                    color_continuous_scale=color_scale,
                    labels={"Value": value_label, "Strike": "Strike"},
                    height=620,
                    title=f"{title_label} (Exp {exp0})\n(±{offset} strikes around {spot:.1f})",
                )
                if chart_view == "Net Gamma Exposure":
                    fig_gex.add_vline(x=0, line_dash="dash", line_color="#f1f5f9")
                fig_gex.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.25)",
                    margin=dict(l=40, r=40, t=90, b=40),
                    coloraxis_showscale=chart_view != "Net Gamma Exposure",
                )
                fig_gex.update_xaxes(title=value_label, tickfont=dict(color="#e2e8f0"))
                fig_gex.update_yaxes(tickfont=dict(size=14, color="#e2e8f0"))
                st.plotly_chart(fig_gex, use_container_width=True)
                st.caption("Flip between GEX, OI and volume to understand how positioning, liquidity and flow align.")

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
            # Filter strikes near spot and select nearest 5 DTE buckets
            df = df[(df.strike >= spot - offset) & (df.strike <= spot + offset)]
            dte_buckets = sorted(df['DTE'].unique())[:5]
            df = df[df['DTE'].isin(dte_buckets)]

            # Plot Gamma Exposure
            fig_gex2 = plot_exposure(df, 'GammaExposure', 'Gamma Exposure', ticker, offset, spot)
            st.plotly_chart(fig_gex2, use_container_width=True)

            # Plot Delta Exposure
            fig_dex2 = plot_exposure(df, 'DeltaExposure', 'Delta Exposure', ticker, offset, spot)
            st.plotly_chart(fig_dex2, use_container_width=True)

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
