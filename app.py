import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
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
    get_bond_yield_info
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
from db import init_db, save_analysis, load_analyses

from db import init_db


def inject_global_styles():
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at 20% 20%, #1e293b 0%, #0f172a 40%, #020617 100%);
                color: #e2e8f0;
            }

            .main .block-container {
                padding-top: 2.5rem;
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
                background: rgba(15, 23, 42, 0.55);
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
                opacity: 0.8;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, *, delta: str | None = None, footnote: str | None = None):
    delta_html = f"<div class='metric-delta'>{delta}</div>" if delta else ""
    foot_html = f"<p class='metric-footnote'>{footnote}</p>" if footnote else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <p>{value}</p>
            {delta_html}
            {foot_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        exp = st.selectbox("Expiration", expirations)
        token = st.secrets.get("TRADIER_TOKEN")
        try:
            chain = get_option_chain(ticker, exp, token, include_all_roots=True)
        except Exception:
            st.error("Failed to fetch options chain")
            chain = []

        strikes = sorted({float(opt.get("strike", 0)) for opt in chain})
        default_strike = min(strikes, key=lambda x: abs(x - spot)) if strikes else spot
        strike = st.number_input("Strike", value=float(default_strike))
        opt_side = st.selectbox("Option Type", ["call", "put"])
        steps = st.slider("Steps", 1, 25, 5)

        rf = get_bond_yield_info("^TNX")["spot"] / 100
        iv = None
        for opt in chain:
            if float(opt.get("strike", 0)) == strike and opt.get("option_type") == opt_side:
                iv = opt.get("greeks", {}).get("iv") or opt.get("greeks", {}).get("mid_iv")
                if iv:
                    iv = float(iv)
                    break
        if iv is None:
            iv = 0.2

        T = (
            datetime.strptime(exp, "%Y-%m-%d").date() - datetime.utcnow().date()
        ).days / 365

        if st.button("Build Tree"):
            tree_df = generate_binomial_tree(spot, strike, T, rf, iv, steps, opt_side)
            fig = plot_binomial_tree(tree_df)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(tree_df)
    else:
        st.info("Enter ticker and expiration to build a tree.")

# --- Tab 3: Market Sentiment ---
with sentiment_tab:
    st.header("🌅 Market Sentiment & Futures")
    futs = get_futures_quotes()
    if futs:
        df_fut = pd.DataFrame([
            {"Symbol": k, "Last": v["last"], "Change %": v["change_pct"]}
            for k, v in futs.items()
        ])
        st.table(df_fut)
    b2c = get_bid_to_cover(api_key=st.secrets.get("FRED_API_KEY"))
    if b2c.get("value") is not None:
        st.metric("10Y Auction Bid-to-Cover", f"{b2c['value']:.2f}")
        st.caption(
            "Strong bid-to-cover typically means solid demand and can help stabilize yields."
        )
    else:
        st.write("Bid-to-cover ratio unavailable.")
    ten = get_bond_yield_info("^TNX")
    st.metric(
        "10Y Treasury Yield",
        f"{ten['spot']:.2f}%",
        delta=f"{ten['1d_return']:.2f}% 1d",
    )
    st.caption(
        "Rising yields often signal expectations of higher inflation or interest rates, while falling yields suggest the opposite."
    )

# --- Tab 5: Market News ---
with news_tab:
    st.header("📰 Market & Sentiment News")
    try:
        articles = fetch_and_filter_rss(limit_per_feed=30)
        if not articles:
            st.write("No recent articles matching your topics.")
        else:
            for art in articles[:100]:               
                st.markdown(
                    f"**[{art['title']}]({art['link']})**  \n"
                    f"<small>{art['source']} — {art['date']}</small>",
                    unsafe_allow_html=True
                )
        fetch_economic_calendar()
    except Exception as e:
        st.error(f"Error fetching news: {e}")

with calender_tab:
    st.header("📅 Economic Calendar")
    st.components.v1.html(
        """
        <iframe src="https://sslecal2.investing.com?columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&importance=2,3&features=datepicker,timezone,filters&countries=5&calType=week&timeZone=8&lang=1"
         width="800" height="1000" frameborder="0" allowtransparency="true" marginwidth="0" marginheight="0"></iframe>
        <div class="poweredBy" style="font-family: Arial, Helvetica, sans-serif;">
          <span style="font-size: 11px;color: #333333;text-decoration: none;">
            Real Time Economic Calendar provided by 
            <a href="https://www.investing.com/" rel="nofollow" target="_blank" style="font-size: 11px;color: #06529D; font-weight: bold;" class="underline_link">Investing.com</a>.
          </span>
        </div>
        """,
        height=520,  # Match iframe + a little for the attribution
        scrolling=True,
    )

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
