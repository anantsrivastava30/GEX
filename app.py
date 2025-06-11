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
    plot_price_and_delta_projection
)
from quant import openai_query
from db import init_db, save_analysis, load_analyses

from db import init_db
init_db()

# ---------------- Streamlit Config ----------------
st.set_page_config(layout="wide", page_title="Options Analytics Dashboard")
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
tab_names = ["Overview Metrics", "Options Positioning", "Market Sentiment", "Market News", "Liquidity"]
if enable_ai:
    tab_names.append("AI Analysis")
tab_names.append("Economic Calendar")
tabs = st.tabs(tab_names)
tab1 = tabs[0]
tab2 = tabs[1]
sentiment_tab = tabs[2]
news_tab = tabs[3]
liq_tab = tabs[4]
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
            pd.DataFrame([{ 
                "Strike": opt["strike"],
                "GEX": opt.get("gamma", 0) * opt.get("open_interest", 0) * opt.get("contract_size", 100)
            } for opt in df0.to_dict('records')])
            .groupby("Strike").sum().reset_index().sort_values("Strike")
        )
        fig_gex = px.bar(
            df_net,
            x="GEX", y="Strike",
            orientation="h",
            title=f"Net Gamma Exposure (Exp {exp0})\n(Showing ±{offset} around {spot:.1f})",
            labels={"GEX": "Net GEX", "Strike": "Strike"},
            template="seaborn",
            height=600
        )
        fig_gex.update_yaxes(tickfont=dict(size=16))
        fig_gex.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_gex, use_container_width=True)
        with st.expander("🔍 More details…", expanded=False):
            st.markdown("\n\n".join(interpret_net_gex(df_net, spot)))

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
            template="seaborn",
            height=600
        )
        fig_skew.update_yaxes(tickfont=dict(size=16))
        fig_skew.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_skew, use_container_width=True)
        st.markdown("""
        **Interpretation:**
        - **Net GEX** shows dealer hedging pressure by strike (positive = dealers bought deltas).
        - **IV Skew > 0** means puts richer than calls: bearish tail premium.
        - **Put/Call Volume & OI Ratios** > 1 indicate more bearish flow.
        - **Unusual Volume/OI Spikes** highlight strikes with outsized trading interest.
        - **When daily volume / open interest** jumps above its 90th percentile, it signals unusual flow—often institutions entering or exiting positions.
        """)
        with st.expander("🔍 More details…", expanded=False):
            st.markdown("""
            - The term "bearish tail risk premium" refers to the additional compensation 
              investors demand for holding assets that are more likely to experience extreme negative returns (i.e., "left tail" events) during bearish market conditions or a downturn.
            - A negative skew means calls are richer → bullish bias or "callers" fear.”
              - **Bullish bias** reflects an optimistic outlook, anticipating rising asset prices and market gains
            """)
        st.markdown("---")

        vol_ratio, oi_ratio = compute_put_call_ratios(df0)
        col1, col2 = st.columns([1, 2])
        fig = plot_put_call_ratios(vol_ratio, oi_ratio)
        with col1:
            st.metric("Put/Call Volume Ratio", f"{vol_ratio:.2f}")
            st.metric("Put/Call OI Ratio", f"{oi_ratio:.2f}")
        with col2:
            st.plotly_chart(fig, use_container_width=True)

        try:
            liq = get_liquidity_metrics(ticker, tradier_token)
            c1, c2, c3 = st.columns(3)
            c1.metric("Trading Volume", f"{liq['volume']:,}")
            if liq.get("bid_ask_spread_pct") is not None:
                delta = None
                hist = liq.get("avg_spread_pct")
                if hist is not None:
                    delta = f"{(liq['bid_ask_spread_pct']/hist-1)*100:+.1f}% vs avg"
                c2.metric(
                    "Bid-Ask Spread (%)",
                    f"{liq['bid_ask_spread_pct']*100:.2f}",
                    delta=delta
                )
            else:
                c2.write("N/A")
            if liq.get("order_book_depth") is not None:
                c3.metric("Order Book Depth", f"{liq['order_book_depth']:,}")
            else:
                c3.write("N/A")
            st.caption(
                "Lower volume, wider spreads and shallow depth typically signal **low liquidity**."
            )
        except Exception as e:
            st.warning(f"Liquidity metrics unavailable: {e}")

        spikes_df = compute_unusual_spikes(df0)
        st.write(spikes_df)
        fig = plot_volume_spikes_stacked(spikes_df, offset=offset, spot=spot)
        st.plotly_chart(fig, use_container_width=True)

        # Price and dealers detla hedge and projection
        # fig = plot_price_and_delta_projection(ticker, exp0, tradier_token, offset=offset)
        # st.plotly_chart(fig, use_container_width=True)
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

# --- Tab 4: Liquidity Metrics ---
with liq_tab:
    st.header("💧 Liquidity Metrics")
    if ticker:
        try:
            liq = get_liquidity_metrics(ticker, st.secrets.get("TRADIER_TOKEN"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Trading Volume", f"{liq['volume']:,}")
            if liq.get("bid_ask_spread_pct") is not None:
                c2.metric("Bid-Ask Spread (%)", f"{liq['bid_ask_spread_pct']*100:.2f}")
            else:
                c2.write("N/A")
            if liq.get("order_book_depth") is not None:
                c3.metric("Order Book Depth", f"{liq['order_book_depth']:,}")
            else:
                c3.write("N/A")
            st.markdown(
                "Lower volume, wider spreads and shallow depth typically signal **low liquidity**."
            )
        except Exception as e:
            st.error(f"Error fetching liquidity metrics: {e}")
    else:
        st.info("Enter a ticker to view liquidity metrics.")

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
        PIN = st.secrets["AI_PIN"]  # e.g. "1234"
        st.header("🤖 AI Analysis")
        st.write("Use the button below to query the OpenAI API for trade insights based on the charts and news.")
        if "want_ai" not in st.session_state:
            st.session_state.want_ai = False
        
        if st.button("Run AI Analysis"):
            st.session_state.want_ai = True
        
        if st.session_state.want_ai:
            user_pin = st.text_input("Enter 4-digit PIN to confirm", type="password")
            if user_pin:
                if user_pin == PIN:
                    st.success("PIN accepted — running AI…")
                    openai_query(df, iv_skew_df, vol_ratio, oi_ratio, articles, spot, offset, ticker, selected_exps)
                    st.session_state.want_ai = False
                else:
                    st.error("❌ Incorrect PIN, try again.")

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
