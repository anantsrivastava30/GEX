import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
from datetime import datetime
from helpers import (
    get_expirations,
    get_option_chain,
    get_stock_quote,
    compute_put_call_ratios,
    compute_unusual_spikes
)
from utils import (
    plot_put_call_ratios,
    plot_volume_spikes,
    plot_volume_spikes_stacked,
    interpret_net_gex,
)
from quant import (
    openai_query
)

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
    except:
        st.sidebar.error("Error fetching expirations.")
selected_exps = st.sidebar.multiselect(
    "Select Expiration Dates",
    options=expirations,
    default=expirations[:3]
)
# Strike filter range around spot
offset = st.sidebar.slider("Strike Range ±", min_value=1, max_value=300, value=35)
# Fetch spot price
spot = None
if ticker:
    try:
        spot = get_stock_quote(ticker, st.secrets.get("TRADIER_TOKEN"))
        st.sidebar.markdown(f"**Spot Price:** ***{spot:.2f}***")
    except:
        st.sidebar.error("Error fetching spot price.")

# Enable AI Analysis tab
enable_ai = st.sidebar.checkbox("Enable AI Analysis", "checked")


# --- Tabs ---
tab_names = ["Overview Metrics", "Options Positioning", "Market News"]
if enable_ai:
    tab_names.append("AI Analysis")
# Unpack tabs dynamically
tabs = st.tabs(tab_names)
# Map by index
tab1 = tabs[0]
tab2 = tabs[1]
tab3 = tabs[2]
ai_tab = tabs[3] if enable_ai else None

# --- Tab 1: Overview Metrics ---
with tab1:
    st.header("📈 Overview Metrics")
    if ticker and selected_exps and spot is not None:
        exp0 = selected_exps[0]
        # Fetch chain
        try:
            chain0 = get_option_chain(
                ticker, exp0, st.secrets.get("TRADIER_TOKEN"), include_all_roots=True
            )
        except:
            st.error(f"Failed to fetch options for {exp0}")
            st.stop()
        # Build DataFrame and flatten greeks
        df0 = pd.DataFrame(chain0)
        greeks_df0 = pd.json_normalize(df0.pop("greeks"))
        df0 = pd.concat([df0, greeks_df0], axis=1)
        # Filter strikes near spot
        df0 = df0[(df0.strike >= spot - offset) & (df0.strike <= spot + offset)]

        # Compute Net Gamma Exposure by strike
        df_net = (
            pd.DataFrame([{
                "Strike": opt["strike"],
                "GEX": opt.get("gamma", 0) * opt.get("open_interest", 0) * opt.get("contract_size",100) # * (1 if opt.get("option_type")=="call" else -1)
            } for opt in df0.to_dict('records')])
            .groupby("Strike").sum().reset_index().sort_values("Strike")
        )
        # Plot Net GEX
        fig_gex = px.bar(
            df_net,
            x="GEX", y="Strike",
            orientation="h",
            title=f"Net Gamma Exposure (Exp {exp0})\n(Showing ±{offset} around {spot:.1f})",
            labels={"GEX":"Net GEX","Strike":"Strike"},
            template="seaborn",
            height=600
        )
        fig_gex.update_yaxes(tickfont=dict(size=16))
        fig_gex.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_gex, use_container_width=True)
        with st.expander("🔍 More details…", expanded=False):
            st.markdown("\n\n".join(interpret_net_gex(df_net, spot)))
        
        # Compute IV Skew
        calls = df0[df0.option_type=="call"][['strike','mid_iv']].rename(columns={'mid_iv':'iv_call'})
        puts = df0[df0.option_type=="put"][['strike','mid_iv']].rename(columns={'mid_iv':'iv_put'})
        iv_skew_df = pd.merge(calls, puts, on='strike')
        iv_skew_df['IV Skew'] = iv_skew_df['iv_put'] - iv_skew_df['iv_call']
        
        # Plot IV Skew
        fig_skew = px.line(
            iv_skew_df,
            x='strike', y='IV Skew',
            markers=True,
            title=f"IV Skew (Put IV - Call IV)\n(±{offset} around {spot:.1f})",
            labels={'strike':'Strike','IV Skew':'IV Skew'},
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
            - A negative skew means calls are richer → bullish bias or “callers’ fear.”
              - **Bullish bias** reflects an optimistic outlook, anticipating rising asset prices and market gains
            """)
        st.markdown("---")

        # Put/Call Ratios
        vol_ratio, oi_ratio = compute_put_call_ratios(df0)
        col1, col2 = st.columns([1, 2])
        fig = plot_put_call_ratios(vol_ratio, oi_ratio)
        with col1:      
            st.metric("Put/Call Volume Ratio", f"{vol_ratio:.2f}")
            st.metric("Put/Call OI Ratio", f"{oi_ratio:.2f}")
        with col2:
            st.plotly_chart(fig, use_container_width=True)

        # Unusual Volume/OI Spikes (styled like before)
        spikes_df = compute_unusual_spikes(df0)
        st.write(spikes_df)
        fig = plot_volume_spikes_stacked(spikes_df, offset=35, spot=spot)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select ticker, expirations, and ensure spot price loaded.")
        st.info("Select ticker, expirations, and ensure spot price loaded.")

with tab2:
    st.header("🎯 Options Positioning")                                                           
    if ticker and selected_exps and spot is not None:
        all_opts = []
        for exp in selected_exps:
            try:
                chain = get_option_chain(ticker, exp, st.secrets.get("TRADIER_TOKEN"), include_all_roots=True)
                for opt in chain:
                    opt['expiration_date'] = exp
                all_opts.extend(chain)
            except:
                continue
        df = pd.DataFrame(all_opts)
        greeks_df = pd.json_normalize(df.pop('greeks'))
        df = pd.concat([df, greeks_df], axis=1)
        df['expiration_date'] = pd.to_datetime(df['expiration_date'])
        df['DTE'] = (df['expiration_date'] - datetime.now()).dt.days
        # Compute exposures
        df['GammaExposure'] = df.gamma * df.open_interest * df.contract_size
        df['DeltaExposure'] = df.delta * df.open_interest * df.contract_size
        # Mirror puts
        df.loc[df.option_type=='put', ['GammaExposure','DeltaExposure']] *= -1
        df['strike'] = df['strike'].astype(float)
        # Filter strikes near spot
        df = df[(df.strike >= spot - offset) & (df.strike <= spot + offset)]
        # Select nearest 5 DTE buckets
        dtes = sorted(df['DTE'].unique())[:5]
        df = df[df['DTE'].isin(dtes)]
        # Plot mirror-bar for Gamma
        fig_gex2 = px.bar(
            df,
            x='GammaExposure', y='strike', color='DTE', orientation='h',
            facet_col='option_type',
            category_orders={'option_type':['put','call']},
            labels={'GammaExposure':'GEX','strike':'Strike','DTE':'Days to Exp'},
            title=f"{ticker} Gamma Exposure by Strike & DTE (±{offset} around {spot:.1f})",
            template='presentation'
        )
        fig_gex2.update_layout(barmode='relative', height=800)
        fig_gex2.update_yaxes(tickfont=dict(size=16))
        fig_gex2.update_xaxes(tickfont=dict(size=16))

        fig_gex2.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, griddash='dot')
        fig_gex2.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, griddash='dot', autorange='reversed')
        fig_gex2.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        fig_gex2.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_gex2, use_container_width=True)
        # Plot mirror-bar for Delta
        fig_dex2 = px.bar(
            df,
            x='DeltaExposure', y='strike', color='DTE', orientation='h',
            facet_col='option_type',
            category_orders={'option_type':['put','call']},
            labels={'DeltaExposure':'DEX','strike':'Strike','DTE':'Days to Exp'},
            title=f"{ticker} Delta Exposure by Strike & DTE (±{offset} around {spot:.1f})",
            template='presentation'
        )
        fig_dex2.update_yaxes(tickfont=dict(size=16))
        fig_dex2.update_xaxes(tickfont=dict(size=16))
        fig_dex2.update_layout(barmode='relative', height=800)
        fig_dex2.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, griddash='dot')
        fig_dex2.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, griddash='dot', autorange='reversed')
        fig_dex2.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        fig_dex2.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_dex2, use_container_width=True)
    else:
        st.info("Select ticker and expirations to view positioning.")

with tab3:
    st.header("📰 Market & Sentiment News")
    try:
        NEWS_API_KEY = st.secrets.get("NEWSAPI_KEY")
        params = {
            "q": "finance OR market OR stock OR Fed OR inflation OR CPI OR "
                  "economy OR wall street OR rates OR trump OR tarrifs OR gdb or GDP",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": NEWS_API_KEY
        }
        news_url = "https://newsapi.org/v2/everything"
        resp = requests.get(news_url, params=params)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

        # Display top 10 relevant finance/market headlines
        if articles:
            for art in articles[:30]:
                title = art.get("title", "No title")
                src   = art.get("source", {}).get("name", "Unknown")
                url   = art.get("url", "#")
                published = art.get("publishedAt", "")[:10]
                st.markdown(
                    f"<p style='font-size:18px; line-height:1.5;'>"
                    f"<strong>{published}</strong>: "
                    f"<a href='{url}' target='_blank'>{title}</a> "
                    f"<em style='color:gray;'>(via {src})</em>"
                    f"</p>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No recent market or sentiment news found.")
    except Exception as e:
        st.error(f"Error fetching news: {e}")

# --- Tab 4: AI Analysis ---
if enable_ai and ai_tab:
    with ai_tab:
        st.header("🤖 AI Analysis")
        st.write("Use the button below to query the OpenAI API for trade insights based on the charts and news.")
        run_query = st.button("Run AI Analysis")
        if run_query:
            openai_query(df, iv_skew_df, vol_ratio, oi_ratio, articles, spot, offset, ticker, selected_exps)