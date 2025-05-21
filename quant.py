import requests
import streamlit as st
from openai import OpenAI
import tiktoken
from datetime import datetime, timedelta

from helpers import get_market_snapshot, augment_payload_with_extras

def openai_query(df_net, iv_skew_df, vol_ratio, oi_ratio, articles, spot, offset, ticker, exp):
    st.info("Querying OpenAI… this may take a few seconds.")
    api_key = st.secrets.get("OPENAI_API_KEY")
    # Prepare prompt (customize as needed)
    prompt = (
        f"Analyze SPY option positioning and market news. "
        f"Provide summary and suggest 0-DTE, weekly, and swing strategies."
    )

    if not df_net.empty:
        df_net = df_net[(df_net['strike'] >= spot-offset) & (df_net['strike'] <= spot+offset)]
        # Select only columns displayed in the chart
        pos_df = df_net[['strike','DTE','option_type','GammaExposure', 'DeltaExposure']]
        pos_summary = pos_df.to_markdown(index=False)
    else:
        pos_summary = "No positioning data"
    
    overview_summary = f"Net GEX (±{offset} around {spot:.1f}):"
    iv_summary = iv_skew_df.to_markdown(index=False) if 'iv_skew_df' in locals() else ""
    ratios_summary = f"Put/Call Vol Ratio: {vol_ratio:.2f}, OI Ratio: {oi_ratio:.2f}"
    headlines = []
    # reuse articles list from tab3
    try:
        for art in articles[:15]:
            headlines.append(f"- {art['publishedAt'][:10]}: {art['title']}")
    except:
        pass
    news_summary = "\n".join(headlines)

    snapshot = get_market_snapshot(
        tradier_token=st.secrets["TRADIER_TOKEN"],
        ticker=ticker,
        expirations=exp,
        offset=offset
    )
    st.write(snapshot)
    snap_summary = payload_to_markdown(snapshot)

    # 2. Assemble prompt
    system_msg = {
        "role": "system",
        "content": ("You are an experienced volatility trader. "
                    "Analyse the option-dealer positioning charts ")
    }
                  
    user_msg = {
        "role": "user",
        "content": (
            f"Ticker : {ticker}\n"
            f"Overview Metrics: {overview_summary}\n"
            f"Positioning data gamma and delta exposure : \n{pos_summary}\n"
            f"IV Skew: {iv_summary}\n"
            f"Ratios: {ratios_summary}\n"
            f"News Headlines: {news_summary}\n"
            f"Snapshot Summary: {snap_summary}\n" 
            "Please summarise expected dealer hedging behaviour\n" 
            "I can only BUY puts and calls or straddles (which I prefer not to) (optinos level 2)\n"
            "Suggest any concrete 0-DTE, weekly, and/or swing trade options strategy.\n"
            "and with each strategy recommended give a trade confidence (1-100)\n"
            "Figure out the max-pain and also suggest support and resistances."
        )
    }
    data_packet = {"messages": [system_msg, user_msg]}
    st.subheader("Data Packet JSON")
    st.json(data_packet)

    # 3. Estimate token count
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
        # count tokens in both messages
        tokens = sum(len(enc.encode(m["content"])) for m in data_packet["messages"])
        st.write(f"Estimated total tokens: **{tokens}**")
    except Exception as e:
        st.error(f"Token estimation error: {e}")

    client = OpenAI(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            # model="gpt-4o",
            model="o4-mini",
            messages=data_packet["messages"],
        )

        analysis = completion.choices[0].message.content
        st.markdown(f"### AI Trade Analysis\n{analysis}")
 
    except Exception as e:
        st.error(f"OpenAI API error: {e}")

import pandas as pd

def payload_to_markdown(payload):
    """
    Convert the market‐snapshot payload into a Markdown report.
    """
    ticker = payload["ticker"]
    exp = payload["expirations"]
    offset = payload["offset"]
    md = []
    # Header
    md.append(f"**Snapshot Date:** {payload['payload_date']}  ")
    md.append(f"**Timestamp (UTC):** {payload['timestamp']}  ")
    md.append(f"**Spot Price:** {payload['spot']:.2f}  ")
    md.append("")

    # Returns
    md.append("### Returns")
    returns = payload.get("returns", {})
    if returns:
        md.append("| Period | Return (%) |")
        md.append("|---|---:|")
        for period, pct in returns.items():
            md.append(f"| {period} | {pct:.2f} |")
    md.append("")

    # Technical
    md.append("### Technical Indicators")
    tech = payload.get("technical", {})
    for name, val in tech.items():
        md.append(f"- **{name}**: {val:.2f}")
    md.append("")

    # … VIX Indicators …
    md.append("### VIX (CBOE Volatility Index)")
    vix = payload.get("vix", {})
    if vix:
        md.append(f"- **Spot**: {vix['spot']:.2f}")
        md.append(f"- **1-Day Return**: {vix['1d_return']:.2f}%")
        md.append(f"- **5-Day Return**: {vix['5d_return']:.2f}%")
        md.append("")

    # IV Skew
    iv_skew = payload.get("iv_skew", [])
    if iv_skew:
        df_skew = pd.DataFrame(iv_skew)
        md.append("### IV Skew by Strike")
        md.append(df_skew.to_markdown(index=False))
        md.append("")

    # Greek Exposures
    md.append("### Greek Exposures (±10 strikes around spot)")
    for greek, recs in payload.get("greek_exposures", {}).items():
        md.append(f"#### {greek}")
        dfg = pd.DataFrame(recs)
        md.append(dfg.to_markdown(index=False))
        md.append("")

    # Volume/OI Spikes
    spikes = payload.get("vol_oi_spikes", [])
    if spikes:
        df_sp = pd.DataFrame(spikes)
        md.append("### Top Volume/OI Spikes by Strike")
        md.append(df_sp.to_markdown(index=False))
        md.append("")

    # Events
    events = payload.get("events", [])
    if events:
        md.append("### Upcoming Events")
        for ev in events:
            md.append(f"- **{ev['name']}**: {ev['date']}")
        md.append("")

    payload = augment_payload_with_extras(
        payload, st.secrets["TRADIER_TOKEN"], ticker, exp, offset, payload['spot']
    )
    user_md = f"""
    ## Market Snapshot (extended)

    **Spot:** {payload['spot']:.2f}  
    **1-Day Return:** {payload['returns']['1d']:.2f}%  
    **RSI14:** {payload['technical']['RSI14']:.2f}  

    **Risk Reversal (25Δ):** {payload['risk_reversal_25']:.2f}  
    **Butterfly Skew:** {payload['butterfly_skew']:.2f}  
    **Avg Bid-Ask Spread:** {payload['avg_bid_ask_spread']*100:.2f}%  
    """
    md.append(user_md)

    # Headlines
    headlines = payload.get("headlines", [])
    if headlines:
        md.append("### Recent Headlines")
        for art in headlines:
            # art could be a dict with 'title' and optionally 'source'/'date'
            title = art.get("title", art)
            src   = art.get("source", "")
            date  = art.get("date", "")
            line = f"- {date} {title}"
            if src: line += f" _(via {src})_"
            md.append(line)
        md.append("")

    return "\n".join(md)
