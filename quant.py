import requests
from openai import OpenAI
import tiktoken
from datetime import datetime, timedelta
from helpers import get_market_snapshot, augment_payload_with_extras
import streamlit as st
import pandas as pd
from db import save_analysis
import yaml
import os


# Load configuration from YAML file
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)


# New helper functions for modularized markdown building

def build_header(payload):
    m = []
    m.append(f"**Snapshot Date:** {payload['payload_date']}  ")
    m.append(f"**Timestamp (UTC):** {payload['timestamp']}  ")
    m.append(f"**Spot Price:** {payload['spot']:.2f}  ")
    m.append("")
    return m

def build_returns_section(payload):
    m = []
    returns = payload.get("returns", {})
    if returns:
        m.append("### Returns")
        m.append("| Period | Return (%) |")
        m.append("|---|---:|")
        for period, pct in returns.items():
            m.append(f"| {period} | {pct:.2f} |")
        m.append("")
    return m

def build_technical_section(payload):
    m = []
    m.append("### Technical Indicators")
    tech = payload.get("technical", {})
    for name, val in tech.items():
        m.append(f"- **{name}**: {val:.2f}")
    m.append("")
    return m

def build_vix_section(payload):
    m = []
    m.append("### VIX (CBOE Volatility Index)")
    vix = payload.get("vix", {})
    if vix:
        m.append(f"- **Spot**: {vix['spot']:.2f}")
        m.append(f"- **1-Day Return**: {vix['1d_return']:.2f}%")
        m.append(f"- **5-Day Return**: {vix['5d_return']:.2f}%")
        m.append("")
    return m

def build_iv_skew_section(payload):
    m = []
    iv_skew = payload.get("iv_skew", [])
    if iv_skew:
        m.append("### IV Skew by Strike")
        df_skew = pd.DataFrame(iv_skew)
        m.append(df_skew.to_markdown(index=False))
        m.append("")
    return m

def build_greek_exposures_section(payload):
    m = []
    m.append("### Greek Exposures (±10 strikes around spot)")
    for greek, recs in payload.get("greek_exposures", {}).items():
        m.append(f"#### {greek}")
        dfg = pd.DataFrame(recs)
        m.append(dfg.to_markdown(index=False))
        m.append("")
    return m

def build_vol_spikes_section(payload):
    m = []
    spikes = payload.get("vol_oi_spikes", [])
    if spikes:
        m.append("### Top Volume/OI Spikes by Strike")
        df_sp = pd.DataFrame(spikes)
        m.append(df_sp.to_markdown(index=False))
        m.append("")
    return m

def build_events_section(payload):
    m = []
    events = payload.get("events", [])
    if events:
        m.append("### Upcoming Events")
        for ev in events:
            m.append(f"- **{ev['name']}**: {ev['date']}")
        m.append("")
    return m

def build_extended_snapshot_section(payload, ticker, exp, offset):
    # Augment payload with extras
    payload = augment_payload_with_extras(payload, st.secrets["TRADIER_TOKEN"], ticker, exp, offset, payload['spot'])
    user_md = f"""
    ## Market Snapshot (extended)

    **Spot:** {payload['spot']:.2f}  
    **1-Day Return:** {payload['returns']['1d']:.2f}%  
    **RSI14:** {payload['technical']['RSI14']:.2f}  

    **Risk Reversal (25Δ):** {payload['risk_reversal_25']:.2f}  
    **Butterfly Skew:** {payload['butterfly_skew']:.2f}  
    **Avg Bid-Ask Spread:** {payload['avg_bid_ask_spread']*100:.2f}%  
    """
    return [user_md]

def build_headlines_section(payload):
    m = []
    headlines = payload.get("headlines", [])
    if headlines:
        m.append("### Recent Headlines")
        for art in headlines:
            title = art.get("title", art)
            src   = art.get("source", "")
            date  = art.get("date", "")
            line = f"- {date} {title}"
            if src:
                line += f" _(via {src})_"
            m.append(line)
        m.append("")
    return m

def build_treasury_section(payload):
    md = []
    # … prior sections …

    # Bond Yields
    md.append("### Key Treasury Yields")
    for key, info in payload.get("bond_yields", {}).items():
        md.append(
            f"- **{key.upper()}** ({info['symbol']}): {info['spot']:.2f}%  "
            f"1d: {info['1d_return']:.2f}%, 5d: {info['5d_return']:.2f}%"
        )
    md.append("")
    return md

# Updated modularized payload_to_markdown function
def payload_to_markdown(payload, ticker=None, exp=None, offset=None):
    """
    Convert the market‐snapshot payload into Markdown using modular helper functions.
    """
    md = []
    md.extend(build_header(payload))
    md.extend(build_returns_section(payload))
    md.extend(build_technical_section(payload))
    md.extend(build_greek_exposures_section(payload))
    md.extend(build_vix_section(payload))
    md.extend(build_treasury_section(payload))
    md.extend(build_iv_skew_section(payload))
    md.extend(build_vol_spikes_section(payload))
    md.extend(build_events_section(payload))
    if ticker is not None and exp is not None and offset is not None:
        md.extend(build_extended_snapshot_section(payload, ticker, exp, offset))
    md.extend(build_headlines_section(payload))
    return "\n".join(md)

# New helper function to build the data packet (prompt messages)
MAX_MODEL_TOKENS = 128000

def create_data_packet(ticker, overview_summary, pos_summary, iv_summary, ratios_summary, news_summary, snap_summary):
    system_msg = {
        "role": "system",
        "content": ("You are an experienced volatility trader. "
                    "Analyse the option-dealer positioning charts")
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
            "I can only buy PUTS and CALLS, don't suggest anything else no straddles, stranges, verticles etc, only calls and puts\n"
            "Suggest any concrete 0-DTE, weekly, and/or swing trade options strategy.\n"
            "For each trade provide a trade confidence (1-100), suggested stop-loss level,"
            " and a risk/reward ratio.\n"
            "Figure out the max-pain and also suggest support and resistances."
        )
    }
    return {"messages": [system_msg, user_msg]}

# Helper function to estimate token count
def estimate_token_count(data_packet, max_tokens=MAX_MODEL_TOKENS):
    """Estimate tokens and truncate the user message if above model limit."""
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
        tokens = sum(len(enc.encode(m["content"])) for m in data_packet["messages"])
        st.write(f"Estimated total tokens: **{tokens}**")
        if tokens > max_tokens:
            st.warning(
                f"Token count {tokens} exceeds model limit {max_tokens}. Truncating user message."
            )
            user_msg = data_packet["messages"][-1]
            user_tokens = len(enc.encode(user_msg["content"]))
            allowed = max_tokens - (tokens - user_tokens)
            if allowed > 0:
                truncated = enc.decode(enc.encode(user_msg["content"])[:allowed])
                user_msg["content"] = truncated
                tokens = sum(len(enc.encode(m["content"])) for m in data_packet["messages"])
                st.write(f"Truncated token count: **{tokens}**")
    except Exception as e:
        print(f"Token estimation error: {e}")

    return tokens
    
# Helper function to call OpenAI API and return response analysis
def dummy_response_decorator(func):
    def wrapper(*args, **kwargs):
        # Return a dummy response without querying the API
        return """
        
        Dealer Gamma/Delta Hedging Summary v2

        Around spot (≈592.8) dealers are net long gamma (net positive GEX from roughly 585-600 strikes). That means:
        • On an uptick they will be selling futures/ETF to stay hedged (‐ve delta adjustments).
        • On a downtick they will be buying futures/ETF to stay hedged (+ve delta adjustments).
        → In practice this creates a mean-reversion pinning effect in the 590-595 zone during the day.
        Beyond roughly ±10 points (~<582 or >602), dealers become net short gamma, so rapid moves beyond those levels tend to accelerate (sharp breakouts or breakdowns).

        Max-Pain and Key Option Barriers

        Max-Pain ≈592-593 (highest combined OI on calls and puts, and volume spikes).
        Resistance: 595 (very heavy call gamma/oi), then 600 (clustered call strikes).
        Support: 590 (put gamma/oi concentration), then 585 (next big put block around 580-585).

        Suggested Trades (you can only buy calls or puts)

        A. Intraday (0-DTE)
        Trade: Buy SPY May 21 expiration 590 put (nearest strike)
        Rationale: Overbought RSI, dealers long gamma will buy into any dip toward 590; if you hit 590 quickly you get favorable dealer gamma squeeze.
        Size: small, due to high theta.
        Confidence: 50/100

        B. Short-Term (Weekly expiry May 23)
        Trade: Buy SPY May 23 590 put
        Rationale: Dealers’ positive gamma pegs 592-595, so a drop back into that area will squeeze front-week vols higher. Vega is still relatively low (≈15% on ATM), so buying puts gets vol kicker if realized vol ticks up.
        Confidence: 65/100

        C. Swing (2-4 weeks)
        Trade: Buy SPY June 20 580 put
        Rationale: A broader pullback into the strong put-gamma wall at 585-580 will accelerate if the fence breaks. Gives time for any Fed/rate headlines to drag SPY lower.
        Confidence: 55/100

        D. Optional Bull-Breakout (Weekly expiry May 23)
        Trade: Buy SPY May 23 600 call
        Rationale: If SPY breaks >595 (dealer pin), gamma flips short above ~600, fueling a squeeze. A small runner if you see a sustained break.
        Confidence: 40/100

        Positioning Risk & Execution Notes

        Keep very tight stops on 0DTE (large theta burn).
        For weekly/swing, size for a 1-2% move; your breakeven is vol-driven.
        Watch printed dealer hedges: quick take-profits if you see futures selling into your move (tells you dealers are hedging).

        Support/Resistance Levels Recap

        Near-term support: 590, then 585
        Near-term resistance: 595, then 600

        Final Comment
        Dealers’ long gamma in the 590-595 zone will work against trend extensions within that band, so your best odds come from trades that either fade rallies into 595 or catch a breakdown through 590-585.

        """
    return wrapper

# @dummy_response_decorator
def call_openai_api(data_packet, api_key):
    client = OpenAI(api_key=api_key)
    st.write("model used :", CONFIG.get("openai", {}).get("model", "gpt-4o"))
    try:
        completion = client.chat.completions.create(
            model=CONFIG.get("openai", {}).get("model", "gpt-4o"),
            messages=data_packet["messages"],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

# Refactored openai_query function
def openai_query(df_net, iv_skew_df, vol_ratio, oi_ratio, articles, spot, offset, ticker, exp):
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
            headlines.append(f"- {art['date'][:10]}: {art['title']}")
    except:
        pass
    news_summary = "\n".join(headlines)
    
    snapshot = get_market_snapshot(
        tradier_token=st.secrets["TRADIER_TOKEN"],
        ticker=ticker,
        expirations=exp,
        offset=offset
    )
    snap_summary = payload_to_markdown(snapshot)
    
    data_packet = create_data_packet(ticker, overview_summary, pos_summary, iv_summary, ratios_summary, news_summary, snap_summary)
    st.subheader("Data Packet JSON")
    st.json(data_packet)
    
    tokens = estimate_token_count(data_packet, MAX_MODEL_TOKENS)
    
    analysis = call_openai_api(data_packet, api_key)
    if analysis:
        st.markdown(f"### AI Trade Analysis\n{analysis}")
        save_analysis(
            ticker       = ticker,
            expirations  = exp,  # replaced selected_exps with exp
            payload      = snapshot,
            response     = analysis,  # replaced response variable with analysis
            token_count = tokens
        )
        st.success("✅ Analysis saved.")
    # ...existing error handling...
