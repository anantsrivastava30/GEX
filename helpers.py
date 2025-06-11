import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import yfinance as yf
import feedparser
import streamlit as st
from zoneinfo import ZoneInfo
import yaml
import os

from tradier_api import TradierAPI

# Load configuration from YAML file
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)

API_URL = CONFIG.get("tradier", {}).get("api_url", "https://api.tradier.com/v1")

RSS_FEEDS = CONFIG.get("news", {}).get("rss_feeds", [])

TOPICS = CONFIG.get("news", {}).get("topics", [])

def fetch_and_filter_rss(feeds=RSS_FEEDS, topics=TOPICS, limit_per_feed=10):
    """
    Fetch items from each RSS URL, filter by topics in title/summary,
    and return as a combined list of dicts. Skip if the article is older than 30 days.
    """
    results = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit_per_feed]:
            text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
            if any(topic.lower() in text for topic in topics):
                # Parse published timestamp
                if hasattr(entry, "published_parsed"):
                    dt = datetime(*entry.published_parsed[:6], tzinfo=ZoneInfo("UTC"))
                else:
                    dt = datetime.now(tz=ZoneInfo("UTC"))
                # Skip if the article is older than 30 days
                if datetime.now(tz=ZoneInfo("UTC")) - dt > timedelta(days=30):
                    continue
                # Convert to local
                local_dt = dt.astimezone(ZoneInfo("America/Los_Angeles"))
                results.append({
                    "url":   entry.link,
                    "title": entry.title,
                    "link":  entry.link,
                    "source": feed.feed.get("title", url),
                    "date":  local_dt.strftime("%Y-%m-%d %H:%M")
                })
    # sort by date descending and dedupe by title
    seen = set()
    uniq = []
    for art in sorted(results, key=lambda x: x["date"], reverse=True):
        if art["title"] not in seen:
            seen.add(art["title"])
            uniq.append(art)
    return uniq

def get_expirations(ticker, token, include_all_roots=False):
    api = TradierAPI(token, API_URL)
    return api.expirations(ticker, include_all_roots)

def load_options_data(ticker, expirations, token):
    """Fetch option chains and process data for positioning."""
    all_opts = []
    for exp in expirations:
        try:
            chain = get_option_chain(ticker, exp, token, include_all_roots=True)
            for opt in chain:
                opt['expiration_date'] = exp
            all_opts.extend(chain)
        except Exception:
            continue

    if not all_opts:
        return pd.DataFrame()

    df = pd.DataFrame(all_opts)
    if 'greeks' in df.columns:
        greeks_df = pd.json_normalize(df.pop('greeks'))
        df = pd.concat([df, greeks_df], axis=1)
    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    df['DTE'] = (df['expiration_date'] - datetime.now()).dt.days
    # Compute exposures
    df['GammaExposure'] = df.gamma * df.open_interest * df.contract_size
    df['DeltaExposure'] = df.delta * df.open_interest * df.contract_size
    # Mirror exposures for puts
    df.loc[df.option_type == 'put', ['GammaExposure', 'DeltaExposure']] *= -1
    df['strike'] = df['strike'].astype(float)
    return df


def get_option_chain(ticker, expiration, token, include_all_roots=True):
    api = TradierAPI(token, API_URL)
    return api.option_chain(ticker, expiration, include_all_roots=include_all_roots)


def get_stock_quote(ticker, token):
    api = TradierAPI(token, API_URL)
    data = api.quote(ticker)
    return data.get("last")


def get_liquidity_metrics(ticker, token):
    """Return volume, bid-ask spread pct and order book depth for a stock."""
    api = TradierAPI(token, API_URL)
    q = api.quote(ticker)
    volume = q.get("volume")
    bid = q.get("bid")
    ask = q.get("ask")
    spread_pct = None
    if bid is not None and ask is not None and (bid + ask) != 0:
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid else None

    depth = None
    try:
        book = api.orderbook(ticker)
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        depth = sum(b.get("size", 0) for b in bids[:5]) + sum(a.get("size", 0) for a in asks[:5])
    except Exception:
        pass

    return {
        "volume": volume,
        "bid_ask_spread_pct": spread_pct,
        "order_book_depth": depth,
    }


def compute_net_gex(chain, spot, offset=20):
    rows = []
    for opt in chain:
        K = opt["strike"]
        if not (spot - offset <= K <= spot + offset):
            continue
        gamma = opt.get("greeks", {}).get("gamma")
        oi = opt.get("open_interest", 0)
        size = opt.get("contract_size", 100)
        if gamma is None:
            continue
        multiplier = 1 if opt["option_type"] == "call" else -1
        rows.append({"Strike": K, "Net GEX": gamma * oi * size * multiplier})
    df = pd.DataFrame(rows)
    return df.groupby("Strike").sum().reset_index().sort_values("Strike")


def compute_iv_skew(chain):
    df = pd.DataFrame(chain)
    if 'greeks' in df.columns:
        greeks_df = pd.json_normalize(df.pop('greeks'))
        df = pd.concat([df, greeks_df], axis = 1)

    calls = df[df['option_type']=='call'][['strike','mid_iv']].rename(columns={'mid_iv':'iv_call'})
    puts  = df[df['option_type']=='put'][['strike','mid_iv']].rename(columns={'mid_iv':'iv_put'})
    skew = pd.merge(calls, puts, on='strike')
    skew['iv_skew'] = skew['iv_put'] - skew['iv_call']
    return skew


def compute_put_call_ratios(df):
    # df = pd.DataFrame(chain)
    vol_r = df[df['option_type']=='put']['volume'].sum() / max(df[df['option_type']=='call']['volume'].sum(), 1)
    oi_r  = df[df['option_type']=='put']['open_interest'].sum() / max(df[df['option_type']=='call']['open_interest'].sum(), 1)
    return vol_r, oi_r


def compute_unusual_spikes(df, top_n=10):
    # df = pd.DataFrame(chain)
    # df['vol_oi'] = df['volume'] / df['open_interest'].replace(0, np.nan)
    # thr = df['vol_oi'].quantile(0.9)
    # return df[df['vol_oi'] >= thr]
    agg = (
        df
        .groupby(['strike', 'option_type'], as_index=False)
        .agg({'volume': 'sum', 'open_interest': 'sum'})
    )

    # 2) Compute volume/open_interest ratio
    agg['vol_oi'] = agg['volume'] / agg['open_interest'].replace(0, np.nan)

    # 3) Pivot so we have separate columns for puts and calls
    pivot = (
        agg
        .pivot(index='strike', columns='option_type', values='vol_oi')
        .fillna(0)
        .rename(columns={'put': 'vol_oi_put', 'call': 'vol_oi_call'})
    )

    # 4) Compute the combined total and select top_n strikes
    pivot['total_vol_oi'] = pivot['vol_oi_put'] + pivot['vol_oi_call']
    spikes = (
        pivot
        .sort_values('total_vol_oi', ascending=False)
        .head(top_n)
        .reset_index()
    )
    return spikes


def compute_greek_exposures(ticker, expirations, tradier_token, offset, spot):
    api = TradierAPI(tradier_token, API_URL)

    rows = []
    for exp in expirations:
        options = api.option_chain(
            ticker,
            exp,
            greeks="true",
            include_all_roots=True,
        )

        for opt in options:
            strike = float(opt.get("strike", 0))
            # filter by spot ± offset
            if not (spot - offset <= strike <= spot + offset):
                continue

            oi   = opt.get("open_interest", 0)
            size = opt.get("contract_size", 100)
            greeks = opt.get("greeks", {})

            gamma = greeks.get("gamma", 0.0) or 0.0
            delta = greeks.get("delta", 0.0) or 0.0
            vega  = greeks.get("vega",  0.0) or 0.0

            rows.append({
                "strike":          strike,
                "option_type":     opt.get("option_type", "").upper(),
                "GammaExposure":   gamma * oi * size,
                "DeltaExposure":   delta * oi * size,
                "VegaExposure":    vega  * oi * size
            })

    df = pd.DataFrame(rows)
    return df

def get_bond_yield_info(ticker="^TYX"):
    """
    Fetch the spot yield and 1d/5d returns for a given Treasury‐yield index via yfinance.
    Default '^TYX' is the CBOE 30-year Treasury yield.
    """
    hist = yf.Ticker(ticker).history(period="10d")["Close"]
    spot = float(hist.iloc[-1])
    ret_1d = (spot / hist.iloc[-2] - 1) * 100
    ret_5d = (spot / hist.iloc[-6] - 1) * 100
    return {
        "symbol":      ticker,
        "spot":        spot,
        "1d_return":   ret_1d,
        "5d_return":   ret_5d
    }

def get_vix_info():
    """
    Returns current VIX spot price plus 1-day and 5-day % changes.
    """
    v = yf.Ticker("^VIX")
    hist = v.history(period="10d")["Close"]  # last 6 trading days
    spot = hist.iloc[-1]
    ret_1d = (spot / hist.iloc[-2] - 1) * 100
    ret_5d = (spot / hist.iloc[-6] - 1) * 100
    return {
        "spot": float(spot),
        "1d_return": float(ret_1d),
        "5d_return": float(ret_5d)
    }

def get_market_snapshot(tradier_token, ticker, expirations, offset=20):
    api = TradierAPI(tradier_token, API_URL)

    payload = {}
    # ── Spot & Quote
    q = api.quote(ticker)
    spot = q.get("last")
    payload["spot"] = spot
    payload["ticker"] = ticker
    payload["expirations"] = expirations
    payload["offset"] = offset
    
    # ── Price History & Returns/RSI
    today = datetime.utcnow().date()
    # fetch at least 14 trading days to compute RSI(14)
    start = today - timedelta(days=30)
    hist = api.history(
        ticker,
        interval="daily",
        start=start.isoformat(),
        end=today.isoformat(),
    )
    df_hist = pd.DataFrame(hist)
    df_hist["date"]  = pd.to_datetime(df_hist["date"])
    df_hist = df_hist.sort_values("date").set_index("date")
    closes = df_hist["close"]
    
    # 1d and 5d returns in %
    payload["returns"] = {
        "1d": (closes.iloc[-1] / closes.iloc[-2] - 1) * 100,
        "5d": (closes.iloc[-1] / closes.iloc[-6] - 1) * 100
    }
    # RSI(14)
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    payload["technical"] = {"RSI14": float(rsi.iloc[-1])}

    # ── Volume/OI Spikes by Type 
    chain0    = get_option_chain(ticker, expirations[0], tradier_token, include_all_roots=True)
    df_opts = pd.DataFrame(chain0)  # full chain for first expiry
    spikes  = compute_unusual_spikes(df_opts, top_n=10)
    payload["vol_oi_spikes"] = spikes.to_dict(orient="records")
    
    # # # ── Greek Exposures (all expirations)
    df_greeks = compute_greek_exposures(ticker, expirations, tradier_token, offset=offset, spot=spot)
    payload["greek_exposures"] = {
        g: (
            df_greeks
            .pivot_table(
                index="strike",
                columns="option_type",
                values=g,
                aggfunc="sum"         # collapse duplicates by summing
            )
            .fillna(0)
            .reset_index()
            .to_dict(orient="records")
        )
        for g in ("GammaExposure", "DeltaExposure", "VegaExposure")
    }

    # -- VIX term structure -----------------
    payload["vix"] = get_vix_info()

    # -- Bond yield term structure ----------
    payload["bond_yields"] = {
        "30y": get_bond_yield_info("^TYX"),
        "10y": get_bond_yield_info("^TNX"),
        "5y":  get_bond_yield_info("^FVX"),
    }
    
    # ── Timestamp & Payload Date ────────────────────────────────
    ts = datetime.utcnow()
    payload["timestamp"] = ts.isoformat()
    payload["payload_date"] = ts.strftime("%Y-%m-%d")
    
    return payload

# ─── Helper Functions ──────────────────────────────────────────────────────

def compute_risk_reversal(chain):
    """
    Compute 25Δ risk reversal = IV_put(25Δ) - IV_call(25Δ)
    Uses mid_iv and greeks['delta'] to find closest deltas.
    """
    df = chain
    greeks_df = pd.json_normalize(df.pop('greeks'))
    df['delta']  = df['delta'].astype(float)
    df['mid_iv'] = df['mid_iv'].astype(float)
    
    calls = df[df['option_type']=='call']
    puts  = df[df['option_type']=='put']
    # find closest to +0.25 and -0.25
    call_25 = calls.iloc[(calls['delta'] - 0.25).abs().argsort()[:1]]
    put_25  = puts.iloc[(puts['delta'] + 0.25).abs().argsort()[:1]]
    
    iv_call_25 = float(call_25['mid_iv'])
    iv_put_25  = float(put_25['mid_iv'])
    return iv_put_25 - iv_call_25


def compute_butterfly_skew(chain, spot):
    """
    Compute simple butterfly = (OTM_put_iv + OTM_call_iv)/2 - ATM_iv.
    OTM strikes = nearest wings ± distance from ATM strike.
    """
    df = chain
    df['strike']  = df['strike'].astype(float)
    df['mid_iv']  = df['mid_iv'].astype(float)
    
    # ATM strike = nearest to spot
    atm_strike = df.iloc[(df['strike'] - spot).abs().argsort()[:1]]['strike'].iloc[0]
    # pick wings two strikes away
    strikes = sorted(df['strike'].unique())
    idx = strikes.index(atm_strike)
    wing_offset = 2  # two steps away
    low_wing  = strikes[max(0, idx-wing_offset)]
    high_wing = strikes[min(len(strikes)-1, idx+wing_offset)]
    
    iv_atm  = df[(df['strike']==atm_strike) & (df['option_type']=='call')]['mid_iv'].mean()
    iv_low  = df[(df['strike']==low_wing)  & (df['option_type']=='put')]['mid_iv'].mean()
    iv_high = df[(df['strike']==high_wing) & (df['option_type']=='call')]['mid_iv'].mean()
    
    return ((iv_low + iv_high)/2) - iv_atm


def compute_term_structure_slope(tradier_token, ticker, expirations, spot, offset):
    """
    Compute term structure slope = IV_near - IV_far for your ±offset strikes.
    Uses the first and last in expirations list.
    """
    api = TradierAPI(tradier_token, API_URL)
    ivs = []
    for exp in [expirations[0], expirations[-1]]:
        resp = api.option_chain(
            ticker,
            exp,
            greeks="false",
            include_all_roots=True,
        )
        df = pd.DataFrame(resp)
        greeks_df = pd.json_normalize(df.greeks)
        df = pd.concat([df, greeks_df], axis=1)
        df['strike']  = df['strike'].astype(float)
        df['mid_iv']  = df['mid_iv'].astype(float)
        df = df[(df['strike']>=spot-offset)&(df['strike']<=spot+offset)]
        ivs.append(df['mid_iv'].mean())
    return ivs[0] - ivs[1]


def compute_avg_spread(chain, spot, offset):
    """
    Compute average % bid-ask spread for strikes within ±offset of spot.
    """
    df = chain
    df = df[(df['strike']>=spot-offset)&(df['strike']<=spot+offset)]
    # midprice and spread pct
    df['mid']   = (df['bid'] + df['ask'])/2
    df['spread_pct'] = (df['ask'] - df['bid']) / df['mid']
    return float(df['spread_pct'].mean())

# ─── Integration into Payload ──────────────────────────────────────────────

def augment_payload_with_extras(payload, tradier_token, ticker, expirations, offset, spot):
    """
    Adds risk reversal, butterfly skew, term structure slope, and avg spread
    to an existing payload dict.
    """
    # use first expiration's chain
    api = TradierAPI(tradier_token, API_URL)
    all_opts = []
    for exp in expirations:
        chain0 = api.option_chain(
            ticker,
            exp,
            greeks="true",
            include_all_roots=True,
        )
        all_opts.extend(chain0)
    
    df = pd.DataFrame(all_opts)
    greeks_df = pd.json_normalize(df.greeks)
    df = pd.concat([df, greeks_df], axis=1)
    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    df['DTE'] = (df['expiration_date'] - datetime.now()).dt.days

    payload["risk_reversal_25"]    = compute_risk_reversal(df)
    payload["butterfly_skew"]      = compute_butterfly_skew(df, spot)
    # payload["term_structure_slope"]= compute_term_structure_slope(
    #                                     tradier_token, ticker, expirations, spot, offset
                                    #  )
    payload["avg_bid_ask_spread"]  = compute_avg_spread(df, spot, offset)

    return payload
