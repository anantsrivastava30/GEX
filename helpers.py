import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import yfinance as yf
import feedparser
from zoneinfo import ZoneInfo


API_URL = "https://api.tradier.com/v1"

RSS_FEEDS = [
    # Global wire & major publications
    "https://feeds.reuters.com/Reuters/BusinessNews",
    "https://www.ft.com/?format=rss",
    "https://seekingalpha.com/market-news.rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",

    # U.S. business & markets
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://www.marketwatch.com/rss/topstories",
    "https://www.bloomberg.com/feed/podcast/etf-report.xml",
    "https://feeds.bizjournals.com/bizj_national.xml",

    # Tech & innovation (often drives big moves)
    "https://feeds.feedburner.com/TechCrunch/",
    "https://www.theinformation.com/rss/articles",

    # Sector-specific deep dives
    "https://www.spglobal.com/marketintelligence/feed-news",
    "https://www.forbes.com/business/feed2/",

    # Alternative data & sentiment
    "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=LatestNews",
    "https://www.barrons.com/xml/rss/3_7031.xml"
]


TOPICS = [
    "finance","market","stock","fed","inflation",
    "cpi","economy","bonds","yield",
    "rates","trump","tariff","gdp"
]

def parse_av_timestamp(ts_str: str) -> datetime:
    """
    Parse AlphaVantage timestamp which may be ISO8601 or compact (YYYYMMDDTHHMMSS).
    Always returns an aware UTC datetime.
    """
    try:
        # e.g. "2025-05-23T08:26:21Z" or "2025-05-23T08:26:21+00:00"
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        # fallback for "YYYYMMDDTHHMMSS"
        dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return dt

def fetch_and_filter_rss(feeds=RSS_FEEDS, topics=TOPICS, limit_per_feed=10):
    """
    Fetch items from each RSS URL, filter by topics in title/summary,
    and return as a combined list of dicts.
    """
    results = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit_per_feed]:
            text = (entry.get("title","") + " " + entry.get("summary","")).lower()
            if any(topic.lower() in text for topic in topics):
                # Parse published timestamp
                if hasattr(entry, "published_parsed"):
                    dt = datetime(*entry.published_parsed[:6], tzinfo=ZoneInfo("UTC"))
                else:
                    dt = datetime.now(tz=ZoneInfo("UTC"))
                # Convert to local
                local_dt = dt.astimezone(ZoneInfo("America/Los_Angeles"))
                results.append({
                    "title": entry.title,
                    "link":  entry.link,
                    "source": feed.feed.get("title", url),
                    "date":  local_dt.strftime("%Y-%m-%d %H:%M")
                })
    # sort by date descending and dedupe by title
    seen = set(); uniq = []
    for art in sorted(results, key=lambda x: x["date"], reverse=True):
        if art["title"] not in seen:
            seen.add(art["title"])
            uniq.append(art)
    return uniq


def get_expirations(ticker, token, include_all_roots=False):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {"symbol": ticker}
    if include_all_roots:
        params["includeAllRoots"] = "true"
    r = requests.get(f"{API_URL}/markets/options/expirations", params=params, headers=headers)
    r.raise_for_status()
    return r.json().get("expirations", {}).get("date", [])

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
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {"symbol": ticker, "expiration": expiration, "greeks": "true"}
    if include_all_roots:
        params["includeAllRoots"] = "true"
    r = requests.get(f"{API_URL}/markets/options/chains", params=params, headers=headers)
    r.raise_for_status()
    return r.json().get("options", {}).get("option", [])


def get_stock_quote(ticker, token):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    r = requests.get(f"{API_URL}/markets/quotes", params={"symbols": ticker}, headers=headers)
    r.raise_for_status()
    data = r.json().get("quotes", {}).get("quote")
    if isinstance(data, list):
        data = data[0]
    return data.get("last")


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
    headers = {
        "Authorization": f"Bearer {tradier_token}",
        "Accept":        "application/json"
    }

    rows = []
    for exp in expirations:
        response = requests.get(
            f"{API_URL}/markets/options/chains",
            params={
                "symbol": ticker,
                "expiration": exp,
                "greeks": "true",
                "includeAllRoots": "true"
            },
            headers=headers
        )
        response.raise_for_status()
        options = response.json().get("options", {}).get("option", [])

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
    hist = yf.Ticker(ticker).history(period="6d")["Close"]
    spot = float(hist.iloc[-1])
    ret_1d = (spot / hist.iloc[-2] - 1) * 100
    ret_5d = (spot / hist.iloc[-6] - 1) * 100
    return {
        "symbol":      ticker,
        "spot":        spot,
        "1d_return":   ret_1d,
        "5d_return":   ret_5d
    }

def get_market_snapshot(tradier_token, ticker, expirations, offset=20):
    headers = {
        "Authorization": f"Bearer {tradier_token}",
        "Accept":        "application/json"
    }
    
    payload = {}
    # ── Spot & Quote ─────────────────────────────────────────────
    q = requests.get(
        f"{API_URL}/markets/quotes",
        params={"symbols": ticker},
        headers=headers
    ).json()["quotes"]["quote"]
    spot = q["last"]
    payload["spot"] = spot
    payload["ticker"] = ticker
    payload["expirations"] = expirations
    payload["offset"] = offset
    
    # ── Price History & Returns/RSI ─────────────────────────────
    today = datetime.utcnow().date()
    # fetch at least 14 trading days to compute RSI(14)
    start = today - timedelta(days=30)
    hist = requests.get(
        f"{API_URL}/markets/history",
        params={
            "symbol":   ticker,
            "interval": "daily",
            "start":    start.isoformat(),
            "end":      today.isoformat()
        },
        headers=headers
    ).json().get("history", {}).get("day", [])
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

    # ── Volume/OI Spikes by Type ────────────────────────────────
    chain0    = get_option_chain(ticker, expirations[0], tradier_token, include_all_roots=True)
    df_opts = pd.DataFrame(chain0)  # full chain for first expiry
    spikes  = compute_unusual_spikes(df_opts, top_n=10)
    payload["vol_oi_spikes"] = spikes.to_dict(orient="records")
    
    # # # ── Greek Exposures (all expirations) ───────────────────────
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
    API_URL = "https://api.tradier.com/v1"
    headers = {"Authorization": f"Bearer {tradier_token}", "Accept":"application/json"}
    ivs = []
    for exp in [expirations[0], expirations[-1]]:
        resp = requests.get(
            f"{API_URL}/markets/options/chains",
            params={"symbol": ticker, "expiration": exp, "greeks": "false", "includeAllRoots":"true"},
            headers=headers
        ).json().get("options", {}).get("option", [])
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
    API_URL = "https://api.tradier.com/v1"
    headers = {"Authorization": f"Bearer {tradier_token}", "Accept":"application/json"}
    all_opts = []
    for exp in expirations:
        resp = requests.get(
            f"{API_URL}/markets/options/chains",
            params={"symbol": ticker, "expiration": exp, "greeks":"true", "includeAllRoots":"true"},
            headers=headers
        )
        chain0 = resp.json().get("options", {}).get("option", [])
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
