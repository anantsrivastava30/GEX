import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
import math
import yfinance as yf
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
from tradier_api import TradierAPI


def build_gamma_profile(chain, spot: float | None, offset: float = 25.0) -> pd.DataFrame:
    """Return call, put and net gamma exposure grouped by strike."""

    if not chain or spot is None:
        return pd.DataFrame(columns=["Strike", "CallGamma", "PutGamma", "NetGamma"])

    df = pd.DataFrame(chain)
    if df.empty:
        return pd.DataFrame(columns=["Strike", "CallGamma", "PutGamma", "NetGamma"])

    if "greeks" in df.columns:
        greeks_df = pd.json_normalize(df.pop("greeks"))
        df = pd.concat([df, greeks_df], axis=1)

    try:
        strikes = df["strike"].astype(float)
    except Exception:
        strikes = pd.to_numeric(df.get("strike"), errors="coerce")
    df["Strike"] = strikes
    df = df.dropna(subset=["Strike"])
    if df.empty:
        return pd.DataFrame(columns=["Strike", "CallGamma", "PutGamma", "NetGamma"])

    df = df[(df["Strike"] >= spot - offset) & (df["Strike"] <= spot + offset)]
    if df.empty:
        return pd.DataFrame(columns=["Strike", "CallGamma", "PutGamma", "NetGamma"])

    contract_size = df.get("contract_size")
    if contract_size is None:
        contract_size = pd.Series(100, index=df.index)
    elif not isinstance(contract_size, pd.Series):
        contract_size = pd.Series(contract_size, index=df.index)
    contract_size = contract_size.fillna(100).replace(0, 100)

    open_interest = df.get("open_interest")
    if open_interest is None:
        open_interest = pd.Series(0, index=df.index)
    elif not isinstance(open_interest, pd.Series):
        open_interest = pd.Series(open_interest, index=df.index)
    open_interest = open_interest.fillna(0)

    gamma_series = df.get("gamma")
    if gamma_series is None:
        gamma_series = pd.Series(0.0, index=df.index)
    else:
        gamma_series = pd.to_numeric(gamma_series, errors="coerce").fillna(0.0)

    gamma_exposure = gamma_series * open_interest * contract_size

    if "option_type" in df.columns:
        option_type = df["option_type"].astype(str).str.lower()
    else:
        option_type = pd.Series("call", index=df.index)

    call_gamma = (
        gamma_exposure.where(option_type == "call", other=0.0)
        .groupby(df["Strike"])
        .sum()
    )
    put_gamma = (
        gamma_exposure.where(option_type == "put", other=0.0)
        .groupby(df["Strike"])
        .sum()
    )

    profile = (
        pd.concat(
            {
                "CallGamma": call_gamma,
                "PutGamma": put_gamma,
            },
            axis=1,
        )
        .fillna(0.0)
        .reset_index()
    )
    profile["NetGamma"] = profile["CallGamma"] - profile["PutGamma"]
    profile = profile.sort_values("Strike").reset_index(drop=True)
    return profile


def _interpolate_boundary(df: pd.DataFrame, start_idx: int, direction: int) -> float:
    """Return strike where gamma changes sign starting from ``start_idx``."""

    if df.empty:
        return float("nan")

    idx = start_idx
    curr_val = float(df.loc[idx, "NetGamma"])
    curr_strike = float(df.loc[idx, "Strike"])

    if curr_val == 0:
        return curr_strike

    sign = math.copysign(1.0, curr_val)
    while 0 <= idx + direction < len(df):
        next_idx = idx + direction
        next_val = float(df.loc[next_idx, "NetGamma"])
        next_strike = float(df.loc[next_idx, "Strike"])

        if next_val == 0 or math.copysign(1.0, next_val) == sign:
            idx = next_idx
            curr_val = next_val if next_val != 0 else curr_val
            curr_strike = next_strike if next_val != 0 else curr_strike
            continue

        span = abs(curr_val) + abs(next_val)
        if span == 0:
            frac = 0.5
        else:
            frac = abs(curr_val) / span

        delta_strike = abs(next_strike - curr_strike)
        if direction < 0:
            boundary = curr_strike - frac * delta_strike
        else:
            boundary = curr_strike + frac * delta_strike
        return boundary

    return curr_strike


def summarize_gamma_gap(profile: pd.DataFrame, spot: float | None) -> dict:
    """Summarise magnet strike and gap-fill potential from a gamma profile."""

    if spot is None or profile.empty:
        return {
            "has_data": False,
            "message": "No gamma data available",
        }

    df = profile.copy()
    df = df.sort_values("Strike").reset_index(drop=True)
    df["NetGamma"] = pd.to_numeric(df["NetGamma"], errors="coerce").fillna(0.0)

    total_abs_gamma = float(df["NetGamma"].abs().sum())
    net_gamma = float(df["NetGamma"].sum())

    if math.isclose(total_abs_gamma, 0.0):
        return {
            "has_data": False,
            "message": "Gamma exposures are all zero",
        }

    idx_max = int(df["NetGamma"].idxmax())
    idx_min = int(df["NetGamma"].idxmin())
    peak_positive = float(df.loc[idx_max, "NetGamma"])
    peak_negative = float(df.loc[idx_min, "NetGamma"])

    if peak_positive <= 0 and abs(peak_negative) >= abs(peak_positive):
        magnet_idx = idx_min
    else:
        magnet_idx = idx_max

    magnet_row = df.loc[magnet_idx]
    magnet_gamma = float(magnet_row["NetGamma"])
    magnet_strike = float(magnet_row["Strike"])

    magnet_sign = 0.0 if magnet_gamma == 0 else math.copysign(1.0, magnet_gamma)
    if magnet_sign != 0.0:
        lower_bound = _interpolate_boundary(df, magnet_idx, -1)
        upper_bound = _interpolate_boundary(df, magnet_idx, 1)
    else:
        lower_bound = magnet_strike
        upper_bound = magnet_strike

    zero_crossings: list[float] = []
    for i in range(len(df) - 1):
        curr_val = float(df.loc[i, "NetGamma"])
        next_val = float(df.loc[i + 1, "NetGamma"])
        if curr_val == 0:
            zero_crossings.append(float(df.loc[i, "Strike"]))
            continue
        if curr_val * next_val < 0:
            strike_curr = float(df.loc[i, "Strike"])
            strike_next = float(df.loc[i + 1, "Strike"])
            span = abs(curr_val) + abs(next_val)
            frac = abs(curr_val) / span if span else 0.5
            crossing = strike_curr + (strike_next - strike_curr) * frac
            zero_crossings.append(crossing)

    gap = float(spot - magnet_strike)
    gap_abs = abs(gap)
    spot_ref = abs(float(spot)) if spot not in (0, None) else 1.0
    gap_pct = gap_abs / max(spot_ref, 1e-6)

    gamma_strength = abs(magnet_gamma) / total_abs_gamma if total_abs_gamma else 0.0

    inside_band = lower_bound <= spot <= upper_bound if lower_bound <= upper_bound else upper_bound <= spot <= lower_bound

    if magnet_gamma > 0:
        if spot < min(lower_bound, upper_bound):
            bias_note = "Spot below positive gamma band — upward mean-reversion bias"
            positional_bias = 1.0
        elif spot > max(lower_bound, upper_bound):
            bias_note = "Spot above positive gamma band — downward mean-reversion bias"
            positional_bias = 1.0
        elif inside_band:
            bias_note = "Spot pinned inside positive gamma band"
            positional_bias = 0.6
        else:
            bias_note = "Spot between gamma flip levels"
            positional_bias = 0.8
        bias_factor = 1.0
    elif magnet_gamma < 0:
        bias_note = "Peak gamma is negative — expect trend-following flows"
        positional_bias = 0.3
        bias_factor = 0.3
    else:
        bias_note = "No dominant gamma peak"
        positional_bias = 0.2
        bias_factor = 0.2

    gap_signal = float(np.tanh(gap_pct * 5.0))
    score = gap_signal * gamma_strength * positional_bias * bias_factor

    gap_direction = "above" if gap < 0 else "below"
    if math.isclose(gap, 0.0, abs_tol=1e-6):
        gap_direction = "at"

    metrics = {
        "has_data": True,
        "magnet_strike": magnet_strike,
        "magnet_gamma": magnet_gamma,
        "magnet_sign": magnet_sign,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "zero_crossings": sorted(set(zero_crossings)),
        "spot": float(spot),
        "gap": gap,
        "gap_abs": gap_abs,
        "gap_pct": gap_pct,
        "gap_direction": gap_direction,
        "gamma_strength": gamma_strength,
        "total_abs_gamma": total_abs_gamma,
        "net_gamma": net_gamma,
        "score": score,
        "bias_note": bias_note,
        "inside_band": inside_band,
        "positive_gamma_share": float(df.loc[df["NetGamma"] > 0, "NetGamma"].sum()),
        "negative_gamma_share": float(df.loc[df["NetGamma"] < 0, "NetGamma"].sum()),
        "call_gamma_total": float(df["CallGamma"].sum()),
        "put_gamma_total": float(df["PutGamma"].sum()),
    }
    return metrics

def interpret_net_gex(df_net, S, offset=25):
    """
    Returns a list of strings with automated interpretation:
     - magnet strike(s)
     - expected mean-reversion/trend bias
     - steep GEX slope zones
     - gamma-flip strikes
    """
    interp = []
    strikes = df_net["Strike"].values
    gex_vals = df_net["GEX"].values

    # 1) Magnet (max GEX)
    idx_max = gex_vals.argmax()
    magnet = strikes[idx_max]
    interp.append(f"🔴 **Peak Net GEX** at strike {magnet:.1f}")

    # 2) Bias: price vs magnet
    if S < magnet:
        interp.append(f"⚠️ **Price {S:.2f}** is below the magnet → mild upward bias (mean‐revert toward {magnet:.1f})")
    elif S > magnet:
        interp.append(f"⚠️ **Price {S:.2f}** is above the magnet → mild downward bias (mean‐revert toward {magnet:.1f})")
    else:
        interp.append(f"⚖️ Price is exactly at the magnet → expect pinning and low volatility")

    # 3) Steepest slope (largest adjacent |ΔGEX|)
    deltas = np.diff(gex_vals)
    idx_slope = np.abs(deltas).argmax()
    s_low  = strikes[idx_slope]
    s_high = strikes[idx_slope+1]
    interp.append(f"🚀 **Steep GEX** slope between {s_low:.1f}→{s_high:.1f} → potential volatility acceleration through this zone")

    # 4) Gamma‐flip zones (where Net GEX crosses zero)
    signs = np.sign(gex_vals)
    flips = np.where(np.diff(signs)!=0)[0]
    if len(flips):
        zones = [f"{strikes[i]:.1f}↔{strikes[i+1]:.1f}" for i in flips]
        interp.append("🔵 **Gamma-flip** at zone(s): " + ", ".join(zones))
    else:
        interp.append("🔵 **No gamma-flip zones in S±offset range**")

    return interp

def compute_net_gamma_exposure(chain, S, offset=25):
    """
    Filter strikes within S ± OFFSET, extract gamma from chain[i]['greeks']['gamma'],
    compute Gamma Exposure = Y x open_interest x contract_size,
    then Net GEX = call GEX - put GEX per strike.
    """
    rows = []
    for opt in chain:
        K       = opt["strike"]
        if not (S - offset <= K <= S + offset):
            continue

        # pull gamma from nested greeks dict
        gamma_api = opt.get("greeks", {}).get("gamma")
        if gamma_api is None:
            continue

        oi        = opt.get("open_interest", 0)
        multiplier= opt.get("contract_size", 100)
        gex       = gamma_api * oi * multiplier
        rows.append({
            "Strike": K,
            "GEX":    gex,
            "Side":   opt["option_type"].upper()
        })

    if not rows:
        return pd.DataFrame(columns=["Strike","Net GEX"])

    df = pd.DataFrame(rows)
    calls = df[df["Side"] == "CALL"].set_index("Strike")["GEX"]
    puts  = df[df["Side"] == "PUT"].set_index("Strike")["GEX"]
    net   = calls.sub(puts, fill_value=0).reset_index().rename(columns={0:"Net GEX"})
    net.columns = ["Strike","Net GEX"]
    return net.sort_values("Strike")


def cache_current_data(df_current, timestamp, CACHE_FILE):
    """Append current df (with columns Strike, Net GEX) + Timestamp, keep last 10 batches."""
    df = df_current.copy()
    df["Timestamp"] = timestamp
    if os.path.exists(CACHE_FILE):
        cache = pd.read_csv(CACHE_FILE)
    else:
        cache = pd.DataFrame()
    cache = pd.concat([cache, df], ignore_index=True)
    cache["Timestamp"] = pd.to_datetime(cache["Timestamp"], errors="coerce")
    cache = cache.dropna(subset=["Timestamp"])
    times = cache["Timestamp"].drop_duplicates().nlargest(10)
    cache = cache[cache["Timestamp"].isin(times)]
    cache.to_csv(CACHE_FILE, index=False)
    return cache


def get_previous_query(df_cache, current_ts):
    """Return the df for the batch immediately before current_ts."""
    df = df_cache.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    times = df["Timestamp"].drop_duplicates().sort_values(ascending=False)
    if len(times) < 2:
        return None
    prev = times[times < current_ts].max()
    return df[df["Timestamp"] == prev]


def plot_net_gamma_exposure(df_net_current, previous_df, S, ticker, expiration, df_cache):
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))

    # ── LEFT PANEL ────────────────────────────────────────────────────────
    # 1) Make a copy and stringify the Strike so it's treated categorically
    df_plot = df_net_current.copy()
    df_plot['StrikeCat'] = df_plot['Strike'].map(lambda s: f"{s:.1f}")

    # 2) Draw the barplot using the categorical column
    sns.barplot(data=df_plot, x='StrikeCat', y='Net GEX', ax=ax1, color='steelblue')

    # 3) Draw the spot price as a vertical line at the matching category index
    spot_cat = f"{S:.1f}"
    if spot_cat in df_plot['StrikeCat'].values:
        idx = df_plot.index[df_plot['StrikeCat'] == spot_cat][0]
        ax1.axvline(idx, color='red', linestyle='--', linewidth=2, label=f"Spot={S:.2f}")

    # 4) Annotate deltas from previous query, if present
    if previous_df is not None:
        prev = previous_df.rename(columns={'Net GEX':'Net GEX_prev'})
        merged = df_plot.merge(prev, left_on='Strike', right_on='Strike')
        for _, row in merged.iterrows():
            d = row['Net GEX'] - row['Net GEX_prev']
            if abs(d) > 0.1:
                ax1.text(
                    row.name,                    # row.name is the int index
                    row['Net GEX'] * 1.02,
                    f"{d:+.0f}",
                    ha='center',
                    fontsize=9
                )

    # 5) Highlight the max Net GEX
    max_i = df_plot['Net GEX'].idxmax()
    ax1.scatter(
        max_i,
        df_plot.loc[max_i, 'Net GEX'],
        color='crimson',
        s=100,
        label='Max Net GEX'
    )

    ax1.set_title(f"{ticker} Net GEX (Exp: {expiration})")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Net Gamma Exposure")
    ax1.legend(loc='upper right')
    ax1.set_xticklabels(df_plot['StrikeCat'], rotation=45, ha='right')

    # ── RIGHT PANEL ───────────────────────────────────────────────────────
    hist = df_cache.copy()
    hist['Timestamp'] = pd.to_datetime(hist['Timestamp'], errors='coerce')
    hist = hist.dropna(subset=['Timestamp'])
    max_strike = df_plot.loc[max_i, 'Strike']
    hist_strike = hist[hist['Strike']==max_strike].sort_values('Timestamp')

    ax2.set_title(f"History @ Strike {max_strike}")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Net GEX")

    if hist_strike.empty:
        ax2.text(0.5, 0.5, "No history", ha='center', va='center')
    else:
        ax2.plot(
            hist_strike['Timestamp'],
            hist_strike['Net GEX'],
            marker='o', color='navy', label='Net GEX'
        )
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        ax2.legend()

    plt.tight_layout()
    return plt

def plot_iv_skew(iv_skew_df, S):
    plt.figure(figsize=(8,4))
    plt.plot(iv_skew_df['strike'], iv_skew_df['iv_skew'], marker='o')
    plt.axhline(0, color='gray', linewidth=1)
    plt.axvline(S, color='red', linestyle='--', label=f"Spot = {S:.2f}")
    plt.title("IV Skew (Put IV - Call IV) by Strike")
    plt.xlabel("Strike")
    plt.ylabel("IV Skew")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt

def plot_exposure(df, value_column, label, ticker, offset, spot):
    """Generate a mirror bar chart for option exposure."""
    title = f"{ticker} {label} by Strike & DTE (±{offset} around {spot:.1f})"
    fig = px.bar(
        df,
        x=value_column,
        y='strike',
        color='DTE',
        orientation='h',
        facet_col='option_type',
        category_orders={'option_type': ['put', 'call']},
        labels={value_column: label, 'strike': 'Strike', 'DTE': 'Days to Exp'},
        title=title,
        template='presentation'
    )
    fig.update_layout(barmode='relative', height=800)
    fig.update_yaxes(
        tickfont=dict(size=16),
        autorange='reversed',
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        griddash='dot'
    )
    fig.update_xaxes(
        tickfont=dict(size=16),
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        griddash='dot'
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
    return fig

def show_put_call_ratios(vol_ratio, oi_ratio):
    s = f"📈 Put/Call Volume Ratio: {vol_ratio:.2f}\n"
    s += f"\n📊 Put/Call OI Ratio:     {oi_ratio:.2f}"
    return s

def plot_put_call_ratios(vol_ratio, oi_ratio):
    labels = ['Volume', 'Open Interest']
    vals   = [vol_ratio, oi_ratio]
    colors = ['#d62728' if v>1 else '#1f77b4' for v in vals]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=vals,
            marker_color=colors,
            width=0.4    # narrow each bar
        )
    )
    fig.add_hline(y=1, line_dash='dash', line_color='gray')
    fig.update_layout(
        title="Put/Call Ratios",
        xaxis_title="Metric",
        yaxis_title="Ratio",
        template="plotly_white",
        width=300,           # fixed chart width
        height=400,
        bargap=0.2,          # space between bars
        margin=dict(l=40, r=20, t=40, b=40)
    )
    fig.update_xaxes(tickfont=dict(size=20))
    fig.update_yaxes(tickfont=dict(size=14))
    return fig

def plot_volume_spikes(spikes_df, offset=None, spot=None):
    # Sort descending by vol_oi and take top 10
    spikes = spikes_df.sort_values('vol_oi', ascending=False).head(10).copy()
    spikes['strike_str'] = spikes['strike'].astype(str)

    title = "Top 10 Volume/OI Spikes"
    if offset is not None and spot is not None:
        title += f" (±{offset} around {spot:.1f})"

    # Create vertical bar chart
    fig = px.bar(
        spikes,
        x='strike_str',
        y='vol_oi',
        color_discrete_sequence=['#FFA500'],
        labels={'strike_str': 'Strike', 'vol_oi': 'Volume / OpenInterest'},
        template='plotly_white'
    )

    # Ensure bars are in the same descending order
    fig.update_layout(
        title=title,
        xaxis_tickangle=0,
        xaxis={'categoryorder': 'array', 'categoryarray': spikes['strike_str']},
        yaxis_tickformat='.2f',
        margin=dict(l=60, r=20, t=40, b=60),
        height=400
    )
    fig.update_xaxes(tickfont=dict(size=20))
    fig.update_yaxes(tickfont=dict(size=14))
    return fig


def plot_volume_spikes_stacked(spikes_df, offset=None, spot=None):
    title = "Top 10 Volume/OI Spikes by Strike"
    if offset and spot:
        title += f" (±{offset} around {spot:.1f})"
    # Prepare categorical strike labels
    spikes_df['strike_str'] = spikes_df['strike'].astype(str)
    # Melt for px.bar
    melt = spikes_df.melt(
        id_vars=['strike_str'], 
        value_vars=['vol_oi_put','vol_oi_call'],
        var_name='Side', value_name='Volume/OI'
    )
    # Map Side to nice labels
    melt['Side'] = melt['Side'].map({'vol_oi_put':'Put', 'vol_oi_call':'Call'})
    fig = px.bar(
        melt,
        x='strike_str',
        y='Volume/OI',
        color='Side',
        category_orders={'strike_str': spikes_df['strike_str'].tolist()},
        color_discrete_map={'Put':'#1f77b4','Call':'#d62728'},
        barmode='stack',
        labels={'strike_str':'Strike'},
        template='plotly_white',
        title=title
    )
    fig.update_layout(xaxis_tickangle=45, yaxis_tickformat='.2f')
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=14))
    return fig

def get_intraday_prices_with_prev_close(ticker: str, interval: str = "30m") -> pd.Series:
    df = yf.Ticker(ticker).history(period="2d", interval=interval)
    df['date'] = df.index.date
    dates = sorted(df['date'].unique())
    yday, today = dates[0], dates[-1]

    # yesterday’s final bar at 16:00
    df_y = df[df['date'] == yday]
    yday_close = df_y['Close'].iloc[-1]
    tz = df.index.tz
    yday_ts = datetime.combine(yday, time(16, 0)).replace(tzinfo=tz)
    ser_y = pd.Series([yday_close], index=pd.DatetimeIndex([yday_ts], tz=tz), name="Close")

    # today’s intraday
    df_t = df[df['date'] == today]['Close'].copy()
    df_t.name = "Close"

    # combine
    combined = pd.concat([ser_y, df_t]).sort_index()
    return combined

def get_delta_exposure_at_times(
    ticker, expiration, tradier_token, offset, price_series: pd.Series
) -> pd.Series:
    exposures = []
    api = TradierAPI(tradier_token)

    # skip the first index (yesterday’s close)
    for ts, spot in price_series.iloc[1:].items():
        spot = float(spot)
        data = api.option_chain(
            ticker,
            expiration,
            greeks="true",
            include_all_roots=True,
        )
        df_chain = pd.DataFrame(data)
        df_chain['strike'] = df_chain['strike'].astype(float)
        df_chain['delta']  = df_chain['greeks'].apply(lambda g: g.get('delta', 0.0))
        df_chain['oi']     = df_chain['open_interest'].astype(int)

        mask = (df_chain['strike'] >= spot-offset) & (df_chain['strike'] <= spot+offset)
        df_chain = df_chain.loc[mask]

        df_chain['side'] = df_chain['option_type'].str.lower().map({'call':1,'put':-1})
        df_chain['d_exposure'] = (
            df_chain['delta']*
            df_chain['oi']*
            df_chain['contract_size']*
            df_chain['side']
        )
        exposures.append(df_chain['d_exposure'].sum())

    # build a Series aligned to the *trading-day* timestamps
    idx = price_series.index[1:]  # skip the first
    return pd.Series(exposures, index=idx, name="Net Delta Exposure")

def plot_price_and_delta_projection(
    ticker, expiration, tradier_token, offset, interval="30m"
):
    # 1) get price series (with yday close)
    price_series = get_intraday_prices_with_prev_close(ticker, interval)

    # 2) compute exposures only for today’s bars
    delta_series = get_delta_exposure_at_times(
        ticker, expiration, tradier_token, offset, price_series
    )

    # 3) linear trend on exposures
    hours = np.array([
        (ts - delta_series.index[0]).total_seconds()/3600
        for ts in delta_series.index
    ])
    coeff = np.polyfit(hours, delta_series.values, 1)
    trend = np.poly1d(coeff)

    # project to end-of-day 16:00
    last_ts = delta_series.index[-1]
    tz = last_ts.tzinfo
    end_ts = datetime.combine(last_ts.date(), time(16,0)).replace(tzinfo=tz)
    end_hour = (end_ts - delta_series.index[0]).total_seconds()/3600
    proj = float(trend(end_hour))

    # 4) make subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # spot price (includes yday close)
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode="lines+markers",
            name="Spot Price",
            line=dict(color="blue")
        ),
        secondary_y=False
    )

    # delta exposure bars (today only)
    fig.add_trace(
        go.Bar(
            x=delta_series.index,
            y=delta_series.values,
            name="Net Delta Exposure",
            marker_color="red",
            opacity=0.6
        ),
        secondary_y=True
    )

    # projection point
    fig.add_trace(
        go.Scatter(
            x=[end_ts], y=[proj],
            mode="markers+lines",
            name="Projected Exposure",
            marker=dict(color="green", size=10),
            line=dict(dash="dash")
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=f"{ticker} Price & Net Delta Exposure Projection",
        xaxis=dict(title="Time"),
        template="plotly_white",
        height=450, width=900
    )
    fig.update_yaxes(title_text="Spot Price", secondary_y=False)
    fig.update_yaxes(title_text="Delta Exposure", secondary_y=True)

    return fig


def generate_binomial_tree(S0, K, T, r, sigma, steps, option_type="call"):
    """Generate a CRR binomial tree of underlying and option values."""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    prices = np.zeros((steps + 1, steps + 1))
    prices[0, 0] = S0
    for i in range(1, steps + 1):
        prices[i, 0] = prices[i - 1, 0] * u
        for j in range(1, i + 1):
            prices[i, j] = prices[i - 1, j - 1] * d

    option = np.zeros_like(prices)
    if option_type.lower() == "call":
        option[steps, : steps + 1] = np.maximum(prices[steps, : steps + 1] - K, 0)
    else:
        option[steps, : steps + 1] = np.maximum(K - prices[steps, : steps + 1], 0)

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option[i, j] = math.exp(-r * dt) * (
                p * option[i + 1, j] + (1 - p) * option[i + 1, j + 1]
            )

    data = []
    for i in range(steps + 1):
        for j in range(i + 1):
            data.append(
                {
                    "step": i,
                    "node": j,
                    "price": prices[i, j],
                    "option": option[i, j],
                }
            )

    return pd.DataFrame(data)


def plot_binomial_tree(df):
    """Plot a binomial tree using plotly with prices and option values."""
    import plotly.graph_objects as go

    steps = int(df["step"].max())
    fig = go.Figure()

    for i in range(steps + 1):
        df_step = df[df["step"] == i]
        fig.add_trace(
            go.Scatter(
                x=[i] * len(df_step),
                y=df_step["price"],
                mode="markers+text",
                text=[
                    f"S={row.price:.2f}<br>O={row.option:.2f}"
                    for row in df_step.itertuples()
                ],
                textposition="top center",
                marker=dict(size=10, color="blue"),
                showlegend=False,
            )
        )

    for i in range(steps):
        for j in range(i + 1):
            y0 = df[(df["step"] == i) & (df["node"] == j)]["price"].values[0]
            y1 = df[(df["step"] == i + 1) & (df["node"] == j)]["price"].values[0]
            fig.add_shape(type="line", x0=i, y0=y0, x1=i + 1, y1=y1, line=dict(color="gray"))
            y1 = df[(df["step"] == i + 1) & (df["node"] == j + 1)]["price"].values[0]
            fig.add_shape(type="line", x0=i, y0=y0, x1=i + 1, y1=y1, line=dict(color="gray"))

    fig.update_layout(
        title="Binomial Tree",
        xaxis_title="Step",
        yaxis_title="Underlying Price",
        template="seaborn",
    )

    return fig
