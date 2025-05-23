import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go

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