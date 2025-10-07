from openai import OpenAI
import tiktoken
from helpers import get_market_snapshot, augment_payload_with_extras
import streamlit as st
import pandas as pd
from db import save_analysis, get_total_token_usage
import yaml
import os
import textwrap
from html import escape
from typing import Optional, Dict, List, Tuple, Sequence


# Load configuration from YAML file
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)


# New helper functions for modularized markdown building


def _resolve_secret(*names):
    for name in names:
        try:
            value = st.secrets[name]
        except Exception:
            value = None
        if value:
            return value
    return None


def resolve_openai_credentials() -> Dict[str, Optional[str]]:
    """Return OpenAI credentials from secrets/environment with normalisation."""

    api_key = _resolve_secret("OPENAI_API_KEY", "openai_api_key", "OPENAI_KEY")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    api_key = api_key.strip() if isinstance(api_key, str) else None

    organization = _resolve_secret("OPENAI_ORG", "OPENAI_ORGANIZATION")
    if not organization:
        organization = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION")
    organization = organization.strip() if isinstance(organization, str) else None

    project = _resolve_secret("OPENAI_PROJECT", "OPENAI_DEFAULT_PROJECT")
    if not project:
        project = os.getenv("OPENAI_PROJECT") or os.getenv("OPENAI_DEFAULT_PROJECT")
    project = project.strip() if isinstance(project, str) else None

    return {
        "api_key": api_key,
        "organization": organization,
        "project": project,
    }


def _is_model_excluded(model_id: str) -> bool:
    """Return True when the model clearly does not support text advice."""

    excluded_keywords = [
        "embedding",
        "audio",
        "image",
        "vision",
        "whisper",
        "tts",
        "realtime",
        "omni-moderation",
        "moderation",
    ]
    lowered = model_id.lower()
    return any(keyword in lowered for keyword in excluded_keywords)


def _score_model(model_id: str) -> Tuple[float, List[str]]:
    """Assign a heuristic score and justification for a model identifier."""

    lowered = model_id.lower()
    score = 0.0
    reasons: List[str] = []

    quality_rules = [
        ("gpt-4.1", 6, "Latest GPT-4.1 reasoning tier"),
        ("gpt-4o", 5, "GPT-4o balanced reasoning and cost"),
        ("o4", 4, "o4 reasoning family"),
        ("o3", 3, "o3 reasoning tuned for analysis"),
        ("o1", 2, "o1 reasoning model"),
        ("gpt-4", 2, "GPT-4 generation"),
        ("gpt-3.5", 1, "GPT-3.5 generation"),
    ]
    for token, weight, reason in quality_rules:
        if token in lowered:
            score += weight
            reasons.append(reason)

    finance_rules = [
        ("finance", 3, "Finance-tuned variant"),
        ("market", 2, "Market or trading focused naming"),
        ("analysis", 1, "Optimised for analytical tasks"),
        ("research", 1, "Research oriented variant"),
    ]
    for token, weight, reason in finance_rules:
        if token in lowered:
            score += weight
            reasons.append(reason)

    if "mini" in lowered:
        score -= 1
        reasons.append("Cost-efficient mini tier")

    if not reasons:
        reasons.append("General purpose reasoning model")

    return max(score, 0.0), reasons


def discover_financial_models(creds: Dict[str, Optional[str]]) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """Return OpenAI models ranked for financial analysis tasks."""

    api_key = creds.get("api_key") if isinstance(creds, dict) else None
    if not api_key:
        return [], "Missing API key"

    client_kwargs = {"api_key": api_key}
    organization = creds.get("organization") if isinstance(creds, dict) else None
    project = creds.get("project") if isinstance(creds, dict) else None
    if organization:
        client_kwargs["organization"] = organization
    if project:
        client_kwargs["project"] = project

    try:
        client = OpenAI(**client_kwargs)
        response = client.models.list()
    except Exception as exc:
        return [], str(exc)

    models: List[Dict[str, object]] = []
    for item in getattr(response, "data", []):
        model_id = getattr(item, "id", None)
        if not model_id:
            continue
        if _is_model_excluded(model_id):
            continue
        score, reasons = _score_model(model_id)
        models.append(
            {
                "id": model_id,
                "score": score,
                "reasons": reasons,
            }
        )

    models.sort(key=lambda entry: (entry["score"], entry["id"]), reverse=True)
    return models, None

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
def create_data_packet(ticker, overview_summary, pos_summary, iv_summary, ratios_summary, news_summary, snap_summary):
    system_msg = {
        "role": "system",
        "content": (
            "You are an experienced volatility and macro strategist. "
            "Combine dealer positioning, volatility structure, and broad market context "
            "to craft actionable option trade ideas."
        ),
    }
    user_msg = {
        "role": "user",
        "content": textwrap.dedent(
            f"""
            Ticker: {ticker}
            Overview Metrics: {overview_summary}
            Positioning data gamma and delta exposure:
            {pos_summary}
            IV Skew: {iv_summary}
            Ratios: {ratios_summary}
            News Headlines: {news_summary}
            Snapshot Summary: {snap_summary}

            Please:
            - Summarise expected dealer hedging behaviour and current positioning drivers.
            - Discuss the state of overall market health/regime before recommending trades.
            - Only recommend structures where I buy calls or puts (no spreads, straddles, or other multi-leg trades).
            - Deliver concrete ideas for three horizons: 1-2 weeks, 1-2 months, and 6-12 months (0.5-1 year).
            - Explicitly link every idea to broader market health or macro positioning context.
            - Reference data beyond the stated horizons when it materially improves risk context.
            - Provide a trade confidence score (1-100) for each recommendation.
            - Identify max-pain, key supports, and resistances.
            """
        ).strip(),
    }
    return {"messages": [system_msg, user_msg]}

# Helper function to estimate token count
def estimate_token_count(data_packet, model_name):
    tokens = None
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    try:
        tokens = sum(len(enc.encode(m["content"])) for m in data_packet["messages"])
        st.write(f"Estimated prompt tokens for `{model_name}`: **{tokens}**")
    except Exception as e:
        print(f"Token estimation error: {e}")
        tokens = None

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
def call_openai_api(data_packet, creds, model_name: str):
    api_key = creds.get("api_key") if isinstance(creds, dict) else None
    if not api_key:
        raise ValueError("OpenAI API key is missing; cannot send request.")

    client_kwargs = {"api_key": api_key}
    organization = creds.get("organization") if isinstance(creds, dict) else None
    project = creds.get("project") if isinstance(creds, dict) else None
    if organization:
        client_kwargs["organization"] = organization
    if project:
        client_kwargs["project"] = project

    client = OpenAI(**client_kwargs)
    st.write("Model used:", model_name)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=data_packet["messages"],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

# Refactored openai_query function
def _build_ai_form_key(label: str, ticker: str, exp: Optional[Sequence[str]]) -> str:
    exp_fragment = "-".join(exp) if isinstance(exp, (list, tuple)) else (str(exp) if exp else "")
    return f"{label}_{ticker}_{exp_fragment}" if exp_fragment else f"{label}_{ticker}"


def _render_model_discovery_table(models: Sequence[Dict[str, object]]) -> str:
    """Return HTML for a stylised model discovery table."""

    table_style = """
    <style>
        .model-discovery-card {
            margin-top: 0.85rem;
            border-radius: 18px;
            padding: 1rem 1.25rem 0.6rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 20px 42px rgba(2, 6, 23, 0.42);
        }

        .model-discovery-card table {
            width: 100%;
            border-collapse: collapse;
        }

        .model-discovery-card thead th {
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            color: #a5b4fc;
            padding: 0.65rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        }

        .model-discovery-card tbody td {
            padding: 0.85rem 0.75rem;
            font-size: 0.92rem;
            color: #e2e8f0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }

        .model-discovery-card tbody tr:last-child td {
            border-bottom: none;
        }

        .model-discovery-card .model-rank {
            width: 3.25rem;
            font-weight: 600;
            color: #94a3b8;
        }

        .model-discovery-card .model-id {
            font-family: "JetBrains Mono", "Fira Code", "SFMono-Regular", monospace;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .model-discovery-card .model-score {
            font-weight: 600;
            color: #38bdf8;
        }

        .model-discovery-card .model-strengths {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }

        .model-discovery-card .model-strength {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.18);
            border: 1px solid rgba(96, 165, 250, 0.35);
            font-size: 0.75rem;
            color: #bfdbfe;
            white-space: nowrap;
        }

        .model-discovery-card tr.model-row--top td {
            position: relative;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(14, 165, 233, 0.12));
            border-bottom-color: rgba(56, 189, 248, 0.26);
        }

        .model-discovery-card tr.model-row--top td:first-child {
            border-top-left-radius: 12px;
        }

        .model-discovery-card tr.model-row--top td:last-child {
            border-top-right-radius: 12px;
        }

        @media (max-width: 768px) {
            .model-discovery-card table,
            .model-discovery-card tbody,
            .model-discovery-card tr,
            .model-discovery-card td,
            .model-discovery-card thead {
                display: block;
            }

            .model-discovery-card thead {
                display: none;
            }

            .model-discovery-card tbody td {
                border-bottom: none;
                padding: 0.45rem 0;
            }

            .model-discovery-card tbody tr {
                padding: 0.65rem 0;
                border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            }

            .model-discovery-card .model-strengths {
                margin-top: 0.4rem;
            }
        }
    </style>
    """

    row_template = textwrap.dedent(
        """
        <tr class="{row_class}">
            <td class="model-rank">#{rank:02d}</td>
            <td class="model-id">{model_id}</td>
            <td class="model-score">{score}</td>
            <td>
                <div class="model-strengths">{strengths}</div>
            </td>
        </tr>
        """
    )

    rows_html: List[str] = []
    for index, entry in enumerate(models, start=1):
        model_id = escape(str(entry.get("id", "")))
        raw_score = entry.get("score", "")
        if isinstance(raw_score, (int, float)):
            score = f"{raw_score:.1f}"
        else:
            score = escape(str(raw_score))

        reasons = entry.get("reasons") or []
        badges = "".join(
            f'<span class="model-strength">{escape(str(reason))}</span>'
            for reason in reasons
        )

        row_class = "model-row model-row--top" if index == 1 else "model-row"
        if not badges:
            badges = '<span class="model-strength">General purpose</span>'

        rows_html.append(
            row_template.format(
                row_class=row_class,
                rank=index,
                model_id=model_id,
                score=score,
                strengths=badges,
            )
        )

    rows_markup = "".join(rows_html)
    table_html = textwrap.dedent(
        """
        <div class="model-discovery-card">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Score</th>
                        <th>Highlights</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    ).format(rows=rows_markup)

    return textwrap.dedent(table_style) + table_html


def render_model_selection(ticker: str, exp, creds: Optional[Dict[str, Optional[str]]] = None):
    """Render model discovery and selection UI, returning creds and chosen model."""

    openai_creds = creds or resolve_openai_credentials()
    api_key = openai_creds.get("api_key") if isinstance(openai_creds, dict) else None
    if not api_key:
        st.error("OpenAI API key is not configured.")
        return openai_creds, None

    st.subheader("Model Discovery")
    models, discovery_error = discover_financial_models(openai_creds)
    if discovery_error:
        st.warning(
            "Unable to enumerate OpenAI models automatically — falling back to configured defaults.\n"
            f"Details: {discovery_error}"
        )

    if models:
        st.markdown(_render_model_discovery_table(models), unsafe_allow_html=True)
    else:
        st.info(
            "Using fallback model suggestions because no suitable models were returned."
        )

    candidate_ids = [entry["id"] for entry in models] if models else []
    fallback_pool = [
        CONFIG.get("openai", {}).get("model", "gpt-4o"),
        "gpt-4o-mini",
        "gpt-4.1",
        "o4-mini",
    ]
    options = candidate_ids or fallback_pool
    seen = set()
    unique_options = []
    for model_id in options:
        if model_id and model_id not in seen:
            seen.add(model_id)
            unique_options.append(model_id)

    if not unique_options:
        st.error("No OpenAI models are available to select.")
        return openai_creds, None

    default_model = st.session_state.get(
        "ai_selected_model",
        CONFIG.get("openai", {}).get("model", unique_options[0]),
    )
    if default_model not in unique_options:
        default_model = unique_options[0]

    selection_key = _build_ai_form_key("ai_model_select", ticker, exp)
    if selection_key not in st.session_state:
        st.session_state[selection_key] = default_model

    chosen_model = st.selectbox(
        "Select an OpenAI model for the financial analysis",
        unique_options,
        key=selection_key,
    )
    st.session_state["ai_selected_model"] = chosen_model
    st.caption(
        "Models are ranked heuristically based on reasoning strength, finance-focused naming, and cost tier."
    )

    return openai_creds, chosen_model


def openai_query(
    df_net,
    iv_skew_df,
    vol_ratio,
    oi_ratio,
    articles,
    spot,
    offset,
    ticker,
    exp,
    selected_model,
    openai_creds,
):
    api_key = openai_creds.get("api_key") if isinstance(openai_creds, dict) else None
    if not api_key:
        st.error("OpenAI API key is not configured.")
        return

    if not selected_model:
        st.error("Select an OpenAI model before preparing the analysis.")
        return

    st.markdown(f"**Using model:** `{selected_model}`")

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
    
    data_packet = create_data_packet(
        ticker,
        overview_summary,
        pos_summary,
        iv_summary,
        ratios_summary,
        news_summary,
        snap_summary,
    )
    st.subheader("Data Packet JSON")
    st.json(data_packet)

    tokens = estimate_token_count(data_packet, selected_model)

    st.info("Review the prepared data packet and estimated usage before proceeding.")

    form_key = _build_ai_form_key("ai_confirmation", ticker, exp)
    with st.form(form_key):
        proceed = st.checkbox("I have reviewed the packet and wish to proceed with the OpenAI request.")
        pin_entry = st.text_input("Enter security PIN", type="password")
        submitted = st.form_submit_button("Send to OpenAI")

    if not submitted:
        st.info("Submit the confirmation form to continue or adjust the inputs above.")
        return

    if not proceed:
        st.warning("Confirmation required before the request can be sent.")
        return

    pin_expected = st.secrets.get("AI_PIN")
    if not pin_expected:
        st.error("AI_PIN secret is not configured. Set it in Streamlit secrets to continue.")
        return

    if pin_entry != str(pin_expected):
        st.error("❌ Incorrect PIN, try again.")
        return

    st.markdown(f"**Selected model:** `{selected_model}`")

    st.success("PIN accepted — running AI…")
    try:
        analysis = call_openai_api(data_packet, openai_creds, selected_model)
    except ValueError as exc:
        st.error(str(exc))
        return
    if analysis:
        st.markdown(f"### AI Trade Analysis\n{analysis}")
        save_analysis(
            ticker=ticker,
            expirations=exp,  # replaced selected_exps with exp
            payload=snapshot,
            response=analysis,  # replaced response variable with analysis
            token_count=tokens or 0,
        )
        st.success("✅ Analysis saved.")
        st.session_state.want_ai = False
