import requests
from openai import OpenAI
import tiktoken
from datetime import datetime, timedelta
from helpers import get_market_snapshot, augment_payload_with_extras
import streamlit as st
import pandas as pd
from db import save_analysis, get_total_token_usage
import yaml
import os
import textwrap
from typing import Optional, Dict


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
def estimate_token_count(data_packet):
    model_name = CONFIG.get("openai", {}).get("model", "gpt-4o")
    tokens = None
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    try:
        tokens = sum(len(enc.encode(m["content"])) for m in data_packet["messages"])
        st.write(f"Estimated prompt tokens: **{tokens}**")
    except Exception as e:
        print(f"Token estimation error: {e}")
        tokens = None

    return tokens


MODEL_PRICING_PER_MTOKENS = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.6, "output": 2.4},
    "gpt-4.1": {"input": 10.0, "output": 30.0},
    "gpt-4.1-mini": {"input": 1.5, "output": 6.0},
}


def _normalise_model_key(model_name):
    return (model_name or "").lower()


def get_model_pricing(model_name):
    key = _normalise_model_key(model_name)
    if key in MODEL_PRICING_PER_MTOKENS:
        return MODEL_PRICING_PER_MTOKENS[key]
    return MODEL_PRICING_PER_MTOKENS.get("gpt-4o")


class OpenAICreditError(RuntimeError):
    """Raised when all OpenAI billing queries fail."""


def _build_openai_headers(
    api_key: str, organization: Optional[str], project: Optional[str]
) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project
    return headers


def _call_openai_get(url: str, headers: Dict[str, str]) -> Dict:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data or {}


def _fetch_credit_via_grants(headers: Dict[str, str], organization: Optional[str]):
    """Try the direct credit_grants endpoint variants."""

    urls = []
    if organization:
        urls.append(
            f"https://api.openai.com/v1/organizations/{organization}/billing/credit_grants"
        )
        urls.append(
            f"https://api.openai.com/v1/organization/{organization}/billing/credit_grants"
        )
    urls.extend(
        [
            "https://api.openai.com/v1/billing/credit_grants",
            "https://api.openai.com/v1/dashboard/billing/credit_grants",
            "https://api.openai.com/dashboard/billing/credit_grants",
        ]
    )

    errors = []
    for url in urls:
        try:
            payload = _call_openai_get(url, headers)
            if payload:
                return {
                    "total": payload.get("total_granted"),
                    "used": payload.get("total_used"),
                    "available": payload.get("total_available"),
                }
        except requests.HTTPError as exc:
            detail = None
            if exc.response is not None:
                try:
                    detail = exc.response.json().get("error", {}).get("message")
                except Exception:
                    detail = exc.response.text
            status = exc.response.status_code if exc.response is not None else None
            errors.append((url, status, detail))
            # Try other URL variants on 404/403/etc.
            continue
        except requests.RequestException as exc:
            errors.append((url, None, str(exc)))
            continue

    if errors:
        formatted = "; ".join(
            f"{url} -> status={status}, detail={detail}" for url, status, detail in errors
        )
        raise OpenAICreditError(formatted)
    raise OpenAICreditError("Unknown error contacting credit_grants endpoints.")


def _fetch_credit_via_usage(headers: Dict[str, str]):
    """Fallback to subscription + usage endpoints if credit grants fail."""

    now = datetime.utcnow().date()
    # Usage endpoint expects ISO dates; grab the last 90 days for coverage.
    start = (now - timedelta(days=90)).isoformat()
    end = (now + timedelta(days=1)).isoformat()

    subscription = _call_openai_get(
        "https://api.openai.com/v1/dashboard/billing/subscription", headers
    )
    usage = _call_openai_get(
        "https://api.openai.com/v1/dashboard/billing/usage"
        f"?start_date={start}&end_date={end}",
        headers,
    )

    total_granted = subscription.get("hard_limit_usd")
    system_limit = subscription.get("system_hard_limit_usd")
    total_limit = total_granted or system_limit

    total_used = None
    total_available = None

    if usage:
        raw_usage = usage.get("total_usage")
        if raw_usage is not None:
            # OpenAI returns usage in cents.
            total_used = raw_usage / 100.0
    if total_limit is not None and total_used is not None:
        total_available = max(total_limit - total_used, 0)

    if total_limit is None and total_used is None:
        raise OpenAICreditError("usage/subscription endpoints returned no totals")

    return {
        "total": total_limit,
        "used": total_used,
        "available": total_available,
    }


def fetch_openai_credit_balance(
    api_key: str, organization: Optional[str] = None, project: Optional[str] = None
):
    """Return the current OpenAI wallet credit information."""

    if not api_key or not api_key.strip():
        raise ValueError("An OpenAI API key is required to query credit balance.")

    headers = _build_openai_headers(api_key, organization, project)

    try:
        return _fetch_credit_via_grants(headers, organization)
    except OpenAICreditError as primary_err:
        # Attempt fallback through subscription/usage endpoints before surfacing error.
        try:
            return _fetch_credit_via_usage(headers)
        except requests.HTTPError as exc:
            raise exc
        except requests.RequestException as exc:
            raise exc
        except OpenAICreditError:
            raise primary_err


def display_credit_information(tokens_estimated, creds):
    if tokens_estimated is None:
        return

    api_key = creds.get("api_key") if isinstance(creds, dict) else None
    if not api_key:
        st.info("Configure OPENAI_API_KEY in secrets to display live credit balance.")
        return

    try:
        credit = fetch_openai_credit_balance(
            api_key,
            organization=creds.get("organization") if isinstance(creds, dict) else None,
            project=creds.get("project") if isinstance(creds, dict) else None,
        )
    except requests.HTTPError as exc:
        detail = None
        if exc.response is not None:
            try:
                detail = exc.response.json().get("error", {}).get("message")
            except Exception:
                detail = exc.response.text
        message = "Failed to fetch OpenAI credit balance"
        if detail:
            message += f": {detail}"
        else:
            message += "."
        st.error(message)
        return
    except requests.RequestException as exc:
        st.error(f"Network error while contacting OpenAI billing API: {exc}")
        return
    except ValueError as exc:
        st.error(str(exc))
        return
    except OpenAICreditError as exc:
        st.error(
            "Unable to retrieve OpenAI credit information automatically. "
            f"OpenAI returned errors for all billing endpoints: {exc}"
        )
        return

    if not credit:
        st.warning("Unable to parse OpenAI credit balance response.")
        return

    fragments = []
    available = credit.get("available")
    used = credit.get("used")
    total = credit.get("total")

    if available is not None:
        fragments.append(f"**${available:,.2f}** remaining")
    if used is not None and total is not None:
        fragments.append(f"${used:,.2f} used of ${total:,.2f} granted")
    elif used is not None:
        fragments.append(f"${used:,.2f} used")
    elif total is not None:
        fragments.append(f"${total:,.2f} granted")

    if fragments:
        st.write("OpenAI wallet balance: " + ", ".join(fragments) + ".")

    model_name = CONFIG.get("openai", {}).get("model", "gpt-4o")
    pricing = get_model_pricing(model_name)
    if pricing and pricing.get("input"):
        est_cost = (tokens_estimated / 1_000_000) * pricing["input"]
        st.write(
            f"Estimated prompt input cost at {model_name}: **${est_cost:,.4f}** "
            "(input-side pricing only)."
        )
        if available is not None:
            tokens_cover = available * (1_000_000 / pricing["input"])
            st.write(
                f"Remaining balance covers roughly **{tokens_cover:,.0f}** additional input tokens."
            )

    try:
        total_tokens_used = get_total_token_usage()
    except Exception as exc:
        print(f"Token usage aggregation error: {exc}")
    else:
        if total_tokens_used:
            st.write(
                f"Tokens logged across saved analyses so far: **{total_tokens_used:,.0f}**"
            )

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
def call_openai_api(data_packet, creds):
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
    openai_creds = resolve_openai_credentials()

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

    tokens = estimate_token_count(data_packet)
    display_credit_information(tokens, openai_creds)

    st.info("Review the prepared data packet and estimated usage before proceeding.")

    def _build_form_key(label: str) -> str:
        exp_fragment = "-".join(exp) if isinstance(exp, list) else str(exp)
        return f"{label}_{ticker}_{exp_fragment}" if exp_fragment else f"{label}_{ticker}"

    form_key = _build_form_key("ai_confirmation")
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

    api_key = openai_creds.get("api_key")
    if not api_key:
        st.error("OpenAI API key is not configured.")
        return

    st.success("PIN accepted — running AI…")
    try:
        analysis = call_openai_api(data_packet, openai_creds)
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
