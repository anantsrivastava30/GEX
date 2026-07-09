import json
import os
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
from openai import OpenAI

from quant_analysis.config import CONFIG
from quant_analysis.services.market_data import (
    compute_unusual_spikes,
    get_bond_yield_info,
    get_price_stats,
    get_vix_info,
)
from quant_analysis.storage.db import save_analysis


# ── Credentials ─────────────────────────────────────────────────────────────

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


def _client_from_creds(creds: Dict[str, Optional[str]]) -> OpenAI:
    client_kwargs = {"api_key": creds.get("api_key")}
    if creds.get("organization"):
        client_kwargs["organization"] = creds["organization"]
    if creds.get("project"):
        client_kwargs["project"] = creds["project"]
    return OpenAI(**client_kwargs)


# ── Model discovery (opt-in, shown in a collapsed expander) ─────────────────

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

    if "mini" in lowered:
        score -= 1
        reasons.append("Cost-efficient mini tier")

    if not reasons:
        reasons.append("General purpose reasoning model")

    return max(score, 0.0), reasons


def discover_financial_models(creds: Dict[str, Optional[str]]) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """Return OpenAI models ranked for financial analysis tasks."""

    if not (isinstance(creds, dict) and creds.get("api_key")):
        return [], "Missing API key"

    try:
        response = _client_from_creds(creds).models.list()
    except Exception as exc:
        return [], str(exc)

    models: List[Dict[str, object]] = []
    for item in getattr(response, "data", []):
        model_id = getattr(item, "id", None)
        if not model_id or _is_model_excluded(model_id):
            continue
        score, reasons = _score_model(model_id)
        models.append({"id": model_id, "score": score, "reasons": reasons})

    models.sort(key=lambda entry: (entry["score"], entry["id"]), reverse=True)
    return models, None


CURATED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini"]


def render_model_selection(ticker: str, exp, creds: Optional[Dict[str, Optional[str]]] = None):
    """Render a compact model picker, returning creds and the chosen model."""

    openai_creds = creds or resolve_openai_credentials()
    api_key = openai_creds.get("api_key") if isinstance(openai_creds, dict) else None
    if not api_key:
        st.error("OpenAI API key is not configured.")
        return openai_creds, None

    configured = CONFIG.get("openai", {}).get("model")
    options: List[str] = []
    for model_id in [configured, *CURATED_MODELS]:
        if model_id and model_id not in options:
            options.append(model_id)

    previous = st.session_state.get("ai_selected_model")
    if previous and previous not in options:
        options.append(previous)

    default_index = options.index(previous) if previous in options else 0
    chosen_model = st.selectbox("Model", options, index=default_index)

    with st.expander("Advanced: browse account models or use a custom ID"):
        custom_model = st.text_input(
            "Custom model ID",
            key="ai_custom_model",
            help="Overrides the dropdown selection when set.",
        )
        if st.button("List models available on this account"):
            models, discovery_error = discover_financial_models(openai_creds)
            if discovery_error:
                st.warning(f"Could not list models: {discovery_error}")
            st.session_state["ai_model_catalog"] = models
        catalog = st.session_state.get("ai_model_catalog")
        if catalog:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Model": entry["id"],
                            "Score": entry["score"],
                            "Highlights": ", ".join(map(str, entry["reasons"])),
                        }
                        for entry in catalog
                    ]
                ),
                use_container_width=True,
                hide_index=True,
                height=240,
            )

    custom_model = (st.session_state.get("ai_custom_model") or "").strip()
    if custom_model:
        chosen_model = custom_model

    st.session_state["ai_selected_model"] = chosen_model
    st.caption(f"Requests will use `{chosen_model}`.")
    return openai_creds, chosen_model


# ── Compact analysis payload ────────────────────────────────────────────────

def _round(value, ndigits: int = 3):
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _summarise_positioning(df: Optional[pd.DataFrame], spot: float, top_n: int = 8) -> Dict[str, Any]:
    """Reduce the per-contract chain to the strike levels a model actually needs."""

    if df is None or df.empty or "strike" not in df.columns:
        return {}

    work = df.copy()
    for col in ("GammaExposure", "DeltaExposure", "open_interest", "volume"):
        if col not in work.columns:
            work[col] = 0.0
        work[col] = work[col].fillna(0.0)

    net_gamma = work.groupby("strike")["GammaExposure"].sum().sort_index()
    summary: Dict[str, Any] = {
        "net_gamma_exposure_total_m": _round(net_gamma.sum() / 1e6),
        "net_delta_exposure_total_m": _round(work["DeltaExposure"].sum() / 1e6),
    }

    if not net_gamma.empty:
        summary["magnet_strike"] = _round(net_gamma.idxmax(), 2)
        summary["magnet_gamma_m"] = _round(net_gamma.max() / 1e6)
        summary["most_negative_gamma_strike"] = _round(net_gamma.idxmin(), 2)

        top = net_gamma.reindex(
            net_gamma.abs().sort_values(ascending=False).index[:top_n]
        ).sort_index()
        summary["net_gamma_by_strike_m"] = {
            f"{strike:g}": _round(value / 1e6) for strike, value in top.items()
        }

        nonzero = net_gamma[net_gamma != 0]
        strikes = nonzero.index.tolist()
        values = nonzero.values
        flips = [
            [_round(strikes[i - 1], 2), _round(strikes[i], 2)]
            for i in range(1, len(values))
            if values[i - 1] * values[i] < 0
        ]
        summary["gamma_flip_zones"] = flips

    if "option_type" in work.columns:
        for side in ("call", "put"):
            side_gamma = (
                work[work["option_type"] == side]
                .groupby("strike")["GammaExposure"]
                .sum()
                .abs()
            )
            if not side_gamma.empty:
                summary[f"{side}_gamma_peak"] = {
                    "strike": _round(side_gamma.idxmax(), 2),
                    "abs_gamma_m": _round(side_gamma.max() / 1e6),
                }

        oi = (
            work.pivot_table(
                index="strike", columns="option_type", values="open_interest", aggfunc="sum"
            )
            .fillna(0)
        )
        if not oi.empty:
            oi["total"] = oi.sum(axis=1)
            summary["top_open_interest_strikes"] = [
                {
                    "strike": _round(strike, 2),
                    "call_oi": int(row.get("call", 0)),
                    "put_oi": int(row.get("put", 0)),
                }
                for strike, row in oi.sort_values("total", ascending=False).head(5).iterrows()
            ]

    return summary


def _summarise_iv_skew(iv_skew_df: Optional[pd.DataFrame], spot: float) -> Dict[str, Any]:
    if iv_skew_df is None or iv_skew_df.empty:
        return {}

    skew_col = "IV Skew" if "IV Skew" in iv_skew_df.columns else "iv_skew"
    if skew_col not in iv_skew_df.columns or "strike" not in iv_skew_df.columns:
        return {}

    work = iv_skew_df.dropna(subset=["strike", skew_col])
    if work.empty:
        return {}

    atm = work.loc[(work["strike"] - spot).abs().idxmin()]
    peak = work.loc[work[skew_col].idxmax()]
    trough = work.loc[work[skew_col].idxmin()]

    summary = {
        "atm_strike": _round(atm["strike"], 2),
        "atm_skew": _round(atm[skew_col], 4),
        "peak_skew": {"strike": _round(peak["strike"], 2), "skew": _round(peak[skew_col], 4)},
        "trough_skew": {"strike": _round(trough["strike"], 2), "skew": _round(trough[skew_col], 4)},
    }
    if "iv_call" in work.columns and "iv_put" in work.columns:
        summary["atm_call_iv"] = _round(atm.get("iv_call"), 4)
        summary["atm_put_iv"] = _round(atm.get("iv_put"), 4)
    if len(work) >= 2:
        slope = np.polyfit(work["strike"].astype(float), work[skew_col].astype(float), 1)[0]
        summary["skew_slope_per_strike"] = _round(slope, 6)
    return summary


def build_analysis_payload(
    df: Optional[pd.DataFrame],
    iv_skew_df: Optional[pd.DataFrame],
    vol_ratio: float,
    oi_ratio: float,
    articles: Optional[Sequence[dict]],
    spot: float,
    offset: float,
    ticker: str,
    expirations: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """Assemble a compact, structured snapshot for the model prompt."""

    tradier_token = st.secrets.get("TRADIER_TOKEN")

    payload: Dict[str, Any] = {
        "ticker": ticker,
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spot": _round(spot, 2),
        "strike_window": {"low": _round(spot - offset, 2), "high": _round(spot + offset, 2)},
        "expirations_analysed": list(expirations or []),
        "flow": {
            "put_call_volume_ratio": _round(vol_ratio),
            "put_call_oi_ratio": _round(oi_ratio),
        },
        "positioning": _summarise_positioning(df, spot),
        "iv_skew": _summarise_iv_skew(iv_skew_df, spot),
    }

    try:
        payload["price_stats"] = {
            key: _round(value, 2) for key, value in get_price_stats(ticker, tradier_token).items()
        }
    except Exception:
        payload["price_stats"] = {}

    try:
        payload["vix"] = {key: _round(value, 2) for key, value in get_vix_info().items()}
    except Exception:
        payload["vix"] = {}

    try:
        ten = get_bond_yield_info("^TNX")
        payload["treasury_10y"] = {
            "yield_pct": _round(ten.get("spot"), 2),
            "return_1d_pct": _round(ten.get("1d_return"), 2),
        }
    except Exception:
        payload["treasury_10y"] = {}

    try:
        if df is not None and not df.empty:
            spikes = compute_unusual_spikes(df, top_n=5)
            payload["flow"]["top_vol_oi_spikes"] = json.loads(
                spikes.round(3).to_json(orient="records")
            )
    except Exception:
        pass

    headlines = []
    for art in (articles or [])[:10]:
        date = str(art.get("date", ""))[:10]
        title = art.get("title", "")
        source = art.get("source", "")
        if title:
            headlines.append(f"{date} — {title}" + (f" ({source})" if source else ""))
    payload["headlines"] = headlines

    return payload


def create_data_packet(ticker: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap the compact payload into chat messages for the model."""

    system_msg = {
        "role": "system",
        "content": (
            "You are an experienced volatility and macro strategist. You will receive a compact "
            "JSON snapshot of options dealer positioning, volatility structure, order flow, and "
            "macro context for a single ticker. Exposure figures with an _m suffix are in "
            "millions of dollars. Combine dealer positioning, volatility structure, and market "
            "context to craft actionable option trade ideas."
        ),
    }

    instructions = textwrap.dedent(
        f"""
        Analyse the market snapshot for {ticker} below and respond with these sections:

        1. **Dealer positioning** — expected hedging behaviour around the magnet and flip zones.
        2. **Market regime** — what price stats, VIX, rates, and headlines imply for risk appetite.
        3. **Trade ideas** — long calls or long puts ONLY (no spreads or multi-leg structures).
           One idea per horizon: 1-2 weeks, 1-2 months, 6-12 months. For each give strike,
           expiration, entry trigger, invalidation level, and a confidence score (1-100).
        4. **Key levels** — max-pain estimate, supports, and resistances derived from the data.

        Market snapshot JSON:
        """
    ).strip()

    user_msg = {
        "role": "user",
        "content": instructions
        + "\n```json\n"
        + json.dumps(payload, indent=2, default=str)
        + "\n```",
    }
    return {"messages": [system_msg, user_msg]}


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


def call_openai_api(data_packet, creds, model_name: str):
    if not (isinstance(creds, dict) and creds.get("api_key")):
        raise ValueError("OpenAI API key is missing; cannot send request.")

    client = _client_from_creds(creds)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=data_packet["messages"],
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None


def _build_ai_form_key(label: str, ticker: str, exp: Optional[Sequence[str]]) -> str:
    exp_fragment = "-".join(exp) if isinstance(exp, (list, tuple)) else (str(exp) if exp else "")
    return f"{label}_{ticker}_{exp_fragment}" if exp_fragment else f"{label}_{ticker}"


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

    if df_net is not None and not df_net.empty:
        df_net = df_net[(df_net["strike"] >= spot - offset) & (df_net["strike"] <= spot + offset)]

    with st.spinner("Building compact data packet…"):
        payload = build_analysis_payload(
            df_net, iv_skew_df, vol_ratio, oi_ratio, articles, spot, offset, ticker, exp
        )
    data_packet = create_data_packet(ticker, payload)

    with st.expander("Data packet (exactly what the model will see)"):
        st.json(payload)

    tokens = estimate_token_count(data_packet, selected_model)

    form_key = _build_ai_form_key("ai_confirmation", ticker, exp)
    with st.form(form_key):
        proceed = st.checkbox("I reviewed the packet and want to send the request.")
        pin_entry = st.text_input("Enter security PIN", type="password")
        submitted = st.form_submit_button("Send to OpenAI", use_container_width=True)

    if not submitted:
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

    with st.spinner(f"Asking `{selected_model}`…"):
        try:
            analysis = call_openai_api(data_packet, openai_creds, selected_model)
        except ValueError as exc:
            st.error(str(exc))
            return

    if analysis:
        st.markdown(f"### AI Trade Analysis\n{analysis}")
        save_analysis(
            ticker=ticker,
            expirations=exp,
            payload=payload,
            response=analysis,
            token_count=tokens or 0,
        )
        st.success("✅ Analysis saved.")
        st.session_state.want_ai = False


def main() -> int:
    """Console entrypoint placeholder for packaged installs."""

    print("Run the dashboard with: streamlit run app.py")
    return 0
