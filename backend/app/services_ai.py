"""AI analysis service: the Streamlit `render_ai_tab` pipeline, headless.

Reuses the compact-payload builder and prompt packet from
``quant_analysis.services.ai_analysis`` but reads credentials from the
environment and calls OpenAI directly (no ``st.*`` UI side effects).
Spend is gated by the ``AI_PIN`` setting, mirroring the legacy app's PIN.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from quant_analysis.services.ai_analysis import (
    _client_from_creds,
    build_analysis_payload,
    create_data_packet,
    estimate_token_count,
    resolve_openai_credentials,
)
from quant_analysis.services.market_data import compute_iv_skew, compute_put_call_ratios
from quant_analysis.storage.db import load_analyses, save_analysis

from backend.app import services
from backend.app.config import get_settings

logger = logging.getLogger(__name__)


def verify_ai_pin(pin: Optional[str]) -> None:
    settings = get_settings()
    if not settings.ai_pin:
        raise HTTPException(
            status_code=503,
            detail="AI analysis is disabled until AI_PIN is configured.",
        )
    if not pin or not secrets.compare_digest(pin, settings.ai_pin):
        raise HTTPException(status_code=401, detail="Invalid or missing AI PIN.")


def get_ai_status() -> Dict[str, Any]:
    settings = get_settings()
    creds = resolve_openai_credentials()
    return {
        "openai_configured": bool(creds.get("api_key")),
        "pin_required": bool(settings.ai_pin),
        "default_model": settings.openai_model,
    }


def get_ai_history(limit: int, pin: Optional[str]) -> List[Dict[str, Any]]:
    verify_ai_pin(pin)
    rows = load_analyses(limit=limit)
    # Drop the bulky stored payload; the list view only needs the narrative.
    return [{k: v for k, v in row.items() if k != "payload"} for row in rows]


def _build_symbol_packet(
    symbol: str,
    expirations: Optional[List[str]],
    offset: int,
    model: str,
) -> tuple[str, List[str], Dict[str, Any], Dict[str, Any], Optional[int]]:
    """Fetch positioning data and build the compact payload + prompt packet.

    Shared by the paid analyze endpoint and the free payload-only endpoint;
    involves no OpenAI call.
    """

    settings = get_settings()
    symbol = symbol.upper()
    exps = (expirations or services.get_expirations(symbol)[:4])[:8]
    if not exps:
        raise HTTPException(status_code=404, detail=f"No expirations for {symbol}.")

    spot = services._spot(symbol)
    df = services._positioning_frame(symbol, exps)
    work = df[(df["strike"] >= spot - offset) & (df["strike"] <= spot + offset)]

    try:
        iv_skew_df = compute_iv_skew(services._chain(symbol, exps[0]))
    except Exception:
        iv_skew_df = None

    vol_ratio, oi_ratio = compute_put_call_ratios(df)

    try:
        articles = services.get_news()
    except Exception:
        articles = []

    payload = build_analysis_payload(
        work,
        iv_skew_df,
        float(vol_ratio),
        float(oi_ratio),
        articles,
        spot,
        offset,
        symbol,
        exps,
        tradier_token=settings.tradier_token,
    )
    packet = create_data_packet(symbol, payload)
    tokens = estimate_token_count(packet, model)
    return symbol, list(exps), payload, packet, tokens


def get_ai_payload(
    symbol: str,
    expirations: Optional[List[str]],
    offset: int,
) -> Dict[str, Any]:
    """Payload-only mode: the analysis packet without any OpenAI call.

    No PIN and no spend, so the tool stays usable when no agent or API key
    is available - paste the messages into any LLM, or feed the payload to
    an external agent.
    """

    settings = get_settings()
    symbol, exps, payload, packet, tokens = _build_symbol_packet(
        symbol, expirations, offset, settings.openai_model
    )
    return {
        "symbol": symbol,
        "expirations": exps,
        "prompt_tokens": tokens,
        "payload": payload,
        "messages": packet["messages"],
    }


def run_ai_analysis(
    symbol: str,
    expirations: Optional[List[str]],
    model: Optional[str],
    offset: int,
    pin: Optional[str],
) -> Dict[str, Any]:
    settings = get_settings()
    verify_ai_pin(pin)

    creds = resolve_openai_credentials()
    if not creds.get("api_key"):
        raise HTTPException(
            status_code=503, detail="OPENAI_API_KEY is not configured on the server."
        )

    if model and model != settings.openai_model:
        raise HTTPException(status_code=400, detail="Requested model is not allowed.")
    model = settings.openai_model
    symbol, exps, payload, packet, tokens = _build_symbol_packet(
        symbol, expirations, offset, model
    )

    client = _client_from_creds(creds)
    try:
        completion = client.chat.completions.create(
            model=model, messages=packet["messages"]
        )
        response = completion.choices[0].message.content
    except Exception as exc:
        logger.exception("OpenAI analysis failed for %s", symbol)
        raise HTTPException(
            status_code=502, detail="OpenAI analysis request failed."
        ) from exc

    if not response:
        raise HTTPException(status_code=502, detail="OpenAI returned an empty response.")

    try:
        save_analysis(
            ticker=symbol,
            expirations=list(exps),
            payload=payload,
            response=response,
            token_count=tokens or 0,
        )
    except Exception:
        # Persistence must never eat a paid completion.
        pass

    return {
        "symbol": symbol,
        "model": model,
        "response": response,
        "prompt_tokens": tokens,
        "payload": payload,
    }
