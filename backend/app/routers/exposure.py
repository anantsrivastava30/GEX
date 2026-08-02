from typing import List

from fastapi import APIRouter, Query

from backend.app import services
from backend.app.schemas import ExposureResponse

router = APIRouter(prefix="/api/ticker", tags=["exposure"])


@router.get("/{symbol}/exposure", response_model=ExposureResponse)
def vanna_charm_exposure(
    symbol: str,
    expirations: List[str] = Query(..., description="One or more expirations"),
    offset: int = Query(35, ge=1, le=500, description="Strike range around spot"),
):
    """Net vanna/charm exposure by strike, derived locally via Black-Scholes.

    Tradier greeks do not include vanna/charm; these come from
    ``quant_analysis.analytics.greeks`` using mid_iv (smv_vol fallback).
    """

    return services.get_exposure(symbol.upper(), expirations, offset)
