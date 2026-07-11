from typing import List

from fastapi import APIRouter, Query

from backend.app import services
from backend.app.schemas import (
    Candle,
    DeltaProjectionResponse,
    GexProfile,
    MaxPainResponse,
    RatiosResponse,
    SkewResponse,
    TermStructureResponse,
    TickerSnapshot,
)

router = APIRouter(prefix="/api/ticker", tags=["ticker"])


@router.get("/{symbol}/snapshot", response_model=TickerSnapshot)
def ticker_snapshot(symbol: str):
    return services.get_ticker_snapshot(symbol.upper())


@router.get("/{symbol}/expirations", response_model=List[str])
def expirations(symbol: str):
    return services.get_expirations(symbol.upper())


@router.get("/{symbol}/gex", response_model=GexProfile)
def gex_profile(
    symbol: str,
    expiration: str = Query(..., description="Expiration date YYYY-MM-DD"),
    offset: int = Query(35, ge=1, le=500, description="Strike range around spot"),
):
    return services.get_gex_profile(symbol.upper(), expiration, offset)


@router.get("/{symbol}/candles", response_model=List[Candle])
def candles(
    symbol: str,
    days: int = Query(120, ge=5, le=365, description="Lookback window in days"),
):
    return services.get_candles(symbol.upper(), days)


@router.get("/{symbol}/skew", response_model=SkewResponse)
def iv_skew(symbol: str, expiration: str = Query(...)):
    return services.get_skew(symbol.upper(), expiration)


@router.get("/{symbol}/ratios", response_model=RatiosResponse)
def put_call_ratios(
    symbol: str,
    expirations: List[str] = Query(..., description="One or more expirations"),
):
    return services.get_ratios(symbol.upper(), expirations)


@router.get("/{symbol}/delta-projection", response_model=DeltaProjectionResponse)
def delta_projection(
    symbol: str,
    expiration: str = Query(..., description="Expiration date YYYY-MM-DD"),
    offset: int = Query(35, ge=1, le=500, description="Strike range around spot"),
):
    """Intraday net dealer delta exposure along today's price path."""

    return services.get_delta_projection(symbol.upper(), expiration, offset)


@router.get("/{symbol}/max-pain", response_model=MaxPainResponse)
def max_pain(symbol: str, expiration: str = Query(...)):
    return services.get_max_pain(symbol.upper(), expiration)


@router.get("/{symbol}/term-structure", response_model=TermStructureResponse)
def term_structure(
    symbol: str,
    expirations: int = Query(6, ge=2, le=12, description="Expirations to include"),
):
    return services.get_term_structure(symbol.upper(), expirations)
