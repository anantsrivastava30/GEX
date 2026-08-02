"""Endpoints backed by our own accumulated snapshot history."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from backend.app import services
from backend.app.schemas import (
    GammaGapHistoryRow,
    GammaGapOutcomesResponse,
    IVRankResponse,
    OIChangeRow,
)

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/gamma-gap", response_model=List[GammaGapHistoryRow])
def gamma_gap_history(
    ticker: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Logged gamma-gap scans (the raw material for the track-record page)."""

    return services.get_gamma_gap_history(ticker, limit)


@router.get("/gamma-gap/outcomes", response_model=GammaGapOutcomesResponse)
def gamma_gap_outcomes(
    ticker: Optional[str] = Query(None),
    horizon: int = Query(5, ge=1, le=20, description="Sessions after the signal"),
    limit: int = Query(250, ge=1, le=1000),
):
    """Realized outcomes: did the magnet trade within the horizon?

    Signal-day bars never count; recent signals without a full horizon stay
    pending and are excluded from the hit rate.
    """

    return services.get_gamma_gap_outcomes(ticker, horizon, limit)


@router.get("/{symbol}/iv-rank", response_model=IVRankResponse)
def iv_rank(symbol: str):
    return services.get_iv_rank(symbol.upper())


@router.get("/{symbol}/oi-change", response_model=List[OIChangeRow])
def oi_change(symbol: str, limit: int = Query(50, ge=1, le=500)):
    return services.get_oi_change(symbol.upper(), limit)


@router.get("/metrics", response_model=List[Dict[str, Any]])
def daily_metrics(symbol: Optional[str] = Query(None)):
    return services.get_daily_metrics(symbol.upper() if symbol else None)
