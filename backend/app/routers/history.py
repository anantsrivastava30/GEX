"""Endpoints backed by our own accumulated snapshot history."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from backend.app import services
from backend.app.schemas import IVRankResponse, OIChangeRow

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/{symbol}/iv-rank", response_model=IVRankResponse)
def iv_rank(symbol: str):
    return services.get_iv_rank(symbol.upper())


@router.get("/{symbol}/oi-change", response_model=List[OIChangeRow])
def oi_change(symbol: str, limit: int = Query(50, ge=1, le=500)):
    return services.get_oi_change(symbol.upper(), limit)


@router.get("/metrics", response_model=List[Dict[str, Any]])
def daily_metrics(symbol: Optional[str] = Query(None)):
    return services.get_daily_metrics(symbol.upper() if symbol else None)
