from typing import List

from fastapi import APIRouter, Query

from backend.app import services
from backend.app.schemas import CachedFlowResponse, HottestChainsResponse, UnusualResponse

router = APIRouter(prefix="/api/flow", tags=["flow"])


@router.get("/unusual", response_model=UnusualResponse)
def unusual_activity(
    symbol: str = Query(...),
    expirations: List[str] = Query(...),
    top_n: int = Query(10, ge=1, le=100),
):
    """Vol/OI anomaly ranking — the free-data proxy for a flow feed."""

    return services.get_unusual(symbol.upper(), expirations, top_n)


@router.get("/feed", response_model=CachedFlowResponse)
def cached_flow_feed(limit: int = Query(100, ge=1, le=500)):
    """Cached volume/OI/IV anomaly proxy. This endpoint never scans Tradier."""

    return services.get_cached_flow_feed(limit)


@router.get("/hottest-chains", response_model=HottestChainsResponse)
def hottest_chains(limit: int = Query(50, ge=1, le=200)):
    """Cached chain rankings. This endpoint never scans Tradier."""

    return services.get_hottest_chains(limit)
