from typing import List

from fastapi import APIRouter, Query

from backend.app import services
from backend.app.schemas import UnusualResponse

router = APIRouter(prefix="/api/flow", tags=["flow"])


@router.get("/unusual", response_model=UnusualResponse)
def unusual_activity(
    symbol: str = Query(...),
    expirations: List[str] = Query(...),
    top_n: int = Query(10, ge=1, le=100),
):
    """Vol/OI anomaly ranking — the free-data proxy for a flow feed."""

    return services.get_unusual(symbol.upper(), expirations, top_n)
