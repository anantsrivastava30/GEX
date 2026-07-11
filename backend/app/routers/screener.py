"""Snapshot-only options screener endpoints."""

from typing import List, Literal, Optional

from fastapi import APIRouter, Query

from backend.app.schemas import ScreenerResponse
from backend.app.services_screener import get_screener

router = APIRouter(prefix="/api/screener", tags=["screener"])


@router.get("", response_model=ScreenerResponse)
def screener(
    preset: Literal["high_vol_oi", "unusually_bullish", "gamma_squeeze"] = Query(
        "high_vol_oi"
    ),
    symbols: Optional[List[str]] = Query(None, max_length=25),
    min_vol_oi: Optional[float] = Query(None, ge=0, le=1000),
    min_open_interest: Optional[float] = Query(None, ge=0, le=1_000_000_000),
    limit: int = Query(50, ge=1, le=200),
):
    """Screen configured persisted snapshots, never live option flow."""

    return get_screener(preset, symbols, min_vol_oi, min_open_interest, limit)
