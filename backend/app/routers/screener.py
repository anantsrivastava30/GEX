"""Snapshot-only options screener endpoints."""

from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas import (
    CustomScreenerRequest,
    CustomScreenerResponse,
    ScreenerFieldsResponse,
    ScreenerResponse,
)
from backend.app.services_screener import (
    get_screener,
    get_screener_field_specs,
    run_custom_screener,
)

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


@router.get("/fields", response_model=ScreenerFieldsResponse)
def screener_fields():
    return get_screener_field_specs()


@router.post("/query", response_model=CustomScreenerResponse)
def custom_screener(request: CustomScreenerRequest):
    try:
        return run_custom_screener(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
