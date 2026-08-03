"""O'Neil market-direction endpoints: index states, detail, signal feed."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app import services_direction
from backend.app.schemas import (
    DirectionDetail,
    DirectionOutcomesResponse,
    DirectionOverview,
    DirectionSignalsResponse,
)

router = APIRouter(prefix="/api/direction", tags=["direction"])


@router.get("", response_model=DirectionOverview)
def direction_overview():
    """States, scorecards, breadth, and bottom durability for all tracked indices."""

    return services_direction.get_direction_overview()


@router.get("/signals", response_model=DirectionSignalsResponse)
def direction_signals(
    limit: int = Query(50, ge=1, le=500),
    symbol: Optional[str] = Query(None, max_length=12),
):
    """Persisted signal log: follow-throughs, rally days, failures, EMA touches."""

    return services_direction.list_direction_signals(limit, symbol)


@router.get("/outcomes", response_model=DirectionOutcomesResponse)
def direction_outcomes(
    signal_type: Optional[str] = Query(
        None, pattern="^(follow_through_day|stock_breakout)$"
    ),
    horizon: int = Query(25, ge=5, le=120, description="Sessions after the signal"),
    limit: int = Query(200, ge=1, le=500),
):
    """Did logged follow-throughs and breakouts hold? Rates plus excursions.

    Failure is measured against each signal's own invalidation level. The
    signal day never counts and signals without a full horizon stay pending.
    """

    return services_direction.get_signal_outcomes(signal_type, horizon, limit)


@router.get("/{symbol}", response_model=DirectionDetail)
def direction_detail(symbol: str):
    """Annotated candles, EMA series, and the full read for one tracked index."""

    detail = services_direction.get_direction_detail(symbol)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol is not in the tracked index universe or has no history",
        )
    return detail
