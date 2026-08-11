"""Native earnings and economic calendar endpoint."""

from datetime import date, timedelta
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas import CalendarResponse
from backend.app.services_calendar import get_calendar

router = APIRouter(prefix="/api/calendar", tags=["calendar"])


@router.get("", response_model=CalendarResponse)
def calendar(
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    symbols: Optional[List[str]] = Query(None, max_length=25),
    scope: Literal["focus", "large", "all"] = Query("all"),
):
    start_date = start or date.today()
    end_date = end or (start_date + timedelta(days=14))
    if end_date < start_date:
        raise HTTPException(status_code=422, detail="end must be on or after start")
    if (end_date - start_date).days > 90:
        raise HTTPException(status_code=422, detail="calendar range cannot exceed 90 days")
    return get_calendar(start_date, end_date, symbols, scope)
