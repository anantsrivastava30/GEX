from typing import Optional

from fastapi import APIRouter, Query

from backend.app import services_congress
from backend.app.schemas import CongressTradesResponse

router = APIRouter(prefix="/api/congress", tags=["congress"])


@router.get("/trades", response_model=CongressTradesResponse)
def congress_trades(
    ticker: Optional[str] = Query(default=None, max_length=12),
    chamber: Optional[str] = Query(default=None, pattern=r"^(house|senate)$"),
    days: Optional[int] = Query(default=None, ge=1, le=3650),
    limit: int = Query(default=200, ge=1, le=1000),
) -> CongressTradesResponse:
    """Filterable congressional stock-trade disclosures (free public mirrors)."""
    return CongressTradesResponse(
        **services_congress.get_congress_trades(
            ticker=ticker, chamber=chamber, days=days, limit=limit
        )
    )
