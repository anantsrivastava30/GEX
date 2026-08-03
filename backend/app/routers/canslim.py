"""CAN SLIM stock leaders: the buy-flag scan gated by market direction."""

from fastapi import APIRouter, HTTPException

from backend.app import services_canslim
from backend.app.schemas import LeadersResponse, StockLeader

router = APIRouter(prefix="/api/canslim", tags=["canslim"])


@router.get("", response_model=LeadersResponse)
def leaders():
    """Ranked seven-letter evaluation for every stock in the scan universe."""

    return services_canslim.get_leaders()


@router.get("/{symbol}", response_model=StockLeader)
def stock_detail(symbol: str):
    """Full letter table and narrative for one stock."""

    detail = services_canslim.get_stock_detail(symbol)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol is excluded (ETF), outside the scan universe, or has no history",
        )
    return detail
