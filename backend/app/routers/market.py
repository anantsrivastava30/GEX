from fastapi import APIRouter

from backend.app import services
from backend.app.schemas import MarketOverview

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/overview", response_model=MarketOverview)
def market_overview():
    return services.get_market_overview()
