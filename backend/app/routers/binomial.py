from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Query

from backend.app import services
from backend.app.schemas import BinomialTreeResponse

router = APIRouter(prefix="/api/ticker", tags=["binomial"])


@router.get("/{symbol}/binomial-tree", response_model=BinomialTreeResponse)
def binomial_tree(
    symbol: str,
    expiration: date = Query(..., description="Contract expiration YYYY-MM-DD"),
    strike: float = Query(..., gt=0, description="Option strike"),
    option_type: Literal["call", "put"] = Query(..., description="Option payoff type"),
    steps: int = Query(8, ge=2, le=50, description="CRR tree steps"),
    days_to_exp: int = Query(..., ge=1, le=3650, description="Pricing horizon in days"),
    rate_override: Optional[float] = Query(
        None,
        ge=-0.25,
        le=1.0,
        description="Annualized decimal risk-free rate override",
    ),
    iv_override: Optional[float] = Query(
        None,
        gt=0,
        le=5.0,
        description="Annualized decimal implied-volatility override",
    ),
):
    """Return a client-renderable CRR binomial tree and its calibration."""

    return services.get_binomial_tree(
        symbol.upper(),
        expiration.isoformat(),
        strike,
        option_type,
        steps,
        days_to_exp,
        rate_override,
        iv_override,
    )
