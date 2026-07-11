from typing import List, Optional

from fastapi import APIRouter, Header, Query

from backend.app import services_ai
from backend.app.schemas import (
    AIAnalyzeRequest,
    AIAnalyzeResponse,
    AIHistoryItem,
    AIStatus,
)

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/status", response_model=AIStatus)
def ai_status():
    return services_ai.get_ai_status()


@router.get("/analyses", response_model=List[AIHistoryItem])
def ai_history(
    limit: int = Query(15, ge=1, le=100),
    pin: Optional[str] = Header(None, alias="X-AI-PIN", max_length=128),
):
    return services_ai.get_ai_history(limit, pin)


@router.post("/analyze", response_model=AIAnalyzeResponse)
def ai_analyze(req: AIAnalyzeRequest):
    """Build the positioning snapshot for a ticker and get a trade narrative.

    Costs real OpenAI spend; disabled unless AI_PIN is configured.
    """

    return services_ai.run_ai_analysis(
        req.symbol, req.expirations, req.model, req.offset, req.pin
    )
