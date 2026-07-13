"""Admin endpoints: on-demand snapshot capture.

PIN-gated with the same AI_PIN used for paid AI calls, since both spend
scarce resources (Tradier rate budget here, OpenAI dollars there).
"""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from backend.app.config import get_settings
from backend.app.services_ai import verify_ai_pin
from backend.app.schemas import CaptureResponse

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/capture", response_model=CaptureResponse)
def capture_snapshots(
    pin: Optional[str] = Header(None, alias="X-AI-PIN", max_length=128),
):
    """Capture the configured snapshot universe now.

    Works off-hours too: closed-market chains carry the last session's
    OI/volume and key to the last trading date. Synchronous; a full
    universe refresh takes on the order of a minute.
    """

    verify_ai_pin(pin)
    settings = get_settings()
    if not settings.tradier_token:
        raise HTTPException(
            status_code=503, detail="TRADIER_TOKEN is not configured on the server."
        )

    from backend.app.jobs import capture_universe

    return capture_universe()
