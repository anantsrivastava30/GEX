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
    """Capture the configured and shared-watchlist snapshot universe now.

    Runs only after the current US market session has started. Off-session
    captures are rejected because the provider can roll its expiration
    universe and overwrite the previous session with non-comparable data.
    Synchronous; a full universe refresh can take several minutes.
    """

    verify_ai_pin(pin)
    settings = get_settings()
    if not settings.tradier_token:
        raise HTTPException(
            status_code=503, detail="TRADIER_TOKEN is not configured on the server."
        )

    from quant_analysis.storage.snapshots import capture_session_date

    if capture_session_date() is None:
        raise HTTPException(
            status_code=409,
            detail="Snapshot capture is unavailable before the market session or on closed days.",
        )

    from backend.app.jobs import capture_universe

    return capture_universe()
