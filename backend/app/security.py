"""Temporary shared-workspace mutation gate until Phase 3 authentication."""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Header, HTTPException

from backend.app.config import get_settings


def require_workspace_pin(
    pin: Optional[str] = Header(None, alias="X-Workspace-PIN", max_length=128),
) -> None:
    settings = get_settings()
    expected = settings.workspace_pin or settings.ai_pin
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Workspace mutations are disabled until WORKSPACE_PIN or AI_PIN is configured.",
        )
    if not pin or not secrets.compare_digest(pin, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing workspace PIN.")
