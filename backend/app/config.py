"""Backend settings.

Environment variables take precedence (TRADIER_TOKEN, SNAPSHOT_DIR, …);
shared defaults come from the root ``config.yaml`` via ``quant_analysis.config``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from quant_analysis.config import CONFIG
from quant_analysis.storage.snapshots import DEFAULT_BASE_DIR, DEFAULT_TICKERS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tradier_token: Optional[str] = None
    tradier_api_url: str = CONFIG.get("tradier", {}).get(
        "api_url", "https://api.tradier.com/v1"
    )
    snapshot_dir: str = str(DEFAULT_BASE_DIR)
    cors_origins: List[str] = ["http://localhost:3000"]

    # Tradier market-data plans allow ~120 requests/min; keep headroom for
    # background jobs.
    tradier_requests_per_minute: int = 100

    # Background jobs (P1.2). Off by default so tests and one-off local runs
    # never fire Tradier calls; the self-host compose stack turns it on.
    scheduler_enabled: bool = False
    snapshot_tickers: List[str] = list(DEFAULT_TICKERS)

    # AI analysis (Phase 1 port of the Streamlit AI tab).
    openai_model: str = CONFIG.get("openai", {}).get("model", "gpt-4o-mini")
    ai_pin: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
