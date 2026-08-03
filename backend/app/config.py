"""Backend settings.

Environment variables take precedence (TRADIER_TOKEN, SNAPSHOT_DIR, …);
shared defaults come from the root ``config.yaml`` via ``quant_analysis.config``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
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
    snapshot_max_symbols: int = Field(
        default=int(CONFIG.get("snapshots", {}).get("max_symbols", 50)),
        ge=1,
        le=500,
    )
    snapshot_refresh_pause_seconds: float = Field(
        default=float(CONFIG.get("snapshots", {}).get("refresh_pause_seconds", 1.0)),
        ge=0.0,
        le=60.0,
    )

    # AI analysis (Phase 1 port of the Streamlit AI tab).
    openai_model: str = CONFIG.get("openai", {}).get("model", "gpt-4o-mini")
    ai_pin: Optional[str] = None
    workspace_pin: Optional[str] = None

    # Optional Financial Modeling Prep key for the congress-trades feed. When
    # absent the feed falls back to the config.yaml JSON mirrors.
    fmp_api_key: Optional[str] = None

    # Native calendar providers. Earnings work without credentials; FRED
    # release dates require a free API key.
    fred_api_key: Optional[str] = None
    fred_api_url: str = CONFIG.get("fred", {}).get(
        "api_url", "https://api.stlouisfed.org/fred"
    )

    # Shared single-workspace state. Phase 3 auth will add ownership; until
    # then watchlists, rules, and events are intentionally server-global.
    app_db_path: str = "data/gex_app.db"
    alert_discord_webhook_url: Optional[str] = None
    alert_email_to: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: Optional[str] = None
    smtp_starttls: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
