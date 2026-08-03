"""FastAPI application factory.

Run locally with:
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.routers import (
    admin,
    ai,
    alerts,
    binomial,
    calendar,
    congress,
    direction,
    exposure,
    flow,
    history,
    market,
    news,
    screener,
    ticker,
    watchlists,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    from backend.app.storage import ensure_default_watchlist, init_app_storage

    init_app_storage()
    ensure_default_watchlist(settings.snapshot_tickers)
    scheduler = None
    if settings.scheduler_enabled:
        from backend.app.jobs import create_scheduler
        from backend.app.services_watchlists import snapshot_symbols

        scheduler = create_scheduler()
        scheduler.start()
        logger.info(
            "Scheduler started: intraday snapshots for %s",
            ", ".join(snapshot_symbols()),
        )
    try:
        yield
    finally:
        if scheduler:
            scheduler.shutdown(wait=False)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="GEX Options Analytics API",
        description="Dealer-positioning analytics: GEX, vanna/charm, gamma-gap, "
        "skew, max pain, unusual activity, and own-data market history.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ticker.router)
    app.include_router(binomial.router)
    app.include_router(exposure.router)
    app.include_router(market.router)
    app.include_router(direction.router)
    app.include_router(news.router)
    app.include_router(flow.router)
    app.include_router(screener.router)
    app.include_router(calendar.router)
    app.include_router(history.router)
    app.include_router(congress.router)
    app.include_router(watchlists.router)
    app.include_router(alerts.router)
    app.include_router(ai.router)
    app.include_router(admin.router)

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        from pathlib import Path

        from quant_analysis.storage.snapshots import (
            are_consecutive_market_sessions,
            last_trading_date,
            load_daily_metrics,
        )

        metrics = load_daily_metrics(Path(settings.snapshot_dir))
        dates = (
            sorted(metrics["date"].astype(str).unique())
            if not metrics.empty and "date" in metrics
            else []
        )
        return {
            "status": "ok",
            "tradier_configured": bool(settings.tradier_token),
            "scheduler_enabled": settings.scheduler_enabled,
            "snapshot_latest": dates[-1] if dates else None,
            "snapshot_history_days": len(dates),
            "snapshot_history_ready": len(dates) >= 2
            and dates[-1] == last_trading_date()
            and are_consecutive_market_sessions(dates[-2], dates[-1]),
        }

    return app


app = create_app()
