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
from backend.app.routers import ai, binomial, exposure, flow, history, market, news, screener, ticker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    scheduler = None
    if settings.scheduler_enabled:
        from backend.app.jobs import create_scheduler

        scheduler = create_scheduler()
        scheduler.start()
        logger.info(
            "Scheduler started: intraday snapshots for %s",
            ", ".join(settings.snapshot_tickers),
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
    app.include_router(news.router)
    app.include_router(flow.router)
    app.include_router(screener.router)
    app.include_router(history.router)
    app.include_router(ai.router)

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {
            "status": "ok",
            "tradier_configured": bool(settings.tradier_token),
            "scheduler_enabled": settings.scheduler_enabled,
        }

    return app


app = create_app()
