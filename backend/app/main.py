"""FastAPI application factory.

Run locally with:
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.routers import flow, history, market, news, ticker


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="GEX Options Analytics API",
        description="Dealer-positioning analytics: GEX, gamma-gap, skew, "
        "unusual activity, and own-data market history.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ticker.router)
    app.include_router(market.router)
    app.include_router(news.router)
    app.include_router(flow.router)
    app.include_router(history.router)

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {
            "status": "ok",
            "tradier_configured": bool(settings.tradier_token),
        }

    return app


app = create_app()
