"""Shared dependencies: Tradier client with retry + rate limiting."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from fastapi import HTTPException
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from quant_analysis.integrations.tradier import TradierAPI, TradierAPIError

from backend.app.config import get_settings
from backend.app.ratelimit import TokenBucket


@lru_cache
def get_bucket() -> TokenBucket:
    return TokenBucket(get_settings().tradier_requests_per_minute)


@lru_cache
def get_tradier() -> TradierAPI:
    settings = get_settings()
    if not settings.tradier_token:
        raise HTTPException(
            status_code=503,
            detail="TRADIER_TOKEN is not configured on the server.",
        )
    return TradierAPI(settings.tradier_token, settings.tradier_api_url)


def _is_retryable(exc: BaseException) -> bool:
    return (
        isinstance(exc, TradierAPIError)
        and (exc.status_code is None or exc.status_code >= 500)
    )


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)
def _call_with_retry(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return fn(*args, **kwargs)


def tradier_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one Tradier client call under the shared rate limit with retry."""

    get_bucket().acquire_or_raise()
    try:
        return _call_with_retry(fn, *args, **kwargs)
    except TradierAPIError as exc:
        status = exc.status_code or 502
        raise HTTPException(
            status_code=502 if status >= 500 else status,
            detail=str(exc),
        ) from exc
