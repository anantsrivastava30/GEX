"""Shared-workspace watchlist validation and persistence."""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List

from backend.app import storage
from backend.app.config import get_settings

_MAX_WATCHLISTS = 20
_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def configured_symbols() -> List[str]:
    return list(
        dict.fromkeys(symbol.strip().upper() for symbol in get_settings().snapshot_tickers)
    )


def snapshot_symbols(exclude_watchlist_id: int | None = None) -> List[str]:
    symbols = configured_symbols()
    for watchlist in storage.list_watchlists():
        if watchlist["id"] == exclude_watchlist_id:
            continue
        symbols.extend(watchlist["symbols"])
    return list(dict.fromkeys(symbols))


def normalise_symbols(symbols: List[str]) -> List[str]:
    selected = list(
        dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip())
    )
    if not selected:
        raise ValueError("A watchlist requires at least one symbol")
    invalid = [symbol for symbol in selected if not _SYMBOL_PATTERN.fullmatch(symbol)]
    if invalid:
        raise ValueError(
            "Symbols must use 1-10 letters, numbers, dots, or hyphens: "
            + ", ".join(invalid)
        )
    return selected


def _enforce_snapshot_capacity(
    symbols: List[str], exclude_watchlist_id: int | None = None
) -> None:
    universe = list(dict.fromkeys(snapshot_symbols(exclude_watchlist_id) + symbols))
    limit = get_settings().snapshot_max_symbols
    if len(universe) > limit:
        raise ValueError(
            f"Snapshot capacity is {limit} unique symbols; this change would schedule {len(universe)}"
        )


def list_watchlists() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "workspace": "shared",
        "available_symbols": snapshot_symbols(),
        "snapshot_symbol_limit": settings.snapshot_max_symbols,
        "scheduler_enabled": settings.scheduler_enabled,
        "tradier_configured": bool(settings.tradier_token),
        "items": storage.list_watchlists(),
    }


def create_watchlist(name: str, symbols: List[str]) -> Dict[str, Any]:
    if len(storage.list_watchlists()) >= _MAX_WATCHLISTS:
        raise ValueError(f"A maximum of {_MAX_WATCHLISTS} watchlists is allowed")
    selected = normalise_symbols(symbols)
    _enforce_snapshot_capacity(selected)
    try:
        return storage.create_watchlist(name.strip(), selected)
    except sqlite3.IntegrityError as exc:
        raise ValueError("A watchlist with that name already exists") from exc


def update_watchlist(watchlist_id: int, name: str, symbols: List[str]) -> Dict[str, Any]:
    selected = normalise_symbols(symbols)
    _enforce_snapshot_capacity(selected, exclude_watchlist_id=watchlist_id)
    try:
        item = storage.update_watchlist(
            watchlist_id, name.strip(), selected
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError("A watchlist with that name already exists") from exc
    if item is None:
        raise KeyError(watchlist_id)
    return item


def delete_watchlist(watchlist_id: int) -> None:
    if storage.watchlist_in_use(watchlist_id):
        raise RuntimeError("Delete or move alert rules using this watchlist first")
    if not storage.delete_watchlist(watchlist_id):
        raise KeyError(watchlist_id)
