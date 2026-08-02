"""Shared-workspace watchlist validation and persistence."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List

from backend.app import storage
from backend.app.config import get_settings

_MAX_WATCHLISTS = 20


def available_symbols() -> List[str]:
    return list(
        dict.fromkeys(symbol.strip().upper() for symbol in get_settings().snapshot_tickers)
    )


def normalise_symbols(symbols: List[str]) -> List[str]:
    selected = list(
        dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip())
    )
    unsupported = [symbol for symbol in selected if symbol not in available_symbols()]
    if unsupported:
        raise ValueError(
            "Unsupported snapshot symbols: " + ", ".join(unsupported)
        )
    if not selected:
        raise ValueError("A watchlist requires at least one symbol")
    return selected


def list_watchlists() -> Dict[str, Any]:
    return {
        "workspace": "shared",
        "available_symbols": available_symbols(),
        "items": storage.list_watchlists(),
    }


def create_watchlist(name: str, symbols: List[str]) -> Dict[str, Any]:
    if len(storage.list_watchlists()) >= _MAX_WATCHLISTS:
        raise ValueError(f"A maximum of {_MAX_WATCHLISTS} watchlists is allowed")
    try:
        return storage.create_watchlist(name.strip(), normalise_symbols(symbols))
    except sqlite3.IntegrityError as exc:
        raise ValueError("A watchlist with that name already exists") from exc


def update_watchlist(watchlist_id: int, name: str, symbols: List[str]) -> Dict[str, Any]:
    try:
        item = storage.update_watchlist(
            watchlist_id, name.strip(), normalise_symbols(symbols)
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
