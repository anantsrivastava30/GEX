"""Data access helpers for storing AI analyses.

The module attempts to persist data to Supabase when credentials are
available. If Supabase cannot be reached (for example when developing
offline), we transparently fall back to a local SQLite database so that
the rest of the application can continue to function.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from supabase import create_client

DB_FILE = Path("ai_analysis.db")


logger = logging.getLogger(__name__)


_supabase_client = None


def _get_secret(name: str) -> Any:
    try:
        return st.secrets[name]
    except Exception:
        return None


try:
    supabase_url = _get_secret("SUPABASE_URL")
    supabase_key = _get_secret("SUPABASE_KEY")
    if supabase_url and supabase_key:
        _supabase_client = create_client(supabase_url, supabase_key)
    else:
        logger.info("Supabase credentials not provided; using SQLite fallback")
except Exception as exc:  # pragma: no cover - defensive logging
    logger.warning(
        "Failed to initialise Supabase client; using SQLite fallback",
        exc_info=exc,
    )
    _supabase_client = None


def init_db() -> None:
    """Ensure the local SQLite table exists and has the latest schema."""

    con = sqlite3.connect(DB_FILE)
    con.execute(
        """
      CREATE TABLE IF NOT EXISTS ai_analysis (
         id            INTEGER PRIMARY KEY AUTOINCREMENT,
         ts            TEXT NOT NULL,
         ticker        TEXT NOT NULL,
         expirations   TEXT,
         payload       TEXT,
         response      TEXT,
         token_count   TEXT
      )
    """
    )
    # Add newer columns for existing installations.
    try:
        con.execute("ALTER TABLE ai_analysis ADD COLUMN token_count TEXT")
    except sqlite3.OperationalError:
        # Column already exists.
        pass
    con.commit()
    con.close()


def _normalise_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def _save_sqlite(data: Dict[str, Any]) -> None:
    init_db()
    con = sqlite3.connect(DB_FILE)
    con.execute(
        """
        INSERT INTO ai_analysis (ts, ticker, expirations, payload, response, token_count)
        VALUES (:ts, :ticker, :expirations, :payload, :response, :token_count)
        """,
        data,
    )
    con.commit()
    con.close()


def _load_sqlite(limit: int) -> List[Dict[str, Any]]:
    init_db()
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT ts, ticker, expirations, payload, response, token_count
        FROM ai_analysis
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    con.close()
    return [dict(row) for row in rows]


def save_analysis(ticker: str, expirations: Any, payload: Any, response: Any, token_count: Any) -> None:
    data = {
        "ts": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "expirations": _normalise_value(expirations),
        "payload": _normalise_value(payload),
        "response": _normalise_value(response),
        "token_count": _normalise_value(token_count),
    }

    if _supabase_client is not None:
        try:
            _supabase_client.table("ai_analysis").insert(data).execute()
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to save analysis to Supabase; falling back to SQLite",
                exc_info=exc,
            )

    _save_sqlite(data)


def load_analyses(limit: int = 20) -> List[Dict[str, Any]]:
    """Return the last ``limit`` analyses, preferring Supabase when available."""

    if _supabase_client is not None:
        try:
            resp = (
                _supabase_client
                .table("ai_analysis")
                .select("*")
                .order("id", desc=True)
                .limit(limit)
                .execute()
            )
            if resp.data:
                return resp.data
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to load analyses from Supabase; falling back to SQLite",
                exc_info=exc,
            )

    return _load_sqlite(limit)
