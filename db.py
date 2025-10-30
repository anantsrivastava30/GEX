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
from typing import Any, Dict, List, Optional

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
    con.execute(
        """
      CREATE TABLE IF NOT EXISTS gamma_gap_snapshots (
         id             INTEGER PRIMARY KEY AUTOINCREMENT,
         ts             TEXT NOT NULL,
         ticker         TEXT NOT NULL,
         expiration     TEXT,
         dte            INTEGER,
         spot           REAL,
         magnet         REAL,
         magnet_gamma   REAL,
         lower_bound    REAL,
         upper_bound    REAL,
         gap            REAL,
         gap_pct        REAL,
         score          REAL,
         bias_note      TEXT,
         metadata       TEXT
      )
    """
    )
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


def _parse_token_value(value: Any) -> int:
    """Best-effort conversion of a stored token count into an integer."""

    if value in (None, "", "None"):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _sum_sqlite_tokens() -> int:
    init_db()
    con = sqlite3.connect(DB_FILE)
    total = 0
    try:
        for (value,) in con.execute("SELECT token_count FROM ai_analysis"):
            total += _parse_token_value(value)
    finally:
        con.close()
    return total


def _sum_supabase_tokens() -> Optional[int]:
    if _supabase_client is None:
        return None
    try:
        resp = _supabase_client.table("ai_analysis").select("token_count").execute()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Failed to fetch token counts from Supabase; falling back to SQLite",
            exc_info=exc,
        )
        return None

    if not resp.data:
        return 0

    total = 0
    for row in resp.data:
        total += _parse_token_value(row.get("token_count"))
    return total


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


def get_total_token_usage() -> int:
    """Return the aggregate number of tokens recorded across analyses."""

    supabase_total = _sum_supabase_tokens()
    if supabase_total is not None:
        return supabase_total

    return _sum_sqlite_tokens()


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


def save_gamma_gap_snapshot(entry: Dict[str, Any]) -> None:
    """Persist a gamma-gap prediction snapshot for future validation."""

    payload = {
        "ts": entry.get("ts") or datetime.utcnow().isoformat(),
        "ticker": entry.get("ticker"),
        "expiration": entry.get("expiration"),
        "dte": entry.get("dte"),
        "spot": entry.get("spot"),
        "magnet": entry.get("magnet"),
        "magnet_gamma": entry.get("magnet_gamma"),
        "lower_bound": entry.get("lower_bound"),
        "upper_bound": entry.get("upper_bound"),
        "gap": entry.get("gap"),
        "gap_pct": entry.get("gap_pct"),
        "score": entry.get("score"),
        "bias_note": entry.get("bias_note"),
        "metadata": _normalise_value(entry.get("metadata")),
    }

    if _supabase_client is not None:
        try:
            _supabase_client.table("gamma_gap_snapshots").insert(payload).execute()
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to save gamma gap snapshot to Supabase; falling back to SQLite",
                exc_info=exc,
            )

    init_db()
    con = sqlite3.connect(DB_FILE)
    con.execute(
        """
        INSERT INTO gamma_gap_snapshots (
            ts, ticker, expiration, dte, spot, magnet, magnet_gamma,
            lower_bound, upper_bound, gap, gap_pct, score, bias_note, metadata
        )
        VALUES (:ts, :ticker, :expiration, :dte, :spot, :magnet, :magnet_gamma,
                :lower_bound, :upper_bound, :gap, :gap_pct, :score, :bias_note, :metadata)
        """,
        payload,
    )
    con.commit()
    con.close()


def load_gamma_gap_snapshots(limit: int = 100) -> List[Dict[str, Any]]:
    """Return the most recent gamma gap predictions."""

    if _supabase_client is not None:
        try:
            resp = (
                _supabase_client.table("gamma_gap_snapshots")
                .select("ts,ticker,expiration,dte,spot,magnet,magnet_gamma,lower_bound,upper_bound,gap,gap_pct,score,bias_note,metadata")
                .order("ts", desc=True)
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to load gamma gap snapshots from Supabase; falling back to SQLite",
                exc_info=exc,
            )

    init_db()
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT ts, ticker, expiration, dte, spot, magnet, magnet_gamma,
               lower_bound, upper_bound, gap, gap_pct, score, bias_note, metadata
        FROM gamma_gap_snapshots
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    con.close()
    return [dict(row) for row in rows]


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
