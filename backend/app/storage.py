"""SQLite persistence for shared watchlists, alert rules, and alert events."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from backend.app.config import get_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = Path(get_settings().app_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_app_storage() -> None:
    with connection() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                symbols_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alert_rules (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                watchlist_id INTEGER NOT NULL,
                query_json TEXT NOT NULL,
                notify_discord INTEGER NOT NULL,
                notify_email INTEGER NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                last_evaluated_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE RESTRICT
            );

            CREATE TABLE IF NOT EXISTS alert_events (
                id INTEGER PRIMARY KEY,
                alert_rule_id INTEGER NOT NULL,
                rule_version INTEGER NOT NULL,
                fingerprint TEXT NOT NULL UNIQUE,
                row_key TEXT NOT NULL,
                snapshot_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                scope TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                read_at TEXT,
                discord_status TEXT NOT NULL,
                discord_attempts INTEGER NOT NULL DEFAULT 0,
                discord_last_error TEXT,
                discord_sent_at TEXT,
                email_status TEXT NOT NULL,
                email_attempts INTEGER NOT NULL DEFAULT 0,
                email_last_error TEXT,
                email_sent_at TEXT,
                FOREIGN KEY (alert_rule_id) REFERENCES alert_rules(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS alert_events_unread_idx
            ON alert_events(read_at, created_at DESC);
            CREATE INDEX IF NOT EXISTS alert_events_rule_idx
            ON alert_events(alert_rule_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS direction_signals (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                label TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                signal_date TEXT NOT NULL,
                state TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(symbol, signal_type, signal_date)
            );

            CREATE INDEX IF NOT EXISTS direction_signals_date_idx
            ON direction_signals(signal_date DESC, created_at DESC);

            CREATE TABLE IF NOT EXISTS fundamentals_history (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                as_of TEXT NOT NULL,
                institutional_pct REAL,
                institutional_holder_count INTEGER,
                quarterly_eps_growth_pct REAL,
                quarterly_revenue_growth_pct REAL,
                annual_eps_growth_pct REAL,
                roe_pct REAL,
                created_at TEXT NOT NULL,
                UNIQUE(symbol, as_of)
            );

            CREATE INDEX IF NOT EXISTS fundamentals_history_symbol_idx
            ON fundamentals_history(symbol, as_of DESC);
            """
        )
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(alert_events)").fetchall()
        }
        for column in ("discord_attempts", "email_attempts"):
            if column not in columns:
                conn.execute(
                    f"ALTER TABLE alert_events ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                )


def _watchlist(row: sqlite3.Row) -> Dict[str, Any]:
    item = dict(row)
    item["symbols"] = json.loads(item.pop("symbols_json"))
    return item


def list_watchlists() -> List[Dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute("SELECT * FROM watchlists ORDER BY name COLLATE NOCASE").fetchall()
    return [_watchlist(row) for row in rows]


def get_watchlist(watchlist_id: int) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        row = conn.execute("SELECT * FROM watchlists WHERE id = ?", (watchlist_id,)).fetchone()
    return _watchlist(row) if row else None


def ensure_default_watchlist(symbols: List[str]) -> None:
    with connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM watchlists").fetchone()[0]
        if count:
            return
        now = _now()
        conn.execute(
            "INSERT INTO watchlists(name, symbols_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("Default", json.dumps(symbols), now, now),
        )


def create_watchlist(name: str, symbols: List[str]) -> Dict[str, Any]:
    now = _now()
    with connection() as conn:
        cursor = conn.execute(
            "INSERT INTO watchlists(name, symbols_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, json.dumps(symbols), now, now),
        )
        row = conn.execute("SELECT * FROM watchlists WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _watchlist(row)


def update_watchlist(watchlist_id: int, name: str, symbols: List[str]) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        cursor = conn.execute(
            "UPDATE watchlists SET name = ?, symbols_json = ?, updated_at = ? WHERE id = ?",
            (name, json.dumps(symbols), _now(), watchlist_id),
        )
        if not cursor.rowcount:
            return None
        row = conn.execute("SELECT * FROM watchlists WHERE id = ?", (watchlist_id,)).fetchone()
    return _watchlist(row)


def watchlist_in_use(watchlist_id: int) -> bool:
    with connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM alert_rules WHERE watchlist_id = ?", (watchlist_id,)
        ).fetchone()[0]
    return bool(count)


def delete_watchlist(watchlist_id: int) -> bool:
    with connection() as conn:
        cursor = conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))
    return bool(cursor.rowcount)


def _rule(row: sqlite3.Row) -> Dict[str, Any]:
    item = dict(row)
    item["enabled"] = bool(item["enabled"])
    item["notify_discord"] = bool(item["notify_discord"])
    item["notify_email"] = bool(item["notify_email"])
    item["query"] = json.loads(item.pop("query_json"))
    return item


_RULE_SELECT = """
    SELECT r.*, w.name AS watchlist_name
    FROM alert_rules r JOIN watchlists w ON w.id = r.watchlist_id
"""


def list_alert_rules(enabled_only: bool = False) -> List[Dict[str, Any]]:
    sql = _RULE_SELECT
    params: tuple = ()
    if enabled_only:
        sql += " WHERE r.enabled = 1"
    sql += " ORDER BY r.name COLLATE NOCASE"
    with connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_rule(row) for row in rows]


def get_alert_rule(rule_id: int) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        row = conn.execute(_RULE_SELECT + " WHERE r.id = ?", (rule_id,)).fetchone()
    return _rule(row) if row else None


def create_alert_rule(data: Dict[str, Any]) -> Dict[str, Any]:
    now = _now()
    with connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO alert_rules(
                name, enabled, watchlist_id, query_json, notify_discord,
                notify_email, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["name"],
                int(data["enabled"]),
                data["watchlist_id"],
                json.dumps(data["query"]),
                int(data["notify_discord"]),
                int(data["notify_email"]),
                now,
                now,
            ),
        )
        row = conn.execute(_RULE_SELECT + " WHERE r.id = ?", (cursor.lastrowid,)).fetchone()
    return _rule(row)


def update_alert_rule(rule_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        cursor = conn.execute(
            """
            UPDATE alert_rules SET name = ?, enabled = ?, watchlist_id = ?,
                query_json = ?, notify_discord = ?, notify_email = ?,
                version = version + 1, last_error = NULL, updated_at = ?
            WHERE id = ?
            """,
            (
                data["name"],
                int(data["enabled"]),
                data["watchlist_id"],
                json.dumps(data["query"]),
                int(data["notify_discord"]),
                int(data["notify_email"]),
                _now(),
                rule_id,
            ),
        )
        if not cursor.rowcount:
            return None
        row = conn.execute(_RULE_SELECT + " WHERE r.id = ?", (rule_id,)).fetchone()
    return _rule(row)


def delete_alert_rule(rule_id: int) -> bool:
    with connection() as conn:
        cursor = conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
    return bool(cursor.rowcount)


def alert_rule_has_events(rule_id: int) -> bool:
    with connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM alert_events WHERE alert_rule_id = ?", (rule_id,)
        ).fetchone()[0]
    return bool(count)


def update_rule_evaluation(rule_id: int, error: Optional[str]) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE alert_rules SET last_evaluated_at = ?, last_error = ? WHERE id = ?",
            (_now(), error, rule_id),
        )


def insert_alert_event(data: Dict[str, Any]) -> Optional[int]:
    with connection() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO alert_events(
                alert_rule_id, rule_version, fingerprint, row_key, snapshot_date,
                symbol, scope, title, message, payload_json, created_at,
                discord_status, email_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["alert_rule_id"],
                data["rule_version"],
                data["fingerprint"],
                data["row_key"],
                data["snapshot_date"],
                data["symbol"],
                data["scope"],
                data["title"],
                data["message"],
                json.dumps(data["payload"]),
                _now(),
                data["discord_status"],
                data["email_status"],
            ),
        )
        return int(cursor.lastrowid) if cursor.rowcount else None


def update_delivery(
    event_ids: List[int], channel: str, status: str, error: Optional[str] = None
) -> None:
    if not event_ids or channel not in {"discord", "email"}:
        return
    placeholders = ",".join("?" for _ in event_ids)
    sent_at = _now() if status == "sent" else None
    with connection() as conn:
        conn.execute(
            f"""
            UPDATE alert_events SET {channel}_status = ?, {channel}_last_error = ?,
                {channel}_sent_at = ?, {channel}_attempts = {channel}_attempts + 1
            WHERE id IN ({placeholders})
            """,
            (status, error, sent_at, *event_ids),
        )


def list_pending_deliveries(channel: str, limit: int = 100) -> List[Dict[str, Any]]:
    if channel not in {"discord", "email"}:
        return []
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT e.*, r.name AS rule_name
            FROM alert_events e JOIN alert_rules r ON r.id = e.alert_rule_id
            WHERE e.{channel}_status IN ('pending', 'failed')
              AND e.{channel}_attempts < 3
            ORDER BY e.created_at ASC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_event(row) for row in rows]


def _event(row: sqlite3.Row) -> Dict[str, Any]:
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json"))
    return item


def list_alert_events(unread_only: bool, limit: int) -> Dict[str, Any]:
    where = "WHERE e.read_at IS NULL" if unread_only else ""
    sql = f"""
        SELECT e.*, r.name AS rule_name
        FROM alert_events e JOIN alert_rules r ON r.id = e.alert_rule_id
        {where} ORDER BY e.created_at DESC LIMIT ?
    """
    with connection() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
        unread = conn.execute(
            "SELECT COUNT(*) FROM alert_events WHERE read_at IS NULL"
        ).fetchone()[0]
    return {"unread": unread, "items": [_event(row) for row in rows]}


def mark_alert_event_read(event_id: int) -> bool:
    with connection() as conn:
        cursor = conn.execute(
            "UPDATE alert_events SET read_at = COALESCE(read_at, ?) WHERE id = ?",
            (_now(), event_id),
        )
    return bool(cursor.rowcount)


def mark_alert_events_read(event_ids: Sequence[int]) -> int:
    ids = [int(value) for value in event_ids]
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    with connection() as conn:
        cursor = conn.execute(
            "UPDATE alert_events SET read_at = ? "
            f"WHERE read_at IS NULL AND id IN ({placeholders})",
            (_now(), *ids),
        )
    return int(cursor.rowcount)


def mark_all_alert_events_read() -> int:
    with connection() as conn:
        cursor = conn.execute(
            "UPDATE alert_events SET read_at = ? WHERE read_at IS NULL", (_now(),)
        )
    return int(cursor.rowcount)


def insert_direction_signal(data: Dict[str, Any]) -> Optional[int]:
    """Append one market-direction signal; None when it already exists."""

    with connection() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO direction_signals(
                symbol, label, signal_type, signal_date, state, title,
                message, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["symbol"],
                data["label"],
                data["signal_type"],
                data["signal_date"],
                data["state"],
                data["title"],
                data["message"],
                json.dumps(data["payload"]),
                _now(),
            ),
        )
        return int(cursor.lastrowid) if cursor.rowcount else None


_FUNDAMENTAL_METRICS = (
    "institutional_pct",
    "institutional_holder_count",
    "quarterly_eps_growth_pct",
    "quarterly_revenue_growth_pct",
    "annual_eps_growth_pct",
    "roe_pct",
)


def record_fundamentals_observation(
    symbol: str, as_of: str, metrics: Dict[str, Any]
) -> None:
    """Append one dated fundamentals observation (idempotent per day).

    Yahoo reports only current values, so sponsorship and growth *trends*
    have to come from our own accumulated observations.
    """

    values = [metrics.get(name) for name in _FUNDAMENTAL_METRICS]
    if all(v is None for v in values):
        return
    columns = ", ".join(_FUNDAMENTAL_METRICS)
    placeholders = ", ".join("?" for _ in _FUNDAMENTAL_METRICS)
    updates = ", ".join(f"{name} = excluded.{name}" for name in _FUNDAMENTAL_METRICS)
    with connection() as conn:
        conn.execute(
            f"""
            INSERT INTO fundamentals_history(symbol, as_of, {columns}, created_at)
            VALUES (?, ?, {placeholders}, ?)
            ON CONFLICT(symbol, as_of) DO UPDATE SET {updates}
            """,
            (symbol.upper(), as_of, *values, _now()),
        )


def get_fundamentals_trend(symbol: str, limit: int = 12) -> List[Dict[str, Any]]:
    """Recent dated observations for one symbol, newest first."""

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM fundamentals_history
            WHERE symbol = ? ORDER BY as_of DESC LIMIT ?
            """,
            (symbol.upper(), limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_fundamentals_trends(symbols: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Observations for many symbols in one pass, newest first per symbol."""

    if not symbols:
        return {}
    placeholders = ",".join("?" for _ in symbols)
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM fundamentals_history
            WHERE symbol IN ({placeholders})
            ORDER BY symbol, as_of DESC
            """,
            tuple(s.upper() for s in symbols),
        ).fetchall()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        out.setdefault(item["symbol"], []).append(item)
    return out


def list_direction_signals(
    limit: int, symbol: Optional[str] = None
) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM direction_signals"
    params: List[Any] = []
    if symbol:
        sql += " WHERE symbol = ?"
        params.append(symbol)
    sql += " ORDER BY signal_date DESC, created_at DESC LIMIT ?"
    params.append(limit)
    with connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        out.append(item)
    return out
