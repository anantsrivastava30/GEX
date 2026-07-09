"""One-off migration of locally stored AI analyses into Supabase."""

import json
import sqlite3
import sys

from supabase import create_client
import streamlit as st

from quant_analysis.config import CONFIG, PROJECT_ROOT

# Use config for SQLite path and table name:
SQLITE_DB = PROJECT_ROOT / CONFIG.get("database", {}).get("sqlite_db", "ai_analysis.db")
TABLE = CONFIG.get("database", {}).get("supabase_table", "ai_analysis")


def _get_supabase_client():
    supa_url = st.secrets.get("SUPABASE_URL")
    supa_key = st.secrets.get("SUPABASE_KEY")
    if not supa_url or not supa_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set in Streamlit secrets to run the migration."
        )
    return create_client(supa_url, supa_key)


def migrate_sqlite_to_supabase():
    supabase = _get_supabase_client()

    # 1) Open SQLite & fetch everything except the auto-increment id
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.execute(
        "SELECT ts, ticker, expirations, payload, response, token_count "
        "FROM ai_analysis"
    )
    rows = cur.fetchall()
    conn.close()

    # 2) Build a list of JSON-friendly dicts
    to_insert = []
    for ts, ticker, exps_json, payload_json, resp_text, token_count in rows:
        to_insert.append({
            "ts":          ts,
            "ticker":      ticker,
            "expirations": json.loads(exps_json) if exps_json else None,
            "payload":     json.loads(payload_json) if payload_json else None,
            "response":    resp_text,
            "token_count": token_count,
        })

    # 3) Push in batches (Supabase limits ~100 rows per insert)
    BATCH_SIZE = 50
    failures = 0
    for i in range(0, len(to_insert), BATCH_SIZE):
        batch = to_insert[i:i + BATCH_SIZE]
        try:
            supabase.table(TABLE).insert(batch).execute()
            print(f"Inserted batch {i // BATCH_SIZE + 1} ({len(batch)} rows)")
        except Exception as exc:
            failures += 1
            print(f"Error on batch {i // BATCH_SIZE + 1}: {exc}")

    print("Migration complete!" if not failures else f"Migration finished with {failures} failed batch(es).")
    return failures


if __name__ == "__main__":
    sys.exit(1 if migrate_sqlite_to_supabase() else 0)
