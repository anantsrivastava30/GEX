import os
import sqlite3
import json
from supabase import create_client
import streamlit as st

# ─── CONFIG ────────────────────────────────────────────────────────────────
# Make sure your .streamlit/secrets.toml has SUPABASE_URL and SUPABASE_KEY
# SUPA_URL = os.getenv("SUPABASE_URL")
SUPA_URL = st.secrets.get("SUPABASE_URL")
# SUPA_KEY = os.getenv("SUPABASE_KEY")
SUPA_KEY = st.secrets.get("SUPABASE_KEY")
supabase = create_client(SUPA_URL, SUPA_KEY)

SQLITE_DB = "ai_analysis.db"   # path to your local SQLite file
TABLE     = "ai_analysis"      # Supabase table name

# ─── MIGRATION ─────────────────────────────────────────────────────────────
def migrate_sqlite_to_supabase():
    # 1) Open SQLite & fetch everything except the auto-increment id
    conn = sqlite3.connect(SQLITE_DB)
    cur  = conn.execute(
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
            "expirations": json.loads(exps_json),
            "payload":     json.loads(payload_json),
            "response":    resp_text,
            "token_count": token_count
        })

    # 3) Push in batches (Supabase limits ~100 rows per insert)
    BATCH_SIZE = 50
    for i in range(0, len(to_insert), BATCH_SIZE):
        batch = to_insert[i:i+BATCH_SIZE]
        res = supabase.table(TABLE).insert(batch).execute()
        if res.error:
            print("Error on batch", i, res.error)
        else:
            print(f"Inserted batch {i//BATCH_SIZE + 1} ({len(batch)} rows)")

    print("Migration complete!")

if __name__ == "__main__":
    migrate_sqlite_to_supabase()
