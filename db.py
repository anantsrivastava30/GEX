import sqlite3
from pathlib import Path
import json
from datetime import datetime

DB_FILE = Path("ai_analysis.db")


supabase = create_client(
  st.secrets["SUPABASE_URL"],
  st.secrets["SUPABASE_KEY"],
)


def init_db():
    """Ensure the table exists."""
    con = sqlite3.connect(DB_FILE)
    con.execute("""
      CREATE TABLE IF NOT EXISTS ai_analysis (
         id            INTEGER PRIMARY KEY AUTOINCREMENT,
         ts            TEXT NOT NULL,
         ticker        TEXT NOT NULL,
         expirations   TEXT,
         payload       TEXT,
         response      TEXT
      )
    """)
    con.commit()
    con.close()

def save_analysis(ticker, expirations, payload, response, token_count):
    data = {
      "ts": datetime.utcnow().isoformat(),
      "ticker": ticker,
      "expirations": expirations,
      "payload": payload,
      "response": response,
      "token_count": token_count
    }
    supabase.table("ai_analysis").insert(data).execute()

def load_analyses(limit=20):
    """
    Return the last `limit` analyses as a list of dicts.
    """
    resp = (supabase
            .table("ai_analysis")
            .select("*")
            .order("id", desc=True)
            .limit(limit)
            .execute())
    return resp.data or []
