import sqlite3
from pathlib import Path
import json
from datetime import datetime

DB_FILE = Path("ai_analysis.db")

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

def save_analysis(ticker, expirations, payload, response):
    """
    Insert one AI analysis run.
    - ticker: the underlying symbol (str)
    - expirations: list of expirations used
    - payload: dict you sent to the model
    - response: str the model returned
    """
    con = sqlite3.connect(DB_FILE)
    con.execute("""
      INSERT INTO ai_analysis (ts, ticker, expirations, payload, response)
      VALUES (?, ?, ?, ?, ?)
    """, (
      datetime.utcnow().isoformat(),
      ticker,
      json.dumps(expirations),
      json.dumps(payload),
      response
    ))
    con.commit()
    con.close()

def load_analyses(limit=20):
    """
    Return the last `limit` analyses as a list of dicts.
    """
    # Ensure the table exists
    con = sqlite3.connect(DB_FILE)
    cur = con.execute("""
      SELECT id, ts, ticker, expirations, payload, response, token_count
      FROM ai_analysis
      ORDER BY id DESC
      LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    con.close()
    cols = ["id","ts","ticker","expirations","payload","response","token_count"]
    results = []
    for r in rows:
        rec = dict(zip(cols, r))
        rec["expirations"] = json.loads(rec["expirations"])
        rec["payload"]     = json.loads(rec["payload"])
        results.append(rec)
    return results
