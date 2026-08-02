# backend/memory.py
import sqlite3
import os

DB_PATH = os.path.join("data", "memory.db")
os.makedirs("data", exist_ok=True)

def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

# initialize table
_conn = _get_conn()
_cur = _conn.cursor()
_cur.execute("""
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim TEXT UNIQUE,
    label TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
_conn.commit()

def remember_claim(claim: str, label: str, confidence: float):
    c = _get_conn().cursor()
    try:
        c.execute("INSERT OR REPLACE INTO claims (claim, label, confidence) VALUES (?, ?, ?)",
                  (claim, label, float(confidence)))
        _get_conn().commit()
    except Exception:
        pass

def lookup_claim(claim: str):
    c = _get_conn().cursor()
    c.execute("SELECT label, confidence FROM claims WHERE claim = ?", (claim,))
    row = c.fetchone()
    if row:
        return {"label": row[0], "confidence": float(row[1])}
    return None
