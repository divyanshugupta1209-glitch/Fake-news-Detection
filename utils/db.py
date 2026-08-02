# utils/db.py
import sqlite3
import pandas as pd
import os
from difflib import SequenceMatcher

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "detections.db")
os.makedirs(DB_DIR, exist_ok=True)

def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = _conn()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim TEXT,
        title TEXT,
        url TEXT,
        source TEXT,
        result TEXT,
        score REAL,
        model_conf REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def save_detection(claim, title, url, source, result, score, model_conf):
    conn = _conn()
    c = conn.cursor()
    c.execute("INSERT INTO detections (claim, title, url, source, result, score, model_conf) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (claim, title, url, source, result, float(score), float(model_conf)))
    conn.commit()
    conn.close()

def recent_detections(limit=50):
    conn = _conn()
    df = pd.read_sql("SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df

def find_similar_claim(claim, threshold=0.8):
    """
    Simple similarity search using SequenceMatcher over saved claims.
    Returns closest record if ratio >= threshold, else None.
    """
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT claim, result, model_conf FROM detections")
    rows = c.fetchall()
    conn.close()
    best = None
    best_score = 0.0
    for r in rows:
        s = SequenceMatcher(None, claim, r[0]).ratio()
        if s > best_score:
            best_score = s
            best = r
    if best and best_score >= threshold:
        return {"claim": best[0], "result": best[1], "model_conf": float(best[2]), "similarity": best_score}
    return None

def aggregate_trends():
    conn = _conn()
    df = pd.read_sql("SELECT date(timestamp) as date, result, count(*) as cnt FROM detections GROUP BY date(timestamp), result", conn)
    conn.close()
    return df

def top_fake_sources(limit=10):
    conn = _conn()
    df = pd.read_sql("SELECT source, count(*) as cnt FROM detections WHERE result LIKE '%FAKE%' GROUP BY source ORDER BY cnt DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df
