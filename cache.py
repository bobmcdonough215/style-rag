import os
import json
import sqlite3
from typing import Optional
import numpy as np
from config import SIMILARITY_THRESHOLD_CACHE, CACHE_DB_PATH


def init_cache():
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            embedding TEXT,
            answer TEXT,
            citations TEXT,
            mode TEXT,
            strict INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_cache(query_embedding: list, mode: str, strict: bool) -> Optional[dict]:
    """
    Check for a semantically similar cached query.
    Returns dict with answer/citations/similarity on hit, None on miss.
    """
    conn = sqlite3.connect(CACHE_DB_PATH)
    rows = conn.execute(
        "SELECT embedding, answer, citations FROM query_cache WHERE mode = ? AND strict = ?",
        (mode, int(strict))
    ).fetchall()
    conn.close()

    for row in rows:
        cached_embedding = json.loads(row[0])
        similarity = _cosine_similarity(query_embedding, cached_embedding)
        if similarity >= SIMILARITY_THRESHOLD_CACHE:
            return {
                "answer": row[1],
                "citations": json.loads(row[2]),
                "similarity": similarity,
            }
    return None


def store_in_cache(query: str, query_embedding: list, answer: str,
                   citations: list, mode: str, strict: bool):
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        "INSERT INTO query_cache (query, embedding, answer, citations, mode, strict) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (query, json.dumps(query_embedding), answer, json.dumps(citations),
         mode, int(strict))
    )
    conn.commit()
    conn.close()
