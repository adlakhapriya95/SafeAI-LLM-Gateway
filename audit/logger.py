"""
SafeAI — Enhanced Audit Logger
--------------------------------
Writes every pipeline interaction to SQLite with
session tracking, IP address, user agent, and
session fingerprint for user identification.

Schema additions over v1:
    session_id      unique per browser session
    ip_address      from HTTP headers (VPN-obscured but useful)
    user_agent      browser and OS string
    fingerprint     hash of ip + user_agent for consistent tracking
    turn_number     which turn in this session (1, 2, 3...)
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("./audit/log.db")


def init_db():
    """
    Creates the audit log database and table.
    Safe to call on every startup — only creates if not exists.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            session_id      TEXT,
            ip_address      TEXT,
            user_agent      TEXT,
            fingerprint     TEXT,
            turn_number     INTEGER DEFAULT 1,
            original_text   TEXT,
            redacted_text   TEXT,
            entities_found  TEXT,
            redaction_count INTEGER,
            has_pii         INTEGER,
            risk_score      REAL,
            risk_action     TEXT,
            faithfulness    REAL,
            response        TEXT,
            phase           TEXT
        )
    """)

    # Add new columns to existing databases that have the old schema
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(interactions)").fetchall()]
    new_cols = {
        "session_id": "TEXT",
        "ip_address": "TEXT",
        "user_agent": "TEXT",
        "fingerprint": "TEXT",
        "turn_number": "INTEGER DEFAULT 1"
    }
    for col, col_type in new_cols.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE interactions ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()


def make_fingerprint(ip: str, user_agent: str) -> str:
    """
    Creates a consistent but non-reversible fingerprint
    from IP + user agent. Same device/browser = same fingerprint.
    Not personally identifiable but useful for session grouping.
    """
    raw = (ip or "") + "|" + (user_agent or "")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_turn_number(session_id: str) -> int:
    """
    Returns how many turns have happened in this session so far.
    Used to number turns within a session for pattern analysis.
    """
    if not DB_PATH.exists():
        return 1
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM interactions WHERE session_id = ?",
            (session_id,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count + 1
    except Exception:
        return 1


def log_interaction(
    original_text: str,
    redacted_text: str,
    entities_found: list,
    redaction_count: int,
    has_pii: bool,
    risk_score: float = None,
    risk_action: str = None,
    faithfulness: float = None,
    response: str = None,
    phase: str = "pipeline",
    session_id: str = None,
    ip_address: str = None,
    user_agent: str = None,
):
    """
    Writes one interaction row to the audit log.
    """
    init_db()

    fingerprint = make_fingerprint(ip_address, user_agent) if (ip_address or user_agent) else None
    turn = get_turn_number(session_id) if session_id else 1

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO interactions (
            timestamp, session_id, ip_address, user_agent,
            fingerprint, turn_number,
            original_text, redacted_text, entities_found,
            redaction_count, has_pii, risk_score, risk_action,
            faithfulness, response, phase
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        session_id,
        ip_address,
        user_agent,
        fingerprint,
        turn,
        original_text,
        redacted_text,
        json.dumps(entities_found),
        redaction_count,
        int(has_pii),
        risk_score,
        risk_action,
        faithfulness,
        response,
        phase
    ))

    conn.commit()
    conn.close()


def fetch_recent(limit: int = 50) -> list:
    """Returns most recent interactions for the compliance dashboard."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def fetch_session(session_id: str) -> list:
    """Returns all interactions for a specific session in order."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM interactions WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def fetch_by_fingerprint(fingerprint: str) -> list:
    """
    Returns all interactions from a specific device fingerprint
    across all sessions. Useful for tracking repeat offenders.
    """
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM interactions WHERE fingerprint = ? ORDER BY id DESC",
        (fingerprint,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
