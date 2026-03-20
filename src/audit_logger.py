"""
Tier 1 — Audit Logger (SQLite-backed)
=======================================
Records ALL API events: who, what, when, outcome.
In production: swap SQLite DSN for PostgreSQL via DATABASE_URL env var.

Schema:
  id, timestamp, event_type, client_id, role, endpoint, method,
  status_code, latency_ms, ip_address, request_id, extra (JSON blob)
"""

import os
import sqlite3
import json
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_PATH = os.getenv("AUDIT_DB_PATH", "artifacts/audit/audit.db")
_lock   = threading.Lock()


def _ensure_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            event_type  TEXT    NOT NULL,
            client_id   TEXT,
            role        TEXT,
            endpoint    TEXT,
            method      TEXT,
            status_code INTEGER,
            latency_ms  REAL,
            ip_address  TEXT,
            request_id  TEXT,
            extra       TEXT
        )
    """)
    conn.commit()
    return conn


def log_event(
    event_type:  str,
    client_id:   Optional[str]         = None,
    role:        Optional[str]         = None,
    endpoint:    Optional[str]         = None,
    method:      Optional[str]         = None,
    status_code: Optional[int]         = None,
    latency_ms:  Optional[float]       = None,
    ip_address:  Optional[str]         = None,
    request_id:  Optional[str]         = None,
    extra:       Optional[Dict[str, Any]] = None,
) -> bool:
    """Write one audit record. Returns True on success. Never crashes app."""
    try:
        with _lock:
            conn = _ensure_db()
            conn.execute("""
                INSERT INTO audit_log
                (timestamp, event_type, client_id, role, endpoint, method,
                 status_code, latency_ms, ip_address, request_id, extra)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.utcnow().isoformat(),
                event_type,
                client_id,
                role,
                endpoint,
                method,
                status_code,
                latency_ms,
                ip_address,
                request_id,
                json.dumps(extra) if extra else None,
            ))
            conn.commit()
            conn.close()
        return True
    except Exception as exc:
        print(f"[AuditLogger] WARNING: Failed to write audit record: {exc}")
        return False


def query_events(
    event_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    """Fetch recent audit records."""
    try:
        conn = _ensure_db()
        if event_type:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE event_type=? ORDER BY id DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        conn.close()
        cols = [
            "id", "timestamp", "event_type", "client_id", "role",
            "endpoint", "method", "status_code", "latency_ms",
            "ip_address", "request_id", "extra",
        ]
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []


def count_events(event_type: Optional[str] = None) -> int:
    """Return total count of audit records."""
    try:
        conn = _ensure_db()
        if event_type:
            row = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE event_type=?", (event_type,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def clear_all_events() -> None:
    """Truncate audit log (tests only)."""
    try:
        conn = _ensure_db()
        conn.execute("DELETE FROM audit_log")
        conn.commit()
        conn.close()
    except Exception:
        pass
