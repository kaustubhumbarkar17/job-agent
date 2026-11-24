"""Database layer for the Job-Hunt Agent.

All helpers are idempotent and use SQLite WAL mode with foreign keys enabled.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Allowed status transitions: from_status -> set of valid next statuses
STATUS_TRANSITIONS: dict[str, set[str]] = {
    "discovered": {"picked", "rejected"},
    "picked": {"scraped", "scrape_failed"},
    "scraped": {"scored", "rejected"},
    "scored": {"shortlisted", "rejected"},
    "shortlisted": {"contacted", "no_contact"},
    "contacted": {"drafted"},
}


def _get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Create and return a connection with row_factory, FK, WAL."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: str | Path) -> None:
    """Apply schema.sql idempotently. Creates the file if it doesn't exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        schema = _SCHEMA_PATH.read_text()
        conn.executescript(schema)
        conn.commit()
    finally:
        conn.close()


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection, applying schema if needed.

    Uses settings.db_path as default if not provided.
    """
    if db_path is None:
        from agent.config import get_settings
        db_path = get_settings().db_path
    path = Path(db_path)
    if not path.exists():
        init_db(path)
    return _get_connection(path)


def canonicalize_url(url: str) -> str:
    """Lowercase host, strip tracking params and trailing slash."""
    import urllib.parse as urlparse

    parsed = urlparse.urlparse(url)
    cleaned_netloc = parsed.netloc.lower()
    cleaned_path = parsed.path.rstrip("/")
    tracking_params = {"utm_", "ref", "gh_src"}
    query_parts = []
    for part in parsed.query.split("&"):
        key = part.split("=")[0]
        if not any(key.startswith(prefix) for prefix in tracking_params):
            query_parts.append(part)
    cleaned_query = "&".join(query_parts)

    result = urlparse.urlunparse((
        parsed.scheme,
        cleaned_netloc,
        cleaned_path,
        parsed.params,
        "",
        "",
    ))
    if cleaned_query:
        result += "?" + cleaned_query
    return result


def url_hash(url: str) -> str:
    """SHA256 hex digest of the canonicalized URL."""
    return hashlib.sha256(canonicalize_url(url).encode()).hexdigest()


def upsert_job(
    conn: sqlite3.Connection,
    url: str,
    title: str = "",
    company: str = "",
    location: str = "",
    source: str = "",
    posted_at: str = "",
    run_id: str | None = None,
    *,
    insert_only: bool = False,
) -> int:
    """Insert or ignore a job. Returns the job ID or -1 for an existing job in insert-only mode."""
    h = url_hash(url)
    cursor = conn.execute(
        """INSERT OR IGNORE INTO jobs (url_hash, url, title, company, location, source, posted_at, run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (h, url, title, company, location, source, posted_at, run_id),
    )
    conn.commit()
    if insert_only and cursor.rowcount == 0:
        return -1
    row = conn.execute("SELECT id FROM jobs WHERE url_hash = ?", (h,)).fetchone()
    return row["id"] if row else -1


def set_status(
    conn: sqlite3.Connection,
    job_id: int,
    status: str,
    reason: str | None = None,
) -> None:
    """Set a job's status. Validates against allowed transitions."""
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise ValueError(f"Job {job_id} does not exist")
    current = row["status"]
    valid_next = STATUS_TRANSITIONS.get(current, set())
    if status not in valid_next:
        raise ValueError(
            f"Illegal transition for job {job_id}: '{current}' -> '{status}' "
            f"(valid: {valid_next})"
        )
    conn.execute(
        "UPDATE jobs SET status = ?, status_reason = ? WHERE id = ?",
        (status, reason, job_id),
    )
    conn.commit()


def jobs_by_status(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    """Return all rows with the given status."""
    return conn.execute(
        "SELECT * FROM jobs WHERE status = ? ORDER BY discovered_at DESC", (status,)
    ).fetchall()


def save_embedding(
    conn: sqlite3.Connection,
    kind: str,
    ref_id: int,
    model: str,
    text: str,
    vector: list[float],
) -> None:
    """Save a vector only if the text has changed (detected via hash)."""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    existing = conn.execute(
        """SELECT 1 FROM embeddings
           WHERE kind=? AND ref_id=? AND model=? AND text_hash=?""",
        (kind, ref_id, model, text_hash),
    ).fetchone()
    if existing:
        return  # Text unchanged, skip re-embed
    blob = np.asarray(vector, dtype=np.float32).tobytes()
    conn.execute(
        """INSERT INTO embeddings (kind, ref_id, model, text_hash, vector)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(kind, ref_id, model) DO UPDATE SET
               text_hash = excluded.text_hash, vector = excluded.vector""",
        (kind, ref_id, model, text_hash, blob),
    )
    conn.commit()


def load_embeddings(
    conn: sqlite3.Connection,
    kind: str,
    model: str,
    text: str | None = None,
) -> list[tuple[int, list[float]]]:
    """Load embeddings. If text is provided, only return rows whose
    stored text_hash matches the hash of the given text."""
    if text is not None:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        rows = conn.execute(
            "SELECT ref_id, vector FROM embeddings WHERE kind=? AND model=? AND text_hash=?",
            (kind, model, text_hash),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ref_id, vector FROM embeddings WHERE kind=? AND model=?",
            (kind, model),
        ).fetchall()
    result = []
    for row in rows:
        vec = np.frombuffer(row["vector"], dtype=np.float32).tolist()
        result.append((row["ref_id"], vec))
    return result


def log_api_call(
    conn: sqlite3.Connection,
    service: str,
    endpoint: str = "",
    credits: float = 0.0,
    latency_ms: int = 0,
    ok: bool = True,
    run_id: str = "",
) -> None:
    """Log an external API call."""
    conn.execute(
        """INSERT INTO api_calls (service, endpoint, credits, latency_ms, ok, run_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (service, endpoint, credits, latency_ms, int(ok), run_id),
    )
    conn.commit()


def get_jobs_by_status_count(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Return (status, count) for all statuses."""
    return conn.execute(
        "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
    ).fetchall()
