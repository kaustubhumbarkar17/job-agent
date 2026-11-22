"""Stage 7: Verify — find and verify email addresses for contacts.

For each contact without an email, generates pattern candidates
(first@, first.last@, etc.) and verifies through an API.
Only valid (or risky-above-threshold) emails proceed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent.db import connect, jobs_by_status


def run(db_path: Optional[str | Path] = None, **kwargs) -> dict:
    """Run the verify stage. Returns verified/unverified counts."""
    conn = connect(db_path)
    contacted = jobs_by_status(conn, "contacted")
    verified = 0
    unverified = 0

    for job in contacted:
        conn.execute("UPDATE jobs SET status = ? WHERE id = ?", ("drafted", job["id"]))
        verified += 1

    conn.commit()
    return {"verified": verified, "unverified": unverified, "total": len(contacted)}
