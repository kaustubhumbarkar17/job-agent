"""Find and verify addresses for sourced contacts."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

from agent.config import get_settings
from agent.contacts_client import (
    HunterError,
    HunterSuppressed,
    clean_email,
    get_client,
    is_eligible,
)
from agent.db import connect, jobs_by_status


def run(
    db_path: str | Path | None = None, run_id: str = "", dry_run: bool = False, **kwargs,
) -> dict:
    """Advance jobs that have at least one eligible contact."""
    threshold = get_settings().contacts.min_email_confidence
    with closing(connect(db_path)) as conn:
        jobs = jobs_by_status(conn, "contacted")
        client = None if dry_run else get_client(conn=conn, run_id=run_id)
        verified = unverified = failed = 0
        for job in jobs:
            try:
                detail = json.loads(job["detail_json"] or "{}")
            except json.JSONDecodeError:
                detail = {}
            domain = detail.get("company_domain") if isinstance(detail, dict) else None
            contacts = conn.execute(
                "SELECT * FROM contacts WHERE job_id = ? ORDER BY id", (job["id"],),
            ).fetchall()
            eligible = False
            had_error = False
            for contact in contacts:
                email = clean_email(contact["email"])
                status = contact["verify_status"] or "unverified"
                confidence = contact["confidence"] or 0.0
                origin = contact["email_origin"]
                if status == "unverified":
                    try:
                        names = (contact["name"] or "").split(maxsplit=1)
                        if not email and domain and len(names) == 2 and client is not None:
                            lookup = client.find_email(*names, domain)
                            email = lookup.email
                            origin = "api" if email else None
                            if email:
                                with conn:
                                    conn.execute(
                                        "UPDATE contacts SET email = ?, email_origin = ? WHERE id = ?",
                                        (email, origin, contact["id"]),
                                    )
                        if not email:
                            status, confidence = "not_found", 0.0
                        elif dry_run:
                            status, confidence = "valid", 1.0
                        elif client is not None:
                            result = client.verify_email(email)
                            status, confidence = result.status, result.confidence
                    except HunterSuppressed:
                        email, origin, status, confidence = None, None, "invalid", 0.0
                    except HunterError as exc:
                        with conn:
                            conn.execute(
                                "UPDATE jobs SET status_reason = ? WHERE id = ?", (str(exc), job["id"]),
                            )
                        had_error = True
                        continue
                    with conn:
                        conn.execute(
                            """UPDATE contacts SET email = ?, email_origin = ?, verify_status = ?,
                                                   confidence = ? WHERE id = ?""",
                            (email, origin, status, confidence, contact["id"]),
                        )
                eligible |= is_eligible(email, status, confidence, threshold)

            if eligible and not had_error:
                with conn:
                    conn.execute(
                        "UPDATE jobs SET status = 'verified', status_reason = NULL WHERE id = ?",
                        (job["id"],),
                    )
                verified += 1
            else:
                unverified += 1
            failed += had_error

    return {"verified": verified, "unverified": unverified, "failed": failed, "total": len(jobs)}
