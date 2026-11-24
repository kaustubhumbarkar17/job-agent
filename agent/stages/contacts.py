"""Save sourced professional contacts for shortlisted jobs."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

from agent.config import get_settings
from agent.contacts_client import (
    HunterError,
    Person,
    clean_email,
    get_client,
    role_priority,
)
from agent.db import connect, jobs_by_status


def run(
    db_path: str | Path | None = None, run_id: str = "", dry_run: bool = False, **kwargs,
) -> dict:
    """Use named contacts listed in the posting, then fill remaining slots via Hunter."""
    settings = get_settings().contacts
    with closing(connect(db_path)) as conn:
        jobs = jobs_by_status(conn, "shortlisted")
        client = None if dry_run else get_client(conn=conn, run_id=run_id)
        found = no_contact = failed = 0
        for job in jobs:
            try:
                detail = json.loads(job["detail_json"] or "{}")
                if not isinstance(detail, dict):
                    raise TypeError("Job detail must be a JSON object")
                company = detail.get("company") or job["company"] or ""
                domain = detail.get("company_domain") or None
                candidates = []
                posting = detail.get("posting_contact")
                if isinstance(posting, dict):
                    name, role = posting.get("name"), posting.get("role")
                    if all(isinstance(value, str) and value.strip() for value in (name, role)):
                        email = clean_email(posting.get("email"))
                        candidates.append(Person(
                            name=name.strip(), role=role.strip(), company=company,
                            source_url=job["url"], email=email,
                            email_origin="posting" if email else None,
                        ))
                if len(candidates) < settings.max_per_job:
                    if dry_run:
                        candidates.append(Person(
                            name="Fixture Recruiter", role="Recruiter", company=company,
                            source_url=job["url"], email="fixture@example.com", email_origin="api",
                        ))
                    elif client is not None:
                        candidates.extend(client.find_people(company, domain, settings.target_roles))
            except (HunterError, TypeError, ValueError) as exc:
                conn.execute("UPDATE jobs SET status_reason = ? WHERE id = ?", (str(exc), job["id"]))
                conn.commit()
                failed += 1
                continue

            candidates.sort(key=lambda candidate: role_priority(candidate.role, settings.target_roles))
            seen = set()
            with conn:
                for candidate in candidates:
                    key = candidate.name.casefold()
                    if key in seen:
                        continue
                    if len(seen) >= settings.max_per_job:
                        break
                    seen.add(key)
                    conn.execute(
                        """INSERT INTO contacts (job_id, company, name, role, source_url,
                                                email, email_origin, confidence)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(job_id, name) DO NOTHING""",
                        (job["id"], candidate.company, candidate.name, candidate.role,
                         candidate.source_url, candidate.email, candidate.email_origin,
                         candidate.confidence),
                    )
                status = "contacted" if seen else "no_contact"
                conn.execute(
                    "UPDATE jobs SET status = ?, status_reason = ? WHERE id = ?",
                    (status, None if seen else "No sourced contacts found", job["id"]),
                )
            found += bool(seen)
            no_contact += not seen

    return {"found": found, "no_contact": no_contact, "failed": failed, "total": len(jobs)}
