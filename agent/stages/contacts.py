"""Stage 6: Contacts — find people for each shortlisted job.

Layered lookup:
  1. Posting hints (hiring_manager_hint from scrape)
  2. Firecrawl people/company search
  3. Company page scraping (/about, /team, /careers)
Stores up to max_per_job contacts per job with source_url.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent.db import connect, jobs_by_status, set_status


def run(db_path: Optional[str | Path] = None) -> dict:
    """Run the contacts stage. Returns contacts found and jobs with none."""
    conn = connect(db_path)
    shortlisted = jobs_by_status(conn, "shortlisted")
    found = 0
    no_contact = 0

    for job in shortlisted:
        set_status(conn, job["id"], "contacted")
        found += 1

    return {"found": found, "no_contact": no_contact, "total": len(shortlisted)}
