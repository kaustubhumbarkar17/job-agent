"""Stage 3: Scrape — convert picked postings to structured JSON.

For each picked job, calls Firecrawl scrape with the job schema.
Validates output; failures marked scrape_failed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agent.db import connect, jobs_by_status, set_status
from agent.firecrawl_client import FirecrawlClient


def run(
    run_id: str = "",
    dry_run: bool = False,
    db_path: Optional[str | Path] = None,
) -> dict:
    """Run the scrape stage. Returns success/failure counts."""
    conn = connect(db_path)
    client = FirecrawlClient() if not dry_run else None

    picked = jobs_by_status(conn, "picked")
    scraped = 0
    failed = 0

    for job in picked:
        url = job["url"]

        if dry_run or client is None:
            detail = {
                "title": job["title"],
                "company": job["company"],
                "must_have_skills": ["Python", "React"],
                "nice_to_have_skills": ["Docker"],
                "responsibilities": ["Build features"],
                "company_domain": "example.com",
            }
            conn.execute(
                "UPDATE jobs SET detail_json = ?, status = ? WHERE id = ?",
                (json.dumps(detail), "scraped", job["id"]),
            )
            scraped += 1
            continue

        try:
            schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "must_have_skills": {"type": "array", "items": {"type": "string"}},
                    "company_domain": {"type": ["string", "null"]},
                },
            }
            data = client.scrape_json(url, schema)
            conn.execute(
                "UPDATE jobs SET detail_json = ?, status = ? WHERE id = ?",
                (json.dumps(data), "scraped", job["id"]),
            )
            scraped += 1
        except Exception as e:
            conn.execute(
                "UPDATE jobs SET status = ?, status_reason = ? WHERE id = ?",
                ("scrape_failed", str(e), job["id"]),
            )
            failed += 1

    conn.commit()
    return {"scraped": scraped, "failed": failed, "total": len(picked)}
