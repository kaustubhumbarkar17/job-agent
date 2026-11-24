"""Extract structured job details and retain failed postings for retry."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

from jsonschema import ValidationError, validate

from agent.db import connect, jobs_by_status
from agent.firecrawl_client import FirecrawlClient, FirecrawlError


def run(
    run_id: str = "",
    dry_run: bool = False,
    db_path: str | Path | None = None,
    **kwargs,
) -> dict:
    """Scrape picked and previously failed jobs independently."""
    schema = json.loads(Path("schemas/job.json").read_text(encoding="utf-8"))
    with closing(connect(db_path)) as conn:
        picked = jobs_by_status(conn, "picked") + jobs_by_status(conn, "scrape_failed")
        if not picked:
            return {"scraped": 0, "failed": 0, "total": 0}

        client = None if dry_run else FirecrawlClient(conn=conn, run_id=run_id)
        scraped = failed = 0
        for job in picked:
            try:
                detail = client.scrape_json(job["url"], schema) if client else {
                    "title": job["title"] or "Example job",
                    "url": job["url"],
                    "company": job["company"] or "Example",
                    "must_have_skills": ["Python", "React"],
                    "nice_to_have_skills": ["Docker"],
                    "responsibilities": ["Build features"],
                    "company_domain": "example.com",
                }
                validate(detail, schema)
            except (FirecrawlError, ValidationError) as exc:
                conn.execute(
                    "UPDATE jobs SET status = 'scrape_failed', status_reason = ? WHERE id = ?",
                    (str(exc), job["id"]),
                )
                failed += 1
            else:
                conn.execute(
                    """UPDATE jobs SET detail_json = ?, status = 'scraped', status_reason = NULL
                       WHERE id = ?""",
                    (json.dumps(detail), job["id"]),
                )
                scraped += 1
            conn.commit()

    return {"scraped": scraped, "failed": failed, "total": len(picked)}
