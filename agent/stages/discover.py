"""Stage 1: Discover — collect candidate job postings.

Builds query strings from titles x locations, calls Firecrawl search,
normalizes results, applies blocklists, and upserts into the DB.
"""
from __future__ import annotations

from pathlib import Path

from agent.config import get_settings
from agent.db import connect, upsert_job
from agent.firecrawl_client import BudgetExceeded, FirecrawlClient, FirecrawlError


def run(
    run_id: str = "",
    dry_run: bool = False,
    db_path: str | Path | None = None,
    window: str | None = None,
    **kwargs,
) -> dict:
    """Run the discover stage. Returns summary stats."""
    settings = get_settings()
    conn = connect(db_path or settings.db_path)
    client = FirecrawlClient() if not dry_run else None

    window = window or settings.search.window
    discovered = 0
    blocked = 0
    queries_run = 0
    max_queries = settings.search.max_queries_per_run

    for title in settings.search.titles:
        if queries_run >= max_queries:
            break
        for location in settings.search.locations:
            if queries_run >= max_queries:
                break
            query = f'"{title}" {location} jobs posted in the last {window[:-1]} day(s)'
            queries_run += 1

            if dry_run or client is None:
                url = f"https://example.com/job-{title}-{location}"
                upsert_job(
                    conn, url, title=title, company="Example",
                    location=location, source="firecrawl", run_id=run_id,
                )
                discovered += 1
                continue

            try:
                results = client.search(query, limit=settings.search.results_per_query, time_window=window)
                for r in results:
                    title_lower = (r.title or "").lower()
                    company_lower = (r.company or "").lower()
                    if any(w in title_lower for w in settings.search.blocklist_title_words):
                        blocked += 1
                        continue
                    if any(c in company_lower for c in settings.search.blocklist_companies):
                        blocked += 1
                        continue
                    upsert_job(
                        conn, r.url, title=r.title, company=r.company,
                        location=r.location, source=r.source,
                        posted_at=r.posted_at, run_id=run_id,
                    )
                    discovered += 1
            except BudgetExceeded:
                break
            except FirecrawlError as e:
                print(f"  Warning: query '{query}' failed: {e}")

    return {
        "queries_run": queries_run,
        "discovered": discovered,
        "blocked": blocked,
        "run_id": run_id,
    }
