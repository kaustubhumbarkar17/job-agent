"""Stage 1: Discover — collect candidate job postings.

Builds query strings from titles x locations, calls Firecrawl search,
normalizes results, applies blocklists, and upserts into the DB.
"""
from __future__ import annotations

from contextlib import closing
from itertools import islice, product
from pathlib import Path
from urllib.parse import quote

from agent.config import get_settings
from agent.db import connect, upsert_job
from agent.firecrawl_client import (
    BudgetExceeded,
    FirecrawlClient,
    FirecrawlError,
    SearchResult,
)


def run(
    run_id: str = "",
    dry_run: bool = False,
    db_path: str | Path | None = None,
    window: str | None = None,
    **kwargs,
) -> dict:
    """Run the discover stage. Returns summary stats."""
    settings = get_settings()
    with closing(connect(db_path or settings.db_path)) as conn:
        client = None if dry_run else FirecrawlClient(conn=conn, run_id=run_id)
        discovered = blocked = queries_run = 0
        window = window or settings.search.window
        combinations = product(settings.search.titles, settings.search.locations)
        for title, location in islice(combinations, settings.search.max_queries_per_run):
            query = f'"{title}" {location} jobs posted in the last {window[:-1]} day(s)'
            try:
                results = client.search(query, settings.search.results_per_query, window) if client else [
                    SearchResult(title=title, company="Example", location=location,
                                 url=f"https://example.com/job-{quote(title)}-{quote(location)}"),
                ]
            except BudgetExceeded:
                break
            except FirecrawlError as exc:
                queries_run += 1
                print(f"  Warning: query '{query}' failed: {exc}")
                continue

            queries_run += 1
            for result in results:
                if any(word.casefold() in result.title.casefold()
                       for word in settings.search.blocklist_title_words) or any(
                    company.casefold() in (result.company or "").casefold()
                    for company in settings.search.blocklist_companies
                ):
                    blocked += 1
                    continue
                job_id = upsert_job(
                    conn, result.url, title=result.title, company=result.company,
                    location=result.location or location, source=result.source,
                    posted_at=result.posted_at, run_id=run_id, insert_only=True,
                )
                discovered += job_id != -1

    return {"queries_run": queries_run, "discovered": discovered, "blocked": blocked, "run_id": run_id}
