"""Firecrawl client for job search and scraping.

Provides thin wrappers around Firecrawl's search and scrape APIs
with retry/backoff via tenacity.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import BaseModel, ValidationError

from agent.config import get_settings


class FirecrawlError(Exception):
    """Base exception for Firecrawl failures."""


class BudgetExceeded(Exception):
    """Raised when we exceed max_queries_per_run.

    Deliberately NOT a subclass of FirecrawlError so tenacity does not retry.
    """


class SearchResult(BaseModel):
    title: str
    url: str
    company: Optional[str] = ""
    location: Optional[str] = ""
    source: str = "firecrawl"
    posted_at: Optional[str] = None


class FirecrawlClient:
    """Thin wrapper around Firecrawl APIs with retry and budget tracking."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_settings().firecrawl_api_key
        if not self.api_key:
            raise FirecrawlError("FIRECRAWL_API_KEY is not set")
        self.base_url = "https://api.firecrawl.dev"
        self._queries_used = 0

    @property
    def queries_remaining(self) -> int:
        return get_settings().search.max_queries_per_run - self._queries_used

    def _ensure_budget(self) -> None:
        if self.queries_remaining <= 0:
            raise BudgetExceeded(
                f"Query budget exhausted ({get_settings().search.max_queries_per_run} used)"
            )

    @retry(
        retry=retry_if_exception_type(FirecrawlError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def search(
        self,
        query: str,
        limit: int = 10,
        time_window: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search Firecrawl for job postings matching a query."""
        self._ensure_budget()
        self._queries_used += 1

        payload = {
            "search": query,
            "limit": min(limit, self.queries_remaining),
            "timeout": 60000,
        }
        if time_window:
            payload["timeWindow"] = time_window

        resp = httpx.post(
            f"{self.base_url}/v0/search",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120,
        )

        if resp.status_code == 429:
            raise FirecrawlError("Rate limited")
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("data", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                company=item.get("metadata", {}).get("organization", ""),
                posted_at=item.get("publishedDate"),
                source="firecrawl",
            ))
        return results

    @retry(
        retry=retry_if_exception_type(FirecrawlError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def scrape_json(
        self,
        url: str,
        schema: dict,
    ) -> dict[str, Any]:
        """Scrape a URL and extract structured JSON using Firecrawl's extraction."""
        self._ensure_budget()
        self._queries_used += 1

        payload = {
            "url": url,
            "formats": ["json"],
            "extractor": {"schema": schema},
            "timeout": 60000,
        }

        resp = httpx.post(
            f"{self.base_url}/v0/scrape",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120,
        )

        if resp.status_code == 429:
            raise FirecrawlError("Rate limited")
        resp.raise_for_status()
        data = resp.json()

        # Firecrawl returns extraction in various locations depending on version
        extract_data = data.get("data", {}).get("extracted", {})
        if not extract_data:
            extract_data = data.get("extract", {})
        return extract_data


# Module-level singleton (lazy)
_client: FirecrawlClient | None = None


def get_client() -> FirecrawlClient:
    """Return a FirecrawlClient, initializing if needed."""
    global _client
    if _client is None:
        _client = FirecrawlClient()
    return _client
