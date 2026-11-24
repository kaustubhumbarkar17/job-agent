"""Firecrawl search and structured scraping with request accounting."""
from __future__ import annotations

from sqlite3 import Connection
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import httpx
from jsonschema import ValidationError as SchemaValidationError
from jsonschema import validate
from pydantic import BaseModel, ValidationError, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent.config import get_settings
from agent.db import log_api_call


class FirecrawlError(Exception):
    """A provider request or response failed."""


class _TransientError(FirecrawlError):
    """A rate limit, server error, or connection failure can be retried."""


class BudgetExceeded(Exception):
    """The configured search request limit has been reached."""


class SearchResult(BaseModel):
    title: str
    url: str
    company: str | None = ""
    location: str | None = ""
    source: str = "firecrawl"
    posted_at: str | None = None

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Expected an HTTP(S) posting URL")
        return value

    @field_validator("title")
    @classmethod
    def valid_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Expected a posting title")
        return value.strip()


class FirecrawlClient:
    """Thin wrapper around Firecrawl APIs with retry and budget tracking."""

    def __init__(
        self, api_key: str | None = None, *, conn: Connection | None = None, run_id: str = "",
    ):
        settings = get_settings()
        custom_url = settings.firecrawl_base_url.strip()
        self.base_url = (custom_url or "https://api.firecrawl.dev").rstrip("/")
        try:
            url = urlsplit(self.base_url)
            if (url.scheme not in {"http", "https"} or not url.hostname or url.username is not None
                    or url.query or url.fragment):
                raise ValueError
        except ValueError:
            raise FirecrawlError("FIRECRAWL_BASE_URL must be an HTTP(S) base URL without credentials, query, or fragment") from None
        cloud = url.hostname == "api.firecrawl.dev"
        if cloud and url.scheme != "https":
            raise FirecrawlError("Firecrawl Cloud requires HTTPS")
        configured_key = settings.firecrawl_api_key if cloud else settings.firecrawl_self_hosted_api_key
        self.api_key = api_key or configured_key
        if cloud and not self.api_key:
            raise FirecrawlError("FIRECRAWL_API_KEY is not set")
        self._conn = conn
        self._run_id = run_id
        self._queries_used = 0
        if conn is not None and run_id:
            self._queries_used = conn.execute(
                "SELECT COUNT(*) FROM api_calls WHERE service = ? AND endpoint = ? AND run_id = ?",
                ("firecrawl", "/v2/search", run_id),
            ).fetchone()[0]

    @property
    def queries_remaining(self) -> int:
        return get_settings().search.max_queries_per_run - self._queries_used

    @retry(
        retry=retry_if_exception_type(_TransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
        reraise=True,
    )
    def _request(self, endpoint: str, payload: dict, schema: dict) -> dict:
        if endpoint == "/v2/search":
            if self.queries_remaining <= 0:
                raise BudgetExceeded("Firecrawl search request budget exhausted")
            self._queries_used += 1

        started = perf_counter()
        ok = False
        credits = 0.0
        try:
            response = httpx.post(
                f"{self.base_url}{endpoint}", json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                timeout=120,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise _TransientError(f"Firecrawl returned HTTP {response.status_code}")
            response.raise_for_status()
            envelope = response.json()
            if not isinstance(envelope, dict) or envelope.get("success") is not True:
                raise FirecrawlError(f"{endpoint} returned an unsuccessful response")
            data = envelope.get("data")
            metadata = data.get("metadata") if isinstance(data, dict) else None
            reported = envelope.get("creditsUsed")
            if reported is None and isinstance(metadata, dict):
                reported = metadata.get("creditsUsed")
            if isinstance(reported, int | float) and not isinstance(reported, bool):
                credits = reported
            validate(data, schema)
            ok = True
            return data
        except httpx.RequestError as exc:
            raise _TransientError(f"Firecrawl request failed for {endpoint}") from exc
        except (httpx.HTTPStatusError, ValueError, SchemaValidationError) as exc:
            raise FirecrawlError(f"Invalid Firecrawl response for {endpoint}: {exc.message if isinstance(exc, SchemaValidationError) else exc}") from exc
        finally:
            if self._conn is not None:
                log_api_call(
                    self._conn, "firecrawl", endpoint, credits,
                    int((perf_counter() - started) * 1000), ok, self._run_id,
                )

    def search(
        self, query: str, limit: int = 10, time_window: str | None = None,
    ) -> list[SearchResult]:
        """Search the web for job postings and skip malformed individual hits."""
        payload = {"query": query, "limit": limit, "timeout": 60000}
        if time_window:
            payload["tbs"] = {"1d": "qdr:d", "7d": "qdr:w"}[time_window]
        data = self._request("/v2/search", payload, {
            "type": "object",
            "properties": {"web": {"type": "array"}},
        })
        results = []
        for item in data.get("web", []):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            try:
                results.append(SearchResult(
                    title=item.get("title"),
                    url=item.get("url"),
                    company=metadata.get("organization", ""),
                    posted_at=item.get("publishedDate"),
                ))
            except ValidationError:
                continue
        return results

    def scrape_json(self, url: str, schema: dict) -> dict[str, Any]:
        """Scrape a posting and validate the extracted JSON."""
        data = self._request(
            "/v2/scrape",
            {"url": url, "formats": [{"type": "json", "schema": schema}], "timeout": 60000},
            {"type": "object", "required": ["json"], "properties": {"json": schema}},
        )
        return data["json"]


_client: FirecrawlClient | None = None


def get_client() -> FirecrawlClient:
    """Return a FirecrawlClient, initializing if needed."""
    global _client
    if _client is None:
        _client = FirecrawlClient()
    return _client
