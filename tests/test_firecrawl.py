"""Firecrawl client tests using the documented v2 response structure."""
from contextlib import closing
from unittest.mock import patch

import httpx
import pytest

from agent.config import Settings
from agent.db import connect
from agent.firecrawl_client import BudgetExceeded, FirecrawlClient, FirecrawlError


def response(status, body):
    return httpx.Response(status, json=body, request=httpx.Request("POST", "https://api.firecrawl.dev"))


def test_search_validates_hits_and_sends_requested_limit():
    client = FirecrawlClient(api_key="test-key")
    with patch("agent.firecrawl_client.httpx.post") as post:
        post.return_value = response(200, {
            "success": True,
            "data": {"web": [
                {"title": "Engineer", "url": "https://example.com/1",
                 "metadata": {"organization": "Acme"}},
                {"title": "Invalid", "url": "missing-scheme"},
            ]},
        })
        results = client.search("engineer jobs", limit=10, time_window="1d")

    assert [(job.title, job.company) for job in results] == [("Engineer", "Acme")]
    assert post.call_args.args[0].endswith("/v2/search")
    assert post.call_args.kwargs["json"]["limit"] == 10
    assert post.call_args.kwargs["json"]["tbs"] == "qdr:d"


def test_search_retries_and_budget_carries_to_next_client(tmp_path, monkeypatch):
    settings = Settings()
    settings.search.max_queries_per_run = 2
    monkeypatch.setattr("agent.firecrawl_client.get_settings", lambda: settings)
    monkeypatch.setattr(FirecrawlClient._request.retry, "sleep", lambda _: None)
    with closing(connect(tmp_path / "audit.db")) as conn, patch(
        "agent.firecrawl_client.httpx.post",
        side_effect=[
            response(429, {}),
            response(200, {"success": True, "data": {"web": []}, "creditsUsed": 2}),
        ],
    ) as post:
        assert FirecrawlClient("test-key", conn=conn, run_id="r1").search("engineer") == []
        with pytest.raises(BudgetExceeded):
            FirecrawlClient("test-key", conn=conn, run_id="r1").search("blocked")
        assert post.call_count == 2
        calls = conn.execute(
            "SELECT endpoint, credits, ok, run_id FROM api_calls ORDER BY id",
        ).fetchall()
        assert [tuple(call) for call in calls] == [
            ("/v2/search", 0, 0, "r1"), ("/v2/search", 2, 1, "r1"),
        ]


def test_scrape_accepts_valid_json_and_rejects_schema_mismatch():
    client = FirecrawlClient(api_key="test-key")
    schema = {"type": "object", "required": ["title"], "properties": {"title": {"type": "string"}}}
    with patch("agent.firecrawl_client.httpx.post") as post:
        post.side_effect = [
            response(200, {"success": True, "data": {"json": {"title": "Job"}}}),
            response(200, {"success": True, "data": {"json": {"title": 7}}}),
        ]
        assert client.scrape_json("https://example.com/job", schema) == {"title": "Job"}
        with pytest.raises(FirecrawlError, match="Invalid Firecrawl response"):
            client.scrape_json("https://example.com/job", schema)
        assert post.call_args.kwargs["json"]["formats"] == [{"type": "json", "schema": schema}]
        assert post.call_count == 2
