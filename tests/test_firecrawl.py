"""Tests for agent/firecrawl_client.py with mocked HTTP."""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.firecrawl_client import FirecrawlClient, BudgetExceeded


class TestFirecrawlSearch:
    """Firecrawl search tests."""

    @pytest.fixture()
    def client(self):
        return FirecrawlClient(api_key="test-key")

    def test_search_returns_results(self, client):
        with patch("agent.firecrawl_client.httpx.post") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "data": [
                        {"title": "Engineer", "url": "https://example.com/1",
                         "metadata": {"organization": "Acme"},
                         "publishedDate": "2024-01-01"}
                    ]
                },
                raise_for_status=MagicMock(),
            )
            results = client.search("engineer jobs")
            assert len(results) == 1
            assert results[0].title == "Engineer"
            assert results[0].company == "Acme"

    def test_budget_enforced(self, client):
        """Exceeding max_queries_per_run raises BudgetExceeded."""
        class FakeSearch:
            max_queries_per_run = 0
        class FakeSettings:
            search = FakeSearch()
        with patch("agent.firecrawl_client.get_settings", return_value=FakeSettings()):
            with pytest.raises(BudgetExceeded):
                client.search("test")


class TestFirecrawlScrape:
    """Firecrawl scrape tests."""

    @pytest.fixture()
    def client(self):
        return FirecrawlClient(api_key="test-key")

    def test_scrape_returns_extracted_data(self, client):
        with patch("agent.firecrawl_client.httpx.post") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {"data": {"extracted": {"title": "Job", "company": "Acme"}}},
                raise_for_status=MagicMock(),
            )
            result = client.scrape_json(
                "https://example.com/job",
                {"type": "object", "properties": {"title": {"type": "string"}}}
            )
            assert result["title"] == "Job"
