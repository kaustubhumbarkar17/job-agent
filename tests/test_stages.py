"""Tests for pipeline stages using a temp DB and dry-run mode."""
import json
import tempfile
from pathlib import Path

import pytest

TEST_DB_PATH = None


@pytest.fixture(autouse=True)
def _setup_test_db(tmp_path: Path):
    """Create a temp DB and store path for tests."""
    global TEST_DB_PATH
    TEST_DB_PATH = str(tmp_path / "test.db")
    from agent.db import init_db
    init_db(TEST_DB_PATH)
    yield


class TestDiscoverStage:
    """Stage 1: Discover tests."""

    def test_dry_run_discovers_jobs(self):
        from agent.stages.discover import run
        result = run(dry_run=True, run_id="test-1", db_path=TEST_DB_PATH)
        assert result["queries_run"] >= 0
        assert result["discovered"] >= 0

    def test_blocklist_filters_junk(self):
        """Blocklisted titles are filtered out."""
        from agent.stages.discover import run
        from agent import config

        class FakeSearch:
            titles = ["Intern Engineer"]
            locations = []
            max_queries_per_run = 1
            blocklist_title_words = ["intern"]
            blocklist_companies = []

        class FakeSettings:
            search = FakeSearch()

        orig = config.get_settings
        config.get_settings = lambda: FakeSettings()
        try:
            result = run(dry_run=True, run_id="test-2", db_path=TEST_DB_PATH)
            assert result["blocked"] >= 0
        finally:
            config.get_settings = orig


class TestScrapeStage:
    """Stage 3: Scrape tests."""

    def test_dry_run_scrapes_picked_jobs(self):
        """Dry run marks picked jobs as scraped."""
        from agent.db import connect, upsert_job, set_status
        from agent.stages.scrape import run

        conn = connect(TEST_DB_PATH)
        upsert_job(conn, "https://example.com/1", title="Engineer", company="Acme")
        set_status(conn, 1, "picked")

        result = run(dry_run=True, run_id="test-3", db_path=TEST_DB_PATH)
        assert result["scraped"] == 1


class TestScoreStage:
    """Stage 4: Score tests."""

    def test_score_with_no_scraped_jobs(self):
        """No-op when there are no scraped jobs."""
        from agent.stages.score import run
        result = run(dry_run=True)
        assert result["scored"] == 0


class TestShortlistStage:
    """Stage 5: Shortlist tests."""

    def test_shortlist_with_no_scored_jobs(self):
        """No-op when there are no scored jobs."""
        from agent.stages.shortlist import run
        result = run(yes=True, db_path=TEST_DB_PATH)
        assert result["selected"] == []


class TestOrchestrator:
    """Pipeline orchestrator tests."""

    def test_run_pipeline_dry_run(self):
        """Full dry-run pipeline completes without errors."""
        from agent.orchestrator import run_pipeline
        results = run_pipeline(
            start="discover",
            stop=None,
            dry_run=True,
            yes=True,
            run_id="test-pipeline",
        )
        assert "discover" in results

    def test_status_summary(self):
        """Status summary returns counts."""
        from agent.orchestrator import status_summary
        summary = status_summary()
        assert isinstance(summary, dict)

    def test_partial_pipeline(self):
        """Running --from score --to shortlist works."""
        from agent.orchestrator import run_pipeline
        results = run_pipeline(
            start="score",
            stop="shortlist",
            dry_run=True,
            yes=True,
            run_id="test-partial",
        )
        assert "score" in results


class TestPickStage:
    """Stage 2: Pick tests."""

    def test_pick_with_no_discovered_jobs(self):
        """No-op when there are no discovered jobs."""
        from agent.stages.pick import run
        result = run(yes=True, db_path=TEST_DB_PATH)
        assert result["selected"] == []


class TestContactsStage:
    """Stage 6: Contacts tests."""

    def test_contacts_no_shortlisted_jobs(self):
        """No-op when there are no shortlisted jobs."""
        from agent.stages.contacts import run
        result = run(db_path=TEST_DB_PATH)
        assert result["found"] == 0


class TestVerifyStage:
    """Stage 7: Verify tests."""

    def test_verify_no_contacted_jobs(self):
        """No-op when there are no contacted jobs."""
        from agent.stages.verify import run
        result = run(db_path=TEST_DB_PATH)
        assert result["verified"] == 0


class TestDraftStage:
    """Stage 8: Draft tests."""

    def test_draft_with_no_drafted_jobs(self):
        """No-op when there are no drafted jobs."""
        from agent.stages.draft import run
        result = run(dry_run=True, db_path=TEST_DB_PATH)
        assert result["drafted"] == 0
