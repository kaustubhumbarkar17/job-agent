"""Tests for pipeline stages using a temp DB and dry-run mode."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline_pipeline")

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

    def test_blocklist_filters_junk(self, monkeypatch):
        """Blocklisted titles are filtered out."""
        from agent import config
        from agent.firecrawl_client import SearchResult
        from agent.stages import discover

        config.get_settings().search.max_queries_per_run = 1

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def search(self, query, limit, time_window):
                assert time_window == "1d"
                assert "last 1 day" in query
                return [SearchResult(url="https://example.com/intern", title="Intern Engineer")]

        monkeypatch.setattr(discover, "FirecrawlClient", FakeClient)
        result = discover.run(run_id="test-2", db_path=TEST_DB_PATH, window="1d")
        assert result["blocked"] == 1
        assert result["discovered"] == 0


class TestScrapeStage:
    """Stage 3: Scrape tests."""

    def test_dry_run_scrapes_picked_jobs(self):
        """Dry run marks picked jobs as scraped."""
        from agent.db import connect, set_status, upsert_job
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


def test_discovery_counts_only_new_canonical_urls(monkeypatch):
    from agent.config import get_settings
    from agent.firecrawl_client import SearchResult
    from agent.stages import discover

    get_settings().search.max_queries_per_run = 2

    class FakeClient:
        def __init__(self, **kwargs):
            self.query_count = 0

        def search(self, query, limit, time_window):
            self.query_count += 1
            suffix = "/" if self.query_count == 1 else "?utm_source=duplicate"
            return [SearchResult(url=f"https://example.com/job{suffix}", title="Engineer")]

    monkeypatch.setattr(discover, "FirecrawlClient", FakeClient)
    first = discover.run(db_path=TEST_DB_PATH)
    second = discover.run(db_path=TEST_DB_PATH)
    assert (first["queries_run"], first["discovered"], second["discovered"]) == (2, 1, 0)


def test_scrape_retries_only_failed_posting_through_pipeline(monkeypatch):
    from contextlib import closing

    from agent.db import connect, jobs_by_status, set_status, upsert_job
    from agent.orchestrator import run_pipeline
    from agent.stages import scrape

    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def scrape_json(self, url, schema):
            calls.append(url)
            if url.endswith("/bad") and calls.count(url) == 1:
                return {"title": "Invalid", "url": url, "must_have_skills": "Python"}
            return {"title": "Valid", "url": url}

    monkeypatch.setattr(scrape, "FirecrawlClient", FakeClient)
    with closing(connect(TEST_DB_PATH)) as conn:
        for slug in ("bad", "good"):
            job_id = upsert_job(conn, f"https://example.com/{slug}")
            set_status(conn, job_id, "picked")

        options = {"start": "scrape", "stop": "scrape", "db_path": TEST_DB_PATH, "yes": True}
        first = run_pipeline(**options)["scrape"]
        assert (first["scraped"], first["failed"]) == (1, 1)
        assert len(jobs_by_status(conn, "scrape_failed")) == 1

        second = run_pipeline(**options)["scrape"]
        assert (second["scraped"], second["failed"]) == (1, 0)
        assert len(jobs_by_status(conn, "scraped")) == 2
        assert calls == ["https://example.com/bad", "https://example.com/good", "https://example.com/bad"]


def test_score_corrections_report_ranking_and_checkpoint_defaults(monkeypatch):
    import json
    from contextlib import closing
    from unittest.mock import Mock

    from agent.config import get_settings
    from agent.db import connect, upsert_job
    from agent.stages import score, shortlist

    Path("profile.md").write_text("I use Python.", encoding="utf-8")
    get_settings().scoring.min_score_for_shortlist = 35
    with closing(connect(TEST_DB_PATH)) as conn:
        for name in ("Low", "High", "Blocked"):
            job_id = upsert_job(conn, f"https://example.com/{name.lower()}", title=name)
            conn.execute(
                "UPDATE jobs SET status = 'scraped', detail_json = ? WHERE id = ?",
                (json.dumps({"title": name}), job_id),
            )
        conn.commit()

        def model(prompt, schema, **kwargs):
            scores = {"High": 40, "Low": 20, "Blocked": 40}
            title = next(name for name in scores if f'"title": "{name}"' in prompt)
            return {
                "score": 99,
                "breakdown": {"skills": scores[title], "seniority": 10, "domain": 5, "location": 0},
                "matched_skills": ["Python", "Clojure"],
                "missing_skills": ["SQL"],
                "reasoning": "Relevant work",
                "dealbreaker": "Blocked role" if title == "Blocked" else " False ",
            }

        embedding = Mock(side_effect=AssertionError("small batches must skip embeddings"))
        monkeypatch.setattr(score, "embed", embedding)
        monkeypatch.setattr(score, "chat_json", model)
        assert score.run(db_path=TEST_DB_PATH)["scored"] == 3
        embedding.assert_not_called()
        rows = conn.execute("SELECT title, score, score_json FROM jobs ORDER BY score DESC").fetchall()
        assert [(row["title"], row["score"]) for row in rows] == [
            ("High", 35), ("Low", 25), ("Blocked", 0),
        ]
        assert json.loads(rows[0]["score_json"])["matched_skills"] == ["Python"]

        report = Path("out/shortlist.md").read_text(encoding="utf-8")
        assert report.index("| 1 | 35 | High |") < report.index("| 2 | 25 | Low |")
        assert "Relevant work" in report
        assert "Clojure" not in report

        checkpoint = Mock()
        checkpoint.select.return_value = []
        shortlist.run(db_path=TEST_DB_PATH, checkpoint=checkpoint)
        assert checkpoint.select.call_args.kwargs["preselected"] == [2]


def test_embedding_filter_ranks_only_current_batch(monkeypatch):
    from contextlib import closing
    from unittest.mock import Mock

    from agent.config import get_settings
    from agent.db import connect, upsert_job
    from agent.stages import score

    profile = "# Skills\n- Python"
    Path("profile.md").write_text(profile, encoding="utf-8")
    get_settings().scoring.embed_keep_fraction = 0.4
    with closing(connect(TEST_DB_PATH)) as conn:
        for index in range(15):
            job_id = upsert_job(conn, f"https://example.com/job-{index}", title=f"Job {index}")
            conn.execute("UPDATE jobs SET status = 'scraped' WHERE id = ?", (job_id,))
        conn.commit()

        def vectors(texts, **kwargs):
            return [[1, 0]] if len(texts) == 1 else [
                [1, 0] if int(text.split()[-1]) >= 9 else [0, 1] for text in texts
            ]

        embedding = Mock(side_effect=vectors)
        monkeypatch.setattr(score, "embed", embedding)
        result = score.run(db_path=TEST_DB_PATH)
        assert (result["scored"], result["rejected_embed"]) == (6, 9)
        assert embedding.call_count == 2
        ranked = conn.execute(
            "SELECT title, embed_sim FROM jobs WHERE status = 'scored' ORDER BY id",
        ).fetchall()
        assert [job["title"] for job in ranked] == [f"Job {index}" for index in range(9, 15)]
        assert [job["embed_sim"] for job in ranked] == pytest.approx([1.0] * 6)


def test_cli_dry_run_reaches_ranked_shortlist_without_provider_calls(monkeypatch):
    from unittest.mock import Mock

    from typer.testing import CliRunner

    from agent.cli import app
    from agent.stages import score

    model = Mock(side_effect=AssertionError("dry-run must not call Ollama"))
    provider = Mock(side_effect=AssertionError("dry-run must not call Firecrawl"))
    monkeypatch.setattr(score, "embed", model)
    monkeypatch.setattr(score, "chat_json", model)
    monkeypatch.setattr("agent.firecrawl_client.httpx.post", provider)
    result = CliRunner().invoke(app, ["run", "--dry-run", "--yes", "--to", "shortlist"])
    assert result.exit_code == 0, result.output
    path = Path("out/dry-run/shortlist.md")
    assert "Offline fixture score" in path.read_text(encoding="utf-8")
    model.assert_not_called()
    provider.assert_not_called()
