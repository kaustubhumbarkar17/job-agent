"""Phase E: Database layer tests.

All tests use a temp DB file that is cleaned up after each test.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import sqlite3

from agent.db import (
    STATUS_TRANSITIONS,
    canonicalize_url,
    connect,
    init_db,
    jobs_by_status,
    load_embeddings,
    log_api_call,
    save_embedding,
    set_status,
    url_hash,
    upsert_job,
)


@pytest.fixture()
def tmp_db():
    """Create a temporary SQLite DB and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    init_db(path)
    yield path
    # Cleanup
    path.unlink(missing_ok=True)


class TestInitDb:
    """E3: init_db idempotent."""

    def test_init_twice_no_error(self, tmp_db):
        init_db(tmp_db)
        init_db(tmp_db)  # Should not raise

    def test_tables_exist(self, tmp_db):
        conn = connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {t["name"] for t in tables}
        expected = {"api_calls", "contacts", "drafts", "embeddings", "jobs", "runs"}
        assert expected.issubset(names)


class TestCanonicalizeAndHash:
    """URL canonicalization and hashing."""

    def test_trailing_slash_removed(self):
        assert canonicalize_url("https://example.com/") == "https://example.com"

    def test_tracking_params_stripped(self):
        url = "https://example.com/job?utm_source=linkedin&ref=test"
        result = canonicalize_url(url)
        assert "utm_" not in result
        assert "&ref=" not in result and "?ref=" not in result

    def test_same_url_different_trailing_slash_same_hash(self):
        h1 = url_hash("https://example.com/")
        h2 = url_hash("https://example.com")
        assert h1 == h2

    def test_same_url_with_utm_same_hash(self):
        h1 = url_hash("https://example.com/job?id=1")
        h2 = url_hash("https://example.com/job?id=1&utm_source=x")
        assert h1 == h2


class TestUpsertJob:
    """E3: Duplicate URLs collapse to one row."""

    def test_duplicate_url_collapse(self, tmp_db):
        conn = connect(tmp_db)
        id1 = upsert_job(conn, "https://example.com/job1", title="Engineer")
        id2 = upsert_job(conn, "https://example.com/job1/", title="Engineer")
        assert id1 == id2
        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert count == 1

    def test_different_urls_different_rows(self, tmp_db):
        conn = connect(tmp_db)
        upsert_job(conn, "https://example.com/a")
        upsert_job(conn, "https://example.com/b")
        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert count == 2


class TestStatusTransitions:
    """E3: Illegal transitions raise."""

    def test_valid_transition(self, tmp_db):
        conn = connect(tmp_db)
        upsert_job(conn, "https://example.com/job1", title="Engineer")
        set_status(conn, 1, "picked")  # discovered -> picked

    def test_illegal_transition(self, tmp_db):
        conn = connect(tmp_db)
        upsert_job(conn, "https://example.com/job1", title="Engineer")
        with pytest.raises(ValueError):
            set_status(conn, 1, "shortlisted")  # discovered -> shortlisted (invalid)


class TestEmbeddings:
    """E3: Embedding round-trip and text-hash guard."""

    def test_roundtrip(self, tmp_db):
        conn = connect(tmp_db)
        vec = [0.1, 0.2, -0.3, 0.4]
        save_embedding(conn, "profile_chunk", 1, "nomic-embed-text", "hello", vec)
        loaded = load_embeddings(conn, "profile_chunk", "nomic-embed-text")
        assert len(loaded) == 1
        assert loaded[0][0] == 1  # ref_id
        np.testing.assert_allclose(loaded[0][1], vec, rtol=1e-3)

    def test_text_hash_guard(self, tmp_db):
        """Same text -> no new row. Different text -> replaces old."""
        conn = connect(tmp_db)
        vec1 = [0.1, 0.2]
        save_embedding(conn, "profile_chunk", 1, "nomic", "hello", vec1)
        # Same text again should not create a new row
        save_embedding(conn, "profile_chunk", 1, "nomic", "hello", vec1)
        assert len(load_embeddings(conn, "profile_chunk", "nomic")) == 1
        # Different text: load with text="hello" should return nothing
        loaded = load_embeddings(conn, "profile_chunk", "nomic", text="world")
        assert len(loaded) == 0


class TestDraftsUniqueness:
    """E3: UNIQUE(job_id, contact_id) on drafts blocks duplicates."""

    def test_duplicate_draft_blocked(self, tmp_db):
        conn = connect(tmp_db)
        # Insert a job and a contact first
        conn.execute("INSERT INTO jobs (url_hash, url, status) VALUES (?, ?, ?)",
                      ("abc", "https://example.com", "shortlisted"))
        conn.execute("INSERT INTO contacts (job_id, company, name, role) VALUES (?, ?, ?, ?)",
                      (1, "Acme", "John", "Recruiter"))
        conn.commit()
        # First draft succeeds
        conn.execute(
            "INSERT INTO drafts (job_id, contact_id, subject, body) VALUES (?, ?, ?, ?)",
            (1, 1, "Subject", "Body"),
        )
        conn.commit()
        # Duplicate should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO drafts (job_id, contact_id, subject, body) VALUES (?, ?, ?, ?)",
                (1, 1, "Subject2", "Body2"),
            )
            conn.commit()


class TestJobsByStatus:
    """Basic query test."""

    def test_returns_correct_jobs(self, tmp_db):
        conn = connect(tmp_db)
        upsert_job(conn, "https://example.com/a", title="Job A")
        upsert_job(conn, "https://example.com/b", title="Job B")
        set_status(conn, 2, "picked")
        picked = jobs_by_status(conn, "picked")
        assert len(picked) == 1
        assert picked[0]["title"] == "Job B"


class TestStatusTransitionsDefined:
    """Verify STATUS_TRANSITIONS has all expected keys."""

    def test_all_transitions_present(self):
        assert "discovered" in STATUS_TRANSITIONS
        assert "picked" in STATUS_TRANSITIONS
        assert "scraped" in STATUS_TRANSITIONS
        assert "scored" in STATUS_TRANSITIONS
        assert "shortlisted" in STATUS_TRANSITIONS
        assert "contacted" in STATUS_TRANSITIONS


class TestLogApiCall:
    """Basic API call logging."""

    def test_log_and_retrieve(self, tmp_db):
        conn = connect(tmp_db)
        log_api_call(conn, "firecrawl", "/search", credits=1.0, latency_ms=200, ok=True)
        count = conn.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
        assert count == 1
