"""Tests for agent/embeddings.py."""
import numpy as np

from agent.embeddings import chunk_profile, cosine_similarity


class TestChunkProfile:
    """Embedding chunking tests."""

    def test_chunking_deterministic(self):
        md = """# Profile\n## Skills\n- Python\n- React\n## Projects\n- Project A"""
        chunks = chunk_profile(md)
        assert len(chunks) >= 1
        # Same input -> same output
        assert chunks == chunk_profile(md)

    def test_headings_split_chunks(self):
        md = """# Profile\n## Section 1\nHello\n## Section 2\nWorld"""
        chunks = chunk_profile(md)
        # Should have at least 2 chunks (one per section)
        assert len(chunks) >= 2

    def test_bullet_groups(self):
        md = """# Profile\n## Skills\n- Python\n- React\n- Node"""
        chunks = chunk_profile(md)
        assert any("Python" in c for c in chunks)


class TestCosineSimilarity:
    """Cosine similarity tests."""

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_negative_similarity(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert sim < -0.9


def test_embed_records_batched_request(tmp_path, monkeypatch):
    from contextlib import closing
    from unittest.mock import Mock

    import httpx

    from agent.db import connect
    from agent.embeddings import embed

    response = httpx.Response(
        200, json={"embeddings": [[1, 0], [0, 1]]},
        request=httpx.Request("POST", "http://localhost:11434/api/embed"),
    )
    monkeypatch.setattr("agent.embeddings.httpx.post", Mock(return_value=response))
    with closing(connect(tmp_path / "jobs.db")) as conn:
        result = embed(["one", "two"], conn=conn, run_id="fixture")
        assert result.shape == (2, 2)
        call = conn.execute("SELECT service, endpoint, ok, run_id FROM api_calls").fetchone()
        assert tuple(call) == ("ollama", "/api/embed", 1, "fixture")
