"""Phase D: Environment smoke tests.

These require only local Ollama to be running. No external network calls.
"""
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

BASE = Path(__file__).resolve().parent.parent
OLLAMA_HOST = "http://localhost:11434"


def _ollama_chat(prompt: str, model: str = "qwen3:14b") -> dict:
    """Make a raw Ollama /api/chat call and return the parsed response."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        "options": {"temperature": 0, "num_ctx": 8192},
        "stream": False,
    }
    resp = httpx.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _ollama_embeddings(texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
    """Make a raw Ollama /api/embed call. Returns flat list of vectors."""
    payload = {"model": model, "input": texts}
    resp = httpx.post(f"{OLLAMA_HOST}/api/embed", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"]


class TestOllamaGeneration:
    """D1: Ollama returns schema-valid JSON."""

    @pytest.mark.live_local
    def test_json_response(self):
        result = _ollama_chat('Reply with only {"ok": true}')
        assert "message" in result
        content = result["message"]["content"].strip()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
        assert parsed.get("ok") is True


class TestEmbeddings:
    """D2: Embeddings work and similar sentences are closer."""

    @pytest.mark.live_local
    def test_embedding_cosine(self):
        texts = [
            "The system processes requests quickly.",
            "The system handles requests fast.",
            "Python is a programming language.",
        ]
        vectors = _ollama_embeddings(texts)
        assert len(vectors) == 3
        assert len(set(len(v) for v in vectors)) == 1, "All vectors same length"

        def cosine(a, b):
            av, bv = np.array(a), np.array(b)
            return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv)))

        sim_same = cosine(vectors[0], vectors[1])
        sim_diff = cosine(vectors[0], vectors[2])
        assert sim_same > sim_diff, f"Similar > unrelated: {sim_same:.3f} vs {sim_diff:.3f}"


class TestLatency:
    """D3: Record latency baseline for a scoring-sized prompt."""

    @pytest.mark.live_local
    def test_scoring_prompt_latency(self):
        start = time.perf_counter()
        _ollama_chat(
            "What is 2+2? Reply {\"answer\": 4}",
            model="qwen3:14b",
        )
        elapsed = time.perf_counter() - start
        # Just record; accept any value for now.
        assert elapsed < 120, f"Latency {elapsed:.1f}s exceeded budget"


class TestConfigLoads:
    """D4: config.yaml loads correctly."""

    def test_config_loads(self):
        cfg_path = BASE / "config.yaml"
        assert cfg_path.exists()
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        assert "search" in data
        assert "scoring" in data
        assert "contacts" in data
        assert "email" in data
        assert len(data["search"]["titles"]) > 0
