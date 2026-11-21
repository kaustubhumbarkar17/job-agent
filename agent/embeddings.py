"""Embedding helpers using Ollama's nomic-embed-text model.

Provides chunking, embedding, and cosine similarity computation.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, List, Tuple

import numpy as np

import httpx
from agent.config import get_settings


def embed(texts: list[str], model: str | None = None) -> np.ndarray:
    """Batch-embed texts via Ollama. Returns (n, dim) array."""
    model_name = model or get_settings().embed_model
    resp = httpx.post(
        f"{get_settings().ollama_host}/api/embed",
        json={"model": model_name, "input": texts},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return np.array([e["embedding"] for e in data["embeddings"]], dtype=np.float32)


def chunk_profile(md: str, max_tokens: int = 150) -> list[str]:
    """Split a markdown profile into ~100-200 token chunks by heading and bullet."""
    lines = md.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # New top-level heading starts a new chunk
        if stripped.startswith("## ") and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        # Count tokens roughly by words
        tokens = len(stripped.split())
        if current_tokens + tokens > max_tokens and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(stripped)
        current_tokens += tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def ensure_profile_embeddings(
    db_conn: Any,
    profile_md: str,
    model: str | None = None,
) -> int:
    """Embed profile chunks that have changed. Returns number of new embeddings."""
    model_name = model or get_settings().embed_model
    chunks = chunk_profile(profile_md)
    ref_id = 0  # Profile is always ref_id=0
    new_count = 0
    for chunk in chunks:
        save_embedding(
            db_conn, "profile_chunk", ref_id, model_name, chunk,
            list(embed([chunk])[0]),
        )
        new_count += 1
    return new_count


def similarity(job_text: str, profile_matrix: np.ndarray) -> float:
    """Compute mean of top-3 cosine similarities against profile chunks."""
    job_vec = embed([job_text])[0]
    sims = [cosine_similarity(job_vec, pc) for pc in profile_matrix]
    sims.sort(reverse=True)
    return sum(sims[:3]) / min(3, len(sims)) if sims else 0.0


def save_embedding(
    conn: Any,
    kind: str,
    ref_id: int,
    model: str,
    text: str,
    vector: list[float],
) -> None:
    """Thin wrapper around agent.db.save_embedding."""
    from agent.db import save_embedding as _save
    _save(conn, kind, ref_id, model, text, vector)
