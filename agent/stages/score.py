"""Stage 4: Score — rank scraped jobs against the profile.

Two-pass scoring:
  Pass 1: Embedding pre-filter (keep top embed_keep_fraction by similarity)
  Pass 2: LLM rubric scoring (one call per surviving job)

Post-checks in code:
  - Recompute score as sum of breakdown (LLMs are bad at arithmetic)
  - Drop hallucinated matched_skills not in profile.md, reduce skills subscore proportionally
  - If dealbreaker is non-null, force score to 0
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from agent.config import get_settings
from agent.db import (
    connect,
    jobs_by_status,
    load_embeddings,
    save_embedding,
    set_status,
)
from agent.embeddings import chunk_profile, embed, cosine_similarity, similarity
from agent.llm import chat_json, LLMOutputError


SCORE_SCHEMA = {
    "type": "object",
    "required": ["score", "breakdown", "matched_skills", "missing_skills", "seniority_fit", "reasoning"],
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "breakdown": {
            "type": "object",
            "required": ["skills", "seniority", "domain", "location"],
            "properties": {
                "skills": {"type": "integer"},
                "seniority": {"type": "integer"},
                "domain": {"type": "integer"},
                "location": {"type": "integer"},
            },
        },
        "matched_skills": {"type": "array", "items": {"type": "string"}},
        "missing_skills": {"type": "array", "items": {"type": "string"}},
        "seniority_fit": {"type": "string", "enum": ["under", "fit", "over"]},
        "dealbreaker": {"type": ["string", "null"]},
        "reasoning": {"type": "string", "maxLength": 200},
    },
}


def _load_profile(conn: Any) -> str:
    """Load profile.md from disk."""
    try:
        return open("profile.md").read()
    except FileNotFoundError:
        return ""


def _filter_hallucinated(matched_skills: list[str], profile_md: str) -> tuple[list[str], int, int]:
    """Filter matched_skills against profile.md.

    Returns (valid_skills, valid_count, total_count).
    Applies proportional reduction for hallucinated skills.
    """
    profile_lower = profile_md.lower()
    valid = [s for s in matched_skills if s.lower() in profile_lower]
    return valid, len(valid), len(matched_skills)


def _recompute_score(breakdown: dict[str, int], matched_skills: list[str],
                     profile_md: str, dealbreaker: Optional[str]) -> tuple[dict[str, int], int]:
    """Recompute score from breakdown, applying hallucination filter.

    Returns (corrected_breakdown, final_score).
    """
    valid_skills, valid_count, total_count = _filter_hallucinated(matched_skills, profile_md)

    corrected = dict(breakdown)
    if total_count > 0:
        original_skills = breakdown.get("skills", 0)
        corrected["skills"] = round(original_skills * (valid_count / total_count))

    score = sum(corrected.values())
    if dealbreaker:
        score = 0
        for k in corrected:
            corrected[k] = 0

    return corrected, score


def run(run_id: str = "", dry_run: bool = False) -> dict:
    """Run the scoring stage. Returns scored job IDs and rejected count."""
    settings = get_settings()
    conn = connect(settings.db_path)
    profile_md = _load_profile(conn)

    scraped = jobs_by_status(conn, "scraped")
    if not scraped:
        return {"scored": 0, "rejected_embed": 0, "failed": 0}

    # Pass 1: Embedding pre-filter
    if len(scraped) >= 15:
        # Build job texts and embed
        job_texts = []
        job_ids = []
        for job in scraped:
            detail = json.loads(job["detail_json"]) if job["detail_json"] else {}
            text = f"{job['title']} {' '.join(detail.get('must_have_skills', []))} {' '.join(detail.get('responsibilities', []))}"
            job_texts.append(text)
            job_ids.append(job["id"])

        # Embed profile
        profile_chunks = chunk_profile(profile_md)
        profile_matrix = np.array(embed(profile_chunks), dtype=np.float32) if profile_chunks else np.array([])

        for i, text in enumerate(job_texts):
            sim = similarity(text, profile_matrix) if len(profile_matrix) > 0 else 0.0
            save_embedding(conn, "job", job_ids[i], settings.embed_model, text, list(embed([text])[0]))

        # Keep top embed_keep_fraction
        embeddings = load_embeddings(conn, "job", settings.embed_model)
        sims = [(ref_id, similarity(
            next((t for t, jid in zip(job_texts, job_ids) if jid == ref_id), ""),
            profile_matrix
        )) for ref_id, _ in embeddings]
        sims.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(1, int(len(sims) * settings.scoring.embed_keep_fraction))
        kept_ids = {s[0] for s in sims[:keep_count]}

        for job in scraped:
            if job["id"] not in kept_ids:
                set_status(conn, job["id"], "rejected", reason="low_similarity")
    else:
        # Skip embedding filter for < 15 jobs; embed all
        for job in scraped:
            detail = json.loads(job["detail_json"]) if job["detail_json"] else {}
            text = f"{job['title']} {' '.join(detail.get('must_have_skills', []))}"
            vec = embed([text])
            save_embedding(conn, "job", job["id"], settings.embed_model, text, list(vec[0]))

    # Pass 2: LLM rubric scoring
    scored = 0
    failed = 0
    profile_md = _load_profile()

    for job in jobs_by_status(conn, "scraped"):
        detail = json.loads(job["detail_json"]) if job["detail_json"] else {}
        job_json = json.dumps(detail)
        prompt_template = open("prompts/score.md").read()
        prompt = prompt_template.replace("{profile_md}", profile_md).replace("{job_json}", job_json)

        try:
            result = chat_json(prompt, SCORE_SCHEMA)
            breakdown = result.get("breakdown", {})
            matched = result.get("matched_skills", [])
            dealbreaker = result.get("dealbreaker")

            corrected, final_score = _recompute_score(
                breakdown, matched, profile_md, dealbreaker
            )

            result["breakdown"] = corrected
            result["score"] = final_score

            conn.execute(
                "UPDATE jobs SET score = ?, score_json = ?, status = ? WHERE id = ?",
                (final_score, json.dumps(result), "scored"),
            )
            scored += 1
        except (LLMOutputError, Exception) as e:
            conn.execute(
                "UPDATE jobs SET status = ?, status_reason = ? WHERE id = ?",
                ("score_failed", str(e), job["id"]),
            )
            failed += 1

    conn.commit()
    return {"scored": scored, "failed": failed, "total": len(scraped)}
