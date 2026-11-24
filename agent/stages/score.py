"""Rank scraped postings and write a shortlist for human review."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from sqlite3 import Connection

import httpx

from agent.config import get_settings
from agent.db import connect, jobs_by_status, save_embedding
from agent.embeddings import chunk_profile, cosine_similarity, embed
from agent.llm import LLMOutputError, chat_json


def _correct_score(result: dict, profile: str) -> None:
    matched = result["matched_skills"]
    profile_lower = profile.casefold()
    valid = [skill for skill in matched if skill.casefold() in profile_lower]
    breakdown = result["breakdown"]
    breakdown["skills"] = round(breakdown["skills"] * len(valid) / len(matched)) if matched else 0
    dealbreaker = result.get("dealbreaker")
    if isinstance(dealbreaker, str) and dealbreaker.strip().casefold() in {"", "none", "null"}:
        dealbreaker = None
    if dealbreaker is not None:
        breakdown = dict.fromkeys(breakdown, 0)
    result.update(matched_skills=valid, breakdown=breakdown, score=sum(breakdown.values()), dealbreaker=dealbreaker)


def load_rubric() -> tuple[str, dict]:
    """Read the prompt and output schema shared by pipeline and evaluation."""
    template = Path("prompts/score.md").read_text(encoding="utf-8")
    return template, json.loads(Path("schemas/score.json").read_text(encoding="utf-8"))


def score_job(
    detail: dict, profile: str, rubric: tuple[str, dict], *,
    conn: Connection | None = None, run_id: str = "", model: str | None = None,
) -> dict:
    """Apply the same model rubric and score corrections to one job."""
    template, schema = rubric
    prompt = template.replace("{profile_md}", profile).replace("{job_json}", json.dumps(detail))
    result = chat_json(prompt, schema, model=model, conn=conn, run_id=run_id)
    _correct_score(result, profile)
    return result


def _write_shortlist(jobs, path: Path) -> None:
    """Write a ranked table and details for each available scored posting."""
    def cell(value) -> str:
        return " ".join(str(value or "").split()).replace("|", "\\|")

    lines = [
        "# Ranked shortlist", "",
        "| Rank | Score | Title | Company | Location | Matched | Missing | Why | Link |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    details = []
    for rank, job in enumerate(sorted(jobs, key=lambda row: (-row["score"], row["id"])), 1):
        result = json.loads(job["score_json"] or "{}")
        score = job["score"]
        matched = ", ".join(result.get("matched_skills", []))
        missing = ", ".join(result.get("missing_skills", []))
        why = result.get("reasoning", "")
        values = (job["title"], job["company"], job["location"], matched, missing, why, job["url"])
        lines.append(f"| {rank} | {score} | " + " | ".join(map(cell, values)) + " |")
        details.extend([
            "", f"## {rank}. {cell(job['title'])} · {cell(job['company'])}", "",
            f"- Score: {score}",
            f"- Breakdown: {', '.join(f'{k} {v}' for k, v in result.get('breakdown', {}).items())}",
            f"- Matched: {cell(matched) or '—'}", f"- Missing: {cell(missing) or '—'}",
            f"- Reasoning: {cell(why) or '—'}",
            f"- Dealbreaker: {cell(result.get('dealbreaker')) or '—'}",
            f"- Posting: {cell(job['url'])}",
        ])
    if not jobs:
        lines.extend(["", "No postings scored."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + details) + "\n", encoding="utf-8")


def run(run_id: str = "", dry_run: bool = False, db_path=None, **kwargs) -> dict:
    """Score picked postings and retry prior scoring failures."""
    settings = get_settings()
    with closing(connect(db_path or settings.db_path)) as conn:
        jobs = jobs_by_status(conn, "scraped") + jobs_by_status(conn, "score_failed")
        if not jobs:
            return {"scored": 0, "rejected_embed": 0, "failed": 0, "total": 0}

        profile_path = Path("profile.md")
        profile = profile_path.read_text(encoding="utf-8") if profile_path.exists() else ""
        rubric = load_rubric()
        candidates = []
        failed = scored = rejected = 0

        for job in jobs:
            try:
                detail = json.loads(job["detail_json"] or "{}")
                if not isinstance(detail, dict):
                    raise TypeError("Expected job details as a JSON object")
                skills = detail.get("must_have_skills", [])
                duties = detail.get("responsibilities", [])
                text = " ".join([job["title"] or "", *skills, *duties])
            except (ValueError, TypeError) as exc:
                conn.execute(
                    "UPDATE jobs SET status = 'score_failed', status_reason = ? WHERE id = ?",
                    (str(exc), job["id"]),
                )
                failed += 1
            else:
                candidates.append((job, detail, text))
        conn.commit()

        profile_chunks = chunk_profile(profile)
        if len(candidates) >= 15 and profile_chunks and not dry_run:
            profile_vectors = embed(profile_chunks, conn=conn, run_id=run_id)
            vectors = embed([text for _, _, text in candidates], conn=conn, run_id=run_id)
            ranked = []
            for (job, detail, text), vector in zip(candidates, vectors, strict=True):
                similarities = sorted(
                    (cosine_similarity(vector, chunk) for chunk in profile_vectors), reverse=True,
                )[:3]
                similarity = sum(similarities) / len(similarities)
                save_embedding(conn, "job", job["id"], settings.embed_model, text, list(vector))
                conn.execute("UPDATE jobs SET embed_sim = ? WHERE id = ?", (similarity, job["id"]))
                ranked.append((similarity, job, detail, text))
            ranked.sort(key=lambda entry: (-entry[0], entry[1]["id"]))
            keep = max(1, int(len(ranked) * settings.scoring.embed_keep_fraction))
            candidates = [(job, detail, text) for _, job, detail, text in ranked[:keep]]
            rejected = len(ranked) - keep
            conn.executemany(
                "UPDATE jobs SET status = 'rejected', status_reason = 'low_similarity' WHERE id = ?",
                [(job["id"],) for _, job, _, _ in ranked[keep:]],
            )
            conn.commit()

        for job, detail, _ in candidates:
            try:
                result = {
                    "score": 60,
                    "breakdown": {"skills": 0, "seniority": 25, "domain": 20, "location": 15},
                    "matched_skills": [],
                    "missing_skills": [],
                    "seniority_fit": "fit",
                    "reasoning": "Offline fixture score",
                } if dry_run else score_job(detail, profile, rubric, conn=conn, run_id=run_id)
            except (LLMOutputError, httpx.HTTPError) as exc:
                conn.execute(
                    "UPDATE jobs SET status = 'score_failed', status_reason = ? WHERE id = ?",
                    (str(exc), job["id"]),
                )
                failed += 1
            else:
                conn.execute(
                    """UPDATE jobs SET score = ?, score_json = ?, status = 'scored',
                       status_reason = NULL WHERE id = ?""",
                    (result["score"], json.dumps(result), job["id"]),
                )
                scored += 1
            conn.commit()

        output = Path("out/dry-run/shortlist.md" if dry_run else "out/shortlist.md")
        _write_shortlist(jobs_by_status(conn, "scored"), output)
    return {"scored": scored, "rejected_embed": rejected, "failed": failed, "total": len(jobs)}
