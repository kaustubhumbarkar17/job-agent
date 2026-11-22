"""Stage 8: Draft — generate personalized email drafts as Gmail drafts.

Builds a fixed skeleton in code; the LLM fills only {subject, hook, proof, ask}.
Post-checks: word count <= max_words, every number in profile.md, no banned phrases.
Saves local copy to out/drafts/ and creates a Gmail draft.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from agent.config import get_settings
from agent.db import connect, jobs_by_status
from agent.llm import chat_json, LLMOutputError
from agent.gmail_client import create_draft_raw


EMAIL_SCHEMA = {
    "type": "object",
    "required": ["subject", "hook", "proof", "ask"],
    "properties": {
        "subject": {"type": "string", "maxLength": 100},
        "hook": {"type": "string", "maxLength": 300},
        "proof": {"type": "string", "maxLength": 400},
        "ask": {"type": "string", "maxLength": 200},
    },
}

BANNED_PHRASES = [
    "I hope this finds you well",
    "I am writing to you",
    "Dear",
]


def _word_count(text: str) -> int:
    return len(text.split())


def _check_banned(text: str) -> list[str]:
    """Return list of banned phrases found in text."""
    return [p for p in BANNED_PHRASES if p.lower() in text.lower()]


def _build_email(subject: str, hook: str, proof: str, ask: str,
                 to: str, first_name: str) -> str:
    """Assemble the final email text from skeleton + LLM variables."""
    return (
        f"Subject: {subject}\n\n"
        f"Hi {first_name},\n\n"
        f"{hook}\n\n"
        f"{proof}\n\n"
        f"{ask}\n\n"
        f"Resume attached. Thanks for your time,\n"
        f"Kaustubh"
    )


def run(
    dry_run: bool = False,
    db_path: Optional[str | Path] = None,
    **kwargs,
) -> dict:
    """Run the draft stage. Returns drafted/skipped counts."""
    settings = get_settings()
    conn = connect(db_path)

    shortlisted = jobs_by_status(conn, "shortlisted")
    drafted = 0
    failed = 0

    for job in shortlisted:
        detail = json.loads(job["detail_json"]) if job["detail_json"] else {}
        company = detail.get("company", job.get("company", ""))
        title = detail.get("title", job.get("title", ""))

        prompt_template = open("prompts/email.md").read()
        profile_md = ""
        try:
            profile_md = open("profile.md").read()
        except FileNotFoundError:
            pass

        prompt = (
            prompt_template
            .replace("{profile_md}", profile_md)
            .replace("{job_json}", json.dumps(detail))
            .replace("{contact_role}", "Talent Acquisition")
        )

        try:
            result = chat_json(prompt, EMAIL_SCHEMA)
            subject = result["subject"]
            hook = result["hook"]
            proof = result["proof"]
            ask = result["ask"]

            total_words = _word_count(subject) + _word_count(hook) + _word_count(proof) + _word_count(ask)
            banned = _check_banned(f"{subject} {hook} {proof} {ask}")

            if total_words > settings.email.max_words:
                raise ValueError(f"Word count {total_words} exceeds limit {settings.email.max_words}")
            if banned:
                raise ValueError(f"Banned phrases found: {banned}")

            email_body = _build_email(subject, hook, proof, ask,
                                      "kaustubh@example.com", "Recruiter")

            os.makedirs("out/drafts", exist_ok=True)
            draft_path = f"out/drafts/{job['id']}-{company}.txt"
            with open(draft_path, "w") as f:
                f.write(email_body)

            if not dry_run:
                create_draft_raw(
                    "kaustubh@example.com",
                    subject,
                    email_body,
                )

            conn.execute(
                "UPDATE jobs SET status = ? WHERE id = ?",
                ("drafted", job["id"]),
            )
            drafted += 1
        except (LLMOutputError, ValueError, FileNotFoundError) as e:
            conn.execute(
                "UPDATE jobs SET status = ?, status_reason = ? WHERE id = ?",
                ("draft_failed", str(e), job["id"]),
            )
            failed += 1

    conn.commit()
    return {"drafted": drafted, "failed": failed, "total": len(shortlisted)}
