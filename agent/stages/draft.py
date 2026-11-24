"""Review and persist email drafts for verified contacts."""
from __future__ import annotations

import json
import re
from contextlib import closing
from pathlib import Path
from sqlite3 import Connection

from agent.checkpoints import AutoCheckpoint, Checkpoint, InteractiveCheckpoint
from agent.config import get_settings
from agent.contacts_client import is_eligible
from agent.db import connect
from agent.gmail_client import DraftRejected, DraftUncertain, create_draft
from agent.llm import LLMOutputError, chat_json

BANNED_PHRASES = re.compile(r"\b(?:i hope this finds you well|i am writing to you|dear)\b", re.IGNORECASE)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _compose(result, contact_name, title, company, profile, attachment, max_words):
    fields = [result[key].strip() for key in ("subject", "hook", "proof", "ask")]
    if not all(fields) or "\n" in fields[0] or "\r" in fields[0]:
        raise ValueError("Draft fields must be nonempty and the subject must be one line.")
    subject = f"{title} at {company}: {fields[0]}"
    if "\n" in subject or "\r" in subject:
        raise ValueError("Invalid role or company in the subject.")
    claims = " ".join(fields).replace(title, "").replace(company, "")
    if BANNED_PHRASES.search(" ".join(fields)):
        raise ValueError("Draft contains a banned phrase.")
    if set(NUMBER.findall(claims)) - set(NUMBER.findall(profile)):
        raise ValueError("Draft contains a number missing from the profile.")

    name = re.search(r"^#[ \t]+Profile:[ \t]*(.+)$", profile, re.MULTILINE | re.IGNORECASE)
    signature = f"\n{name[1].strip()}" if name else ""
    closing_line = ("Resume attached. " if attachment else "") + "Thanks for your time."
    greeting = contact_name.split()[0] if contact_name and contact_name.split() else "there"
    body = "\n\n".join([f"Hi {greeting},", *fields[1:], closing_line + signature])
    if len(f"{subject} {body}".split()) > max_words:
        raise ValueError(f"Draft exceeds {max_words} words.")
    return subject, body


def _daily_count(conn: Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM drafts WHERE date(created_at) = date('now')",
    ).fetchone()[0]


def _finish(conn: Connection, draft_id: int, job_id: int, gmail_id: str | None) -> None:
    conn.execute(
        "UPDATE drafts SET status = 'created', gmail_draft_id = ? WHERE id = ?",
        (gmail_id, draft_id),
    )
    conn.execute("UPDATE jobs SET status = 'drafted', status_reason = NULL WHERE id = ?", (job_id,))


def reconcile(db_path: str | Path | None, draft_id: int, gmail_id: str | None) -> None:
    """Resolve a reserved draft after the operator inspects Gmail Drafts."""
    with closing(connect(db_path)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        draft = conn.execute(
            "SELECT job_id FROM drafts WHERE id = ? AND status = 'creating'", (draft_id,),
        ).fetchone()
        if draft is None:
            raise ValueError(f"Draft {draft_id} is not awaiting reconciliation.")
        if gmail_id:
            if conn.execute("SELECT 1 FROM drafts WHERE gmail_draft_id = ?", (gmail_id,)).fetchone():
                raise ValueError("That Gmail draft ID is already recorded.")
            _finish(conn, draft_id, draft["job_id"], gmail_id)
        else:
            conn.execute("DELETE FROM drafts WHERE id = ?", (draft_id,))
            conn.execute("UPDATE jobs SET status_reason = NULL WHERE id = ?", (draft["job_id"],))


def run(
    dry_run: bool = False, db_path: str | Path | None = None,
    yes: bool = False, checkpoint: Checkpoint | None = None, **kwargs,
) -> dict:
    """Create at most one draft per eligible job/contact pair."""
    settings = get_settings()
    output = Path("out/dry-run/drafts" if dry_run else "out/drafts")
    drafted = failed = skipped = 0
    with closing(connect(db_path)) as conn:
        candidates = conn.execute(
            """SELECT jobs.id AS job_id, jobs.title, jobs.company, jobs.detail_json, jobs.score_json,
                      contacts.id AS contact_id, contacts.name, contacts.role, contacts.email,
                      contacts.verify_status, contacts.confidence,
                      drafts.id AS draft_id, drafts.status AS draft_status
               FROM jobs JOIN contacts ON contacts.job_id = jobs.id
               LEFT JOIN drafts ON drafts.job_id = jobs.id AND drafts.contact_id = contacts.id
               WHERE jobs.status IN ('verified', 'drafted')
                 AND (drafts.id IS NULL OR drafts.status = 'creating')
               ORDER BY jobs.score DESC, jobs.id, contacts.id""",
        ).fetchall()
        pending = [candidate["draft_id"] for candidate in candidates if candidate["draft_status"] == "creating"]
        for draft_id in pending:
            print(f"  Draft {draft_id} needs Gmail review. Inspect {output / f'{draft_id}.txt'}, then run reconcile-draft.")
        eligible = [
            candidate for candidate in candidates
            if candidate["draft_id"] is None and is_eligible(
                candidate["email"], candidate["verify_status"], candidate["confidence"] or 0.0,
                settings.contacts.min_email_confidence,
            )
        ]
        if not eligible:
            return {"drafted": 0, "failed": 0, "skipped": 0, "pending": len(pending), "total": len(candidates)}

        config = settings.email
        attachment = Path(config.resume_path).expanduser() if config.resume_path and not dry_run else None
        if attachment is not None and not attachment.is_file():
            raise ValueError(f"Configured résumé file does not exist: {attachment}")
        profile_path = Path("profile.md")
        profile = profile_path.read_text(encoding="utf-8") if profile_path.exists() else ""
        if not dry_run and not profile.strip():
            raise ValueError("Create profile.md before generating live drafts.")
        template = Path("prompts/email.md").read_text(encoding="utf-8")
        schema = json.loads(Path("schemas/email.json").read_text(encoding="utf-8"))
        checkpoint = checkpoint or (AutoCheckpoint() if yes else InteractiveCheckpoint())

        for candidate in eligible:
            if _daily_count(conn) >= config.max_drafts_per_day:
                print("  Daily draft cap reached.")
                break
            if candidate["verify_status"] == "risky" and isinstance(checkpoint, AutoCheckpoint) and not dry_run:
                print(f"  Contact {candidate['contact_id']} needs interactive review for a risky email.")
                skipped += 1
                continue

            job_id, contact_id = candidate["job_id"], candidate["contact_id"]
            title, company = candidate["title"], candidate["company"]
            try:
                if not title or not company:
                    raise ValueError("Draft needs a job title and company.")
                detail = json.loads(candidate["detail_json"] or "{}")
                detail.update(title=title, company=company)
                score = json.loads(candidate["score_json"] or "{}")
                context = json.dumps({"job": detail, "matched_skills": score.get("matched_skills", [])})
                prompt = template.replace("{profile_md}", profile).replace("{job_json}", context).replace(
                    "{contact_role}", candidate["role"] or "Hiring contact",
                )
                for attempt in range(2):
                    result = {
                        "subject": "Application",
                        "hook": "I am interested in this role.",
                        "proof": "I would be glad to share relevant work.",
                        "ask": "Could we schedule a short chat?",
                    } if dry_run else chat_json(prompt, schema)
                    try:
                        subject, body = _compose(
                            result, candidate["name"], title, company, profile, attachment, config.max_words,
                        )
                    except ValueError as exc:
                        if attempt or dry_run:
                            raise
                        prompt += f"\nRevise the draft to fix this validation error: {exc}"
                    else:
                        break
            except (LLMOutputError, ValueError, TypeError, AttributeError) as exc:
                with conn:
                    conn.execute("UPDATE jobs SET status_reason = ? WHERE id = ?", (str(exc), job_id))
                failed += 1
                continue

            email = candidate["email"].strip()
            warning = "Risky address. Review before approval.\n" if candidate["verify_status"] == "risky" else ""
            attachment_info = f"Attachment: {attachment.name}\n" if attachment else ""
            copy = f"To: {email}\nSubject: {subject}\n{attachment_info}\n{body}\n"
            if not checkpoint.confirm(f"{warning}{copy}\nApprove this draft?"):
                skipped += 1
                continue

            with conn:
                conn.execute("BEGIN IMMEDIATE")
                if _daily_count(conn) >= config.max_drafts_per_day:
                    break
                inserted = conn.execute(
                    """INSERT INTO drafts (job_id, contact_id, subject, body, status)
                       VALUES (?, ?, ?, ?, 'creating') ON CONFLICT(job_id, contact_id) DO NOTHING""",
                    (job_id, contact_id, subject, body),
                )
                if not inserted.rowcount:
                    skipped += 1
                    continue
                draft_id = inserted.lastrowid

            try:
                output.mkdir(parents=True, exist_ok=True)
                (output / f"{draft_id}.txt").write_text(copy, encoding="utf-8")
                gmail_id = None if dry_run else create_draft(
                    email, subject, body, str(attachment) if attachment else None,
                )
            except (DraftRejected, OSError) as exc:
                with conn:
                    conn.execute("DELETE FROM drafts WHERE id = ?", (draft_id,))
                    conn.execute("UPDATE jobs SET status_reason = ? WHERE id = ?", (str(exc), job_id))
                failed += 1
            except DraftUncertain as exc:
                with conn:
                    conn.execute("UPDATE jobs SET status_reason = ? WHERE id = ?", (str(exc), job_id))
                pending.append(draft_id)
                print(f"  Draft {draft_id} may exist in Gmail. Inspect {output / f'{draft_id}.txt'} before reconciling.")
            else:
                with conn:
                    _finish(conn, draft_id, job_id, gmail_id)
                drafted += 1

    return {"drafted": drafted, "failed": failed, "skipped": skipped, "pending": len(pending), "total": len(candidates)}
