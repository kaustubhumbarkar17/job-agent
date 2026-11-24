"""Draft workflow checks with local fixtures and mocked services."""
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from agent.cli import app
from agent.config import get_settings
from agent.db import connect, upsert_job
from agent.gmail_client import DraftRejected, DraftUncertain
from agent.stages import draft

pytestmark = pytest.mark.usefixtures("offline_pipeline")

COPY = {
    "subject": "Relevant Python experience",
    "hook": "I would like to discuss your Python service.",
    "proof": "I worked on 4 production services.",
    "ask": "Could we schedule a short chat?",
}


@pytest.fixture
def verified_job(tmp_path, monkeypatch):
    db_path = tmp_path / "draft-test.db"
    Path("profile.md").write_text("# Profile: Candidate\nI worked on 4 services.", encoding="utf-8")
    monkeypatch.setattr(draft, "chat_json", Mock(return_value=COPY))
    monkeypatch.setattr(draft, "create_draft", Mock(return_value="gmail-draft-1"))
    with closing(connect(db_path)) as conn:
        job_id = upsert_job(conn, "https://example.com/job", title="SDE 2", company="Example")
        with conn:
            conn.execute("UPDATE jobs SET status = 'verified' WHERE id = ?", (job_id,))
            conn.executemany(
                """INSERT INTO contacts (job_id, name, role, email, verify_status, confidence)
                   VALUES (?, ?, 'Recruiter', ?, ?, ?)""",
                [
                    (job_id, "Alex Recruiter", "alex@example.com", "valid", 1),
                    (job_id, "Mira Recruiter", "mira@example.com", "valid", 1),
                    (job_id, "Invalid Person", "invalid@example.com", "invalid", 1),
                ],
            )
    return db_path


def test_approval_preview_attachment_cap_and_idempotency(verified_job):
    get_settings().email.max_drafts_per_day = 1
    Path("resume.pdf").write_bytes(b"resume fixture")
    get_settings().email.resume_path = "resume.pdf"
    checkpoint = Mock()
    checkpoint.confirm.return_value = True
    assert draft.run(db_path=verified_job, checkpoint=checkpoint)["drafted"] == 1
    to, subject, body, attachment = draft.create_draft.call_args.args
    assert (to, subject, attachment) == ("alex@example.com", "SDE 2 at Example: Relevant Python experience", "resume.pdf")
    assert "Hi Alex," in body
    assert "Resume attached." in body
    assert "To: alex@example.com" in checkpoint.confirm.call_args.args[0]

    with closing(connect(verified_job)) as conn:
        saved = conn.execute("SELECT id, gmail_draft_id FROM drafts").fetchone()
        copy = Path(f"out/drafts/{saved['id']}.txt").read_text(encoding="utf-8")
        assert "Attachment: resume.pdf" in copy
        assert saved["gmail_draft_id"] == "gmail-draft-1"
        assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "drafted"
        assert draft.run(db_path=verified_job, checkpoint=checkpoint)["drafted"] == 0
        with conn:
            conn.execute("UPDATE drafts SET created_at = datetime('now', '-1 day')")
        draft.create_draft.return_value = "gmail-draft-2"
        assert draft.run(db_path=verified_job, checkpoint=checkpoint)["drafted"] == 1
        assert draft.run(db_path=verified_job, checkpoint=checkpoint)["drafted"] == 0

    assert [call.args[0] for call in draft.create_draft.call_args_list] == ["alex@example.com", "mira@example.com"]


@pytest.mark.parametrize("proof", ["I worked on 99 services.", "Dear team, hello.", "word " * 130])
def test_invalid_copy_is_regenerated_once_then_flagged(verified_job, proof):
    draft.chat_json.return_value = {**COPY, "proof": proof}
    result = draft.run(db_path=verified_job, yes=True)
    assert result["failed"] == 2
    assert draft.chat_json.call_count == 4
    draft.create_draft.assert_not_called()
    with closing(connect(verified_job)) as conn:
        assert conn.execute("SELECT status_reason FROM jobs").fetchone()[0]
        assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 0


def test_rejected_approval_and_risky_auto_selection_create_no_drafts(verified_job):
    checkpoint = Mock()
    checkpoint.confirm.return_value = False
    assert draft.run(db_path=verified_job, checkpoint=checkpoint)["skipped"] == 2
    with closing(connect(verified_job)) as conn, conn:
        conn.execute("UPDATE contacts SET verify_status = 'risky' WHERE verify_status = 'valid'")
    assert draft.run(db_path=verified_job, yes=True)["skipped"] == 2
    draft.create_draft.assert_not_called()


def test_uncertain_gmail_outcome_blocks_retry_until_reconciled(verified_job):
    get_settings().email.max_drafts_per_day = 1
    draft.create_draft.side_effect = DraftUncertain("Timeout")
    assert draft.run(db_path=verified_job, yes=True)["pending"] == 1
    assert draft.run(db_path=verified_job, yes=True)["pending"] == 1
    assert draft.create_draft.call_count == 1

    with closing(connect(verified_job)) as conn:
        saved = conn.execute("SELECT id, status FROM drafts").fetchone()
        assert saved["status"] == "creating"
        assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "verified"
        result = CliRunner().invoke(app, [
            "reconcile-draft", str(saved["id"]), "--gmail-draft-id", "existing-gmail-id",
            "--db", str(verified_job),
        ])
        assert result.exit_code == 0, result.output
        assert tuple(conn.execute("SELECT status, gmail_draft_id FROM drafts").fetchone()) == (
            "created", "existing-gmail-id",
        )


def test_explicit_gmail_rejection_leaves_contact_retryable(verified_job):
    get_settings().email.max_drafts_per_day = 1
    draft.create_draft.side_effect = [DraftRejected("Gmail rejected draft"), "gmail-draft-next"]
    checkpoint = Mock()
    checkpoint.confirm.side_effect = [True, False, True]
    assert draft.run(db_path=verified_job, checkpoint=checkpoint)["failed"] == 1
    with closing(connect(verified_job)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 0
    assert draft.run(db_path=verified_job, checkpoint=checkpoint)["drafted"] == 1


def test_full_cli_dry_run_records_drafts_without_provider_calls(monkeypatch):
    external = Mock(side_effect=AssertionError("Offline workflow must not use providers"))
    for path in ("agent.stages.draft.create_draft", "agent.stages.draft.chat_json",
                 "agent.stages.score.chat_json", "agent.stages.score.embed",
                 "agent.contacts_client.httpx.get", "agent.firecrawl_client.httpx.post"):
        monkeypatch.setattr(path, external)

    runner = CliRunner()
    with closing(connect("data/dry-run.db")) as conn:
        for _ in range(2):
            result = runner.invoke(app, ["run", "--dry-run", "--yes"])
            assert result.exit_code == 0, result.output
            assert "ERROR" not in result.output
            assert "api_calls=0, network_calls=0, llm_calls=0" in result.output
            counts = tuple(conn.execute(
                "SELECT COUNT(*), COUNT(gmail_draft_id) FROM drafts WHERE status = 'created'",
            ).fetchone())
            assert counts == (12, 0)
        assert conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'drafted'").fetchone()[0] == 12
    assert len(list(Path("out/dry-run/drafts").glob("*.txt"))) == 12
    external.assert_not_called()
