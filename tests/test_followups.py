"""Manual send and reply tracking through the CLI."""
from contextlib import closing

from typer.testing import CliRunner

from agent.cli import app
from agent.db import connect, upsert_job


def test_mark_sent_uses_draft_id_and_preserves_timestamp(tmp_path):
    db = tmp_path / "jobs.db"
    runner = CliRunner()
    with closing(connect(db)) as conn:
        job_id = upsert_job(conn, "https://example.com/job")
        with conn:
            conn.execute("INSERT INTO contacts (id, job_id, email) VALUES (4, ?, 'alex@example.com')", (job_id,))
            conn.execute(
                "INSERT INTO drafts (id, job_id, contact_id, gmail_draft_id) VALUES (8, ?, 4, 'gmail-8')",
                (job_id,),
            )
        args = ["mark-sent", "8", "--db", str(db)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        saved = conn.execute("SELECT status, sent_at FROM drafts WHERE id = 8").fetchone()
        assert saved["status"] == "sent"
        assert saved["sent_at"]
        assert conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()[0] == "discovered"
        repeat = runner.invoke(app, args)
        assert "already marked" in repeat.output
        assert conn.execute("SELECT sent_at FROM drafts WHERE id = 8").fetchone()[0] == saved["sent_at"]
        missing = runner.invoke(app, ["mark-sent", str(job_id), "--db", str(db)])
        assert missing.exit_code == 2


def test_followups_filter_age_replies_and_offline_drafts(tmp_path):
    db = tmp_path / "jobs.db"
    runner = CliRunner()
    with closing(connect(db)) as conn:
        job_id = upsert_job(conn, "https://example.com/job", title="Engineer", company="Example")
        with conn:
            for draft_id, address, age in (
                (1, "due@example.com", "-5 days"),
                (2, "recent@example.com", "-4 days"),
                (3, "reply@example.com", "-6 days"),
                (4, "offline@example.com", None),
            ):
                conn.execute(
                    "INSERT INTO contacts (id, job_id, email) VALUES (?, ?, ?)", (draft_id, job_id, address),
                )
                conn.execute(
                    """INSERT INTO drafts (id, job_id, contact_id, gmail_draft_id, status, sent_at)
                       VALUES (?, ?, ?, ?, ?, datetime('now', ?))""",
                    (draft_id, job_id, draft_id, f"gmail-{draft_id}" if age else None,
                     "sent" if age else "created", age),
                )
    args = ["--db", str(db)]
    assert runner.invoke(app, ["mark-replied", "3", *args]).exit_code == 0
    assert runner.invoke(app, ["mark-sent", "4", *args]).exit_code == 2
    result = runner.invoke(app, ["followups", "--days", "5", *args])
    assert result.exit_code == 0, result.output
    assert "due@example.com" in result.output
    assert "recent@example.com" not in result.output
    assert "reply@example.com" not in result.output
    assert "offline@example.com" not in result.output
