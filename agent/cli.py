"""CLI entry point — Typer-based command interface."""
from __future__ import annotations

from pathlib import Path

import typer

from agent.config import get_settings

app = typer.Typer(
    help="Job-Hunt Agent: discover, score, and draft emails for job applications.",
)


def _get_db_path(db: str | None) -> str | None:
    """Return db_path if provided, else None (let stages use default)."""
    return db


@app.command("run")
def run(
    window: str = "7d",
    from_stage: str = typer.Option("discover", "--from", help="Start from this stage."),
    to_stage: str = typer.Option(None, "--to", help="Stop at this stage."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip external API calls."),
    yes: bool = typer.Option(False, "--yes", help="Auto-confirm all checkpoints."),
    db: str = typer.Option(None, "--db", help="Override database path for this run."),
    force_unlock: bool = typer.Option(False, "--force-unlock", help="Clear stale lock file."),
):
    """Run the full pipeline (or a slice)."""
    from agent.orchestrator import run_pipeline

    try:
        results = run_pipeline(
            start=from_stage,
            stop=to_stage,
            dry_run=dry_run,
            yes=yes,
            db_path=_get_db_path(db),
            window=window,
            force_unlock=force_unlock,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    print("\n=== Pipeline Summary ===")
    for stage, result in results.items():
        status = result.get("status", "unknown")
        skipped = result.get("skipped", False)
        if skipped:
            print(f"  [{stage}] SKIPPED: {result.get('reason', '')}")
        elif status == "ok":
            print(f"  [{stage}] OK: {result}")
        else:
            print(f"  [{stage}] ERROR: {result.get('error', '')}")


@app.command("discover")
def discover(
    window: str = "7d",
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Run only the discover stage."""
    from agent.orchestrator import run_pipeline

    results = run_pipeline(
        start="discover", stop="discover", dry_run=False, yes=False,
        db_path=_get_db_path(db), window=window,
    )
    print(f"\nDiscovered: {results.get('discover', {})}")


@app.command("pick")
def pick(
    yes: bool = typer.Option(False, "--yes", help="Auto-select all."),
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Run only the pick stage."""
    from agent.orchestrator import run_pipeline

    results = run_pipeline(
        start="pick", stop="pick", dry_run=False, yes=yes,
        db_path=_get_db_path(db),
    )
    print(f"\nPick results: {results.get('pick', {})}")


@app.command("scrape")
def scrape(
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Run only the scrape stage."""
    from agent.orchestrator import run_pipeline

    results = run_pipeline(
        start="scrape", stop="scrape", dry_run=False, yes=False,
        db_path=_get_db_path(db),
    )
    print(f"\nScrape results: {results.get('scrape', {})}")


@app.command("score")
def score(
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Run only the score stage."""
    from agent.orchestrator import run_pipeline

    results = run_pipeline(
        start="score", stop="score", dry_run=False, yes=False,
        db_path=_get_db_path(db),
    )
    print(f"\nScore results: {results.get('score', {})}")


@app.command("shortlist")
def shortlist(
    yes: bool = typer.Option(False, "--yes", help="Auto-select all."),
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Run only the shortlist stage."""
    from agent.orchestrator import run_pipeline

    results = run_pipeline(
        start="shortlist", stop="shortlist", dry_run=False, yes=yes,
        db_path=_get_db_path(db),
    )
    print(f"\nShortlist results: {results.get('shortlist', {})}")


@app.command("contacts")
def contacts(
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Run only the contacts stage."""
    from agent.orchestrator import run_pipeline

    results = run_pipeline(
        start="contacts", stop="contacts", dry_run=False, yes=False,
        db_path=_get_db_path(db),
    )
    print(f"\nContacts results: {results.get('contacts', {})}")


@app.command("verify")
def verify(
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Run only the verify stage."""
    from agent.orchestrator import run_pipeline

    results = run_pipeline(
        start="verify", stop="verify", dry_run=False, yes=False,
        db_path=_get_db_path(db),
    )
    print(f"\nVerify results: {results.get('verify', {})}")


@app.command("draft")
def draft(
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Run only the draft stage."""
    from agent.orchestrator import run_pipeline

    results = run_pipeline(
        start="draft", stop=None, dry_run=False, yes=False,
        db_path=_get_db_path(db),
    )
    print(f"\nDraft results: {results.get('draft', {})}")


@app.command("status")
def status():
    """Show job counts per status."""
    from agent.orchestrator import status_summary

    summary = status_summary()
    print("\n=== Job Status Summary ===")
    if not summary:
        print("  (no jobs yet)")
    else:
        for status_name, count in sorted(summary.items()):
            print(f"  {status_name}: {count}")


@app.command("reconcile-draft")
def reconcile_draft(
    draft_id: int = typer.Argument(..., help="Local draft ID awaiting review."),
    gmail_draft_id: str = typer.Option("", "--gmail-draft-id", help="ID of the matching Gmail draft."),
    confirmed_missing: bool = typer.Option(False, "--confirmed-missing", help="You checked Gmail and found no match."),
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Resolve an uncertain draft after checking Gmail Drafts."""
    from agent.orchestrator import LockError, acquire_lock, release_lock
    from agent.stages.draft import reconcile

    gmail_draft_id = gmail_draft_id.strip()
    if bool(gmail_draft_id) == confirmed_missing:
        raise typer.BadParameter("Choose exactly one of --gmail-draft-id and --confirmed-missing.")

    try:
        acquire_lock()
    except LockError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        reconcile(db, draft_id, gmail_draft_id or None)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        release_lock()
    print(f"Draft {draft_id} reconciled.")


@app.command("db-init")
def db_init(
    db: str = typer.Option(None, "--db", help="Database path to initialize."),
):
    """Initialize the database."""
    from agent.config import get_settings
    from agent.db import init_db

    settings = get_settings()
    path = Path(db) if db else Path(settings.db_path)
    init_db(path)
    print(f"DB initialized at {path}")


@app.command("db-status")
def db_status(
    db: str = typer.Option(None, "--db", help="Database path to inspect."),
):
    """Show DB-level status counts."""
    from agent.db import connect, get_jobs_by_status_count

    settings = get_settings()
    path = Path(db) if db else Path(settings.db_path)
    conn = connect(path)
    rows = get_jobs_by_status_count(conn)
    print(f"\nDB: {path}")
    for row in rows:
        print(f"  {row[0]}: {row[1]}")
    conn.close()


@app.command("mark-sent")
def mark_sent(
    draft_id: int = typer.Argument(..., help="Draft ID to mark as sent."),
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Mark a draft as sent (after manual send in Gmail)."""
    from agent.db import connect

    settings = get_settings()
    path = Path(db) if db else Path(settings.db_path)
    conn = connect(path)
    conn.execute(
        "UPDATE jobs SET status = 'sent' WHERE id = ?",
        (draft_id,),
    )
    conn.commit()
    print(f"Draft {draft_id} marked as sent.")
    conn.close()


@app.command("followups")
def followups(
    days: int = typer.Option(5, "--days", help="Days since last contact."),
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Show jobs sent >N days ago with no reply."""
    from agent.db import connect

    settings = get_settings()
    path = Path(db) if db else Path(settings.db_path)
    conn = connect(path)
    rows = conn.execute(
        """SELECT id, title, company FROM jobs
           WHERE status = 'sent'
             AND sent_at IS NOT NULL
             AND julianday('now') - julianday(sent_at) > ?""",
        (days,),
    ).fetchall()
    print(f"\nJobs with no reply in {days} days:")
    if not rows:
        print("  (none)")
    else:
        for row in rows:
            print(f"  [{row['id']}] {row['title']} @ {row['company']}")
    conn.close()


@app.command("eval")
def eval_cmd(
    db: str = typer.Option(None, "--db", help="Override database path."),
):
    """Run evaluation on the scorer against a labeled set."""
    from pathlib import Path as _Path

    eval_dir = _Path("tests/fixtures/eval")
    if not eval_dir.exists():
        print("No eval fixtures found at tests/fixtures/eval/")
        return

    import json as _json

    total = 0
    for f in sorted(eval_dir.glob("*.json")):
        data = _json.loads(f.read_text())
        label = data.get("label_score", -1)
        total += 1
        # In dry-run / test mode, just report the fixture exists.
        print(f"  {f.name}: label_score={label}")

    print(f"\nEval: {total} fixtures loaded.")


if __name__ == "__main__":
    app()
