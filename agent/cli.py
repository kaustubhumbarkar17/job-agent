"""CLI entry point — Typer-based command interface."""
from __future__ import annotations

import sys
from pathlib import Path

import typer

app = typer.Typer(help="Job-Hunt Agent: discover, score, and draft emails for job applications.")


@app.command("run")
def run(
    window: str = "7d",
    from_stage: str = typer.Option("discover", "--from", help="Start from this stage."),
    to_stage: str = typer.Option(None, "--to", help="Stop at this stage."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip external API calls."),
    yes: bool = typer.Option(False, "--yes", help="Auto-confirm all checkpoints."),
):
    """Run the full pipeline (or a slice)."""
    from agent.orchestrator import run_pipeline
    results = run_pipeline(
        start=from_stage,
        stop=to_stage,
        dry_run=dry_run,
        yes=yes,
        run_id=f"run-{window}",
    )
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
def discover(window: str = "7d"):
    """Run only the discover stage."""
    from agent.orchestrator import run_pipeline
    results = run_pipeline(start="discover", stop="pick", dry_run=False, yes=False)
    print(f"\nDiscovered: {results.get('discover', {})}")


@app.command("pick")
def pick(yes: bool = typer.Option(False, "--yes")):
    """Run only the pick stage."""
    from agent.orchestrator import run_pipeline
    results = run_pipeline(start="pick", stop="scrape", dry_run=False, yes=yes)
    print(f"Pick results: {results.get('pick', {})}")


@app.command("scrape")
def scrape():
    """Run only the scrape stage."""
    from agent.orchestrator import run_pipeline
    results = run_pipeline(start="scrape", stop="score", dry_run=False, yes=False)
    print(f"Scrape results: {results.get('scrape', {})}")


@app.command("score")
def score():
    """Run only the score stage."""
    from agent.orchestrator import run_pipeline
    results = run_pipeline(start="score", stop="shortlist", dry_run=False, yes=False)
    print(f"Score results: {results.get('score', {})}")


@app.command("shortlist")
def shortlist(yes: bool = typer.Option(False, "--yes")):
    """Run only the shortlist stage."""
    from agent.orchestrator import run_pipeline
    results = run_pipeline(start="shortlist", stop="contacts", dry_run=False, yes=yes)
    print(f"Shortlist results: {results.get('shortlist', {})}")


@app.command("contacts")
def contacts():
    """Run only the contacts stage."""
    from agent.orchestrator import run_pipeline
    results = run_pipeline(start="contacts", stop="verify", dry_run=False, yes=False)
    print(f"Contacts results: {results.get('contacts', {})}")


@app.command("draft")
def draft():
    """Run only the draft stage."""
    from agent.orchestrator import run_pipeline
    results = run_pipeline(start="draft", stop=None, dry_run=False, yes=False)
    print(f"Draft results: {results.get('draft', {})}")


@app.command("status")
def status():
    """Show job counts per status."""
    from agent.orchestrator import status_summary
    summary = status_summary()
    print("\n=== Job Status Summary ===")
    for status, count in sorted(summary.items()):
        print(f"  {status}: {count}")


@app.command("db-init")
def db_init():
    """Initialize the database."""
    from agent.config import get_settings
    from agent.db import init_db
    settings = get_settings()
    init_db(settings.db_path)
    print(f"DB initialized at {settings.db_path}")


if __name__ == "__main__":
    app()
