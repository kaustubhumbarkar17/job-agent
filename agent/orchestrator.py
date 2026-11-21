"""Orchestrator — stage registry and pipeline runner.

Each stage is a dict: {name, input_status, output_status, run_fn}.
run_pipeline iterates stages, skipping those with no input rows.
Supports --from and --to for partial runs.
"""
from __future__ import annotations

from typing import Any, Callable

from agent.config import get_settings
from agent.db import connect, get_jobs_by_status_count


STAGE_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "discover",
        "input_status": None,  # Creates rows from scratch
        "output_status": "discovered",
        "module": "agent.stages.discover",
        "fn": "run",
    },
    {
        "name": "pick",
        "input_status": "discovered",
        "output_status": "picked",
        "module": "agent.stages.pick",
        "fn": "run",
    },
    {
        "name": "scrape",
        "input_status": "picked",
        "output_status": "scraped",
        "module": "agent.stages.scrape",
        "fn": "run",
    },
    {
        "name": "score",
        "input_status": "scraped",
        "output_status": "scored",
        "module": "agent.stages.score",
        "fn": "run",
    },
    {
        "name": "shortlist",
        "input_status": "scored",
        "output_status": "shortlisted",
        "module": "agent.stages.shortlist",
        "fn": "run",
    },
    {
        "name": "contacts",
        "input_status": "shortlisted",
        "output_status": "contacted",
        "module": "agent.stages.contacts",
        "fn": "run",
    },
    {
        "name": "verify",
        "input_status": "contacted",
        "output_status": "drafted",
        "module": "agent.stages.verify",
        "fn": "run",
    },
    {
        "name": "draft",
        "input_status": "drafted",  # Final stage; drafts are the output
        "output_status": None,
        "module": "agent.stages.draft",
        "fn": "run",
    },
]


def _get_stage_fn(name: str) -> Callable:
    """Import and return a stage function."""
    import importlib
    for stage in STAGE_REGISTRY:
        if stage["name"] == name:
            mod = importlib.import_module(stage["module"])
            return getattr(mod, stage["fn"])
    raise ValueError(f"Stage '{name}' not found in registry")


def run_pipeline(
    start: str = "discover",
    stop: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    run_id: str = "run-1",
) -> dict[str, Any]:
    """Run the full pipeline (or a slice). Returns per-stage results."""
    settings = get_settings()
    conn = connect(settings.db_path)

    # Find the slice of stages to run
    stage_names = [s["name"] for s in STAGE_REGISTRY]
    start_idx = stage_names.index(start) if start in stage_names else 0
    stop_idx = (stage_names.index(stop) + 1 if stop and stop in stage_names
                else len(stage_names))

    results = {}
    for stage in STAGE_REGISTRY[start_idx:stop_idx]:
        name = stage["name"]
        input_status = stage["input_status"]

        # Check if there are rows to process
        if input_status:
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM jobs WHERE status = ?", (input_status,)
            ).fetchone()["cnt"]
            if count == 0:
                results[name] = {"skipped": True, "reason": f"No {input_status} rows"}
                print(f"  [{name}] Skipped: no {input_status} rows")
                continue

        # Import and run the stage
        fn = _get_stage_fn(name)
        kwargs = {"run_id": run_id, "dry_run": dry_run}
        if name in ("pick", "shortlist"):
            kwargs["yes"] = yes

        print(f"  [{name}] Running...")
        try:
            result = fn(**kwargs)
            results[name] = {**result, "status": "ok"}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
            print(f"  [{name}] Error: {e}")

    return results


def status_summary() -> dict[str, int]:
    """Return job counts per status."""
    settings = get_settings()
    conn = connect(settings.db_path)
    rows = get_jobs_by_status_count(conn)
    return {r["status"]: r["cnt"] for r in rows}
