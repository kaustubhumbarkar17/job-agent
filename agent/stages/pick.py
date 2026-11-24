"""Stage 2: Pick — human selects which jobs to deep-scrape.

Uses questionary checkboxes in the terminal. Pre-ticks nothing.
Supports --yes for scripted runs (picks all).
"""
from __future__ import annotations

from pathlib import Path

from agent.checkpoints import AutoCheckpoint, Checkpoint, InteractiveCheckpoint
from agent.db import connect, jobs_by_status, set_status


def run(
    run_id: str = "",
    yes: bool = False,
    db_path: str | Path | None = None,
    checkpoint: Checkpoint | None = None,
    **kwargs,
) -> dict:
    """Run the pick stage. Returns selected job IDs."""
    conn = connect(db_path)
    jobs = jobs_by_status(conn, "discovered")

    if not jobs:
        print("  No discovered jobs to pick.")
        return {"selected": [], "total": 0}

    items = []
    for j in jobs:
        items.append({
            "id": j["id"],
            "label": f"[{j['id']}] {j['title']} | {j['company']} | {j['location']} | {j['source']}",
        })

    checkpoint = checkpoint or (AutoCheckpoint() if yes else InteractiveCheckpoint())
    selected = checkpoint.select(items, label="Select jobs to deep-scrape")

    for job_id in selected:
        set_status(conn, job_id, "picked")

    return {"selected": selected, "total": len(selected)}
