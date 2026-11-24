"""Stage 5: Shortlist — human selects which scored jobs to pursue.

Shows ranked table via questionary. Pre-ticks those >= min_score_for_shortlist.
"""
from __future__ import annotations

from pathlib import Path

from agent.checkpoints import AutoCheckpoint, Checkpoint, InteractiveCheckpoint
from agent.config import get_settings
from agent.db import connect, jobs_by_status, set_status


def run(
    run_id: str = "",
    yes: bool = False,
    db_path: str | Path | None = None,
    checkpoint: Checkpoint | None = None,
    **kwargs,
) -> dict:
    """Run the shortlist stage. Returns selected job IDs."""
    conn = connect(db_path)
    scored = jobs_by_status(conn, "scored")

    if not scored:
        print("  No scored jobs to shortlist.")
        return {"selected": [], "total": 0}

    scored_sorted = sorted(scored, key=lambda j: j["score"] or 0, reverse=True)

    items = []
    for j in scored_sorted:
        items.append({
            "id": j["id"],
            "label": f"[{j['id']}] Score:{j['score']} | {j['title']} | {j['company']} | {j['location']}",
        })

    checkpoint = checkpoint or (AutoCheckpoint() if yes else InteractiveCheckpoint())
    threshold = get_settings().scoring.min_score_for_shortlist
    preselected = [job["id"] for job in scored_sorted if (job["score"] or 0) >= threshold]
    selected = checkpoint.select(items, preselected=preselected, label="Select jobs to pursue")

    for job_id in selected:
        set_status(conn, job_id, "shortlisted")

    return {"selected": selected, "total": len(selected)}
