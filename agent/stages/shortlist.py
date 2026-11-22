"""Stage 5: Shortlist — human selects which scored jobs to pursue.

Shows ranked table via questionary. Pre-ticks those >= min_score_for_shortlist.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agent.db import connect, jobs_by_status, set_status


def run(
    run_id: str = "",
    yes: bool = False,
    db_path: Optional[str | Path] = None,
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

    if yes:
        selected = [item["id"] for item in items]
    else:
        # Lazy import to avoid EOFError in non-tty environments (tests, CI)
        import questionary

        labels = [item["label"] for item in items]
        chosen = questionary.checkbox(
            "Select jobs to pursue (Space to select, Enter to confirm):",
            choices=labels,
        ).ask()
        if chosen is None:
            print("  Cancelled.")
            return {"selected": [], "total": 0}
        selected = [item["id"] for item in items if item["label"] in chosen]

    for job_id in selected:
        set_status(conn, job_id, "shortlisted")

    return {"selected": selected, "total": len(selected)}
