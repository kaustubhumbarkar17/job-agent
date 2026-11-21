"""Stage 2: Pick — human selects which jobs to deep-scrape.

Uses questionary checkboxes in the terminal. Pre-ticks nothing.
Supports --yes for scripted runs (picks all).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import questionary

from agent.db import connect, jobs_by_status, set_status


def run(
    run_id: str = "",
    yes: bool = False,
    db_path: Optional[str | Path] = None,
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

    if yes:
        selected = [item["id"] for item in items]
    else:
        labels = [item["label"] for item in items]
        chosen = questionary.checkbox(
            "Select jobs to deep-scrape (Space to select, Enter to confirm):",
            choices=labels,
        ).ask()
        if chosen is None:
            print("  Cancelled.")
            return {"selected": [], "total": 0}
        chosen_indices = [i for i, l in enumerate(labels) if l in chosen]
        selected = [items[i]["id"] for i in chosen_indices]

    for job_id in selected:
        set_status(conn, job_id, "picked")

    return {"selected": selected, "total": len(selected)}
