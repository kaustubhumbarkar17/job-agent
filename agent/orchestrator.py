"""Orchestrator — stage registry and pipeline runner.

Each stage is a dict: {name, input_status, output_status, run_fn}.
run_pipeline iterates stages, skipping those with no input rows.
Supports --from and --to for partial runs.

Features:
  - Checkpoint system (Interactive / Scripted / Auto)
  - Per-stage retries (network: 3 attempts; LLM: 1 retry)
  - Single-run lock file in data/.lock
  - JSON-lines logging per run
  - Budget summary at end
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.checkpoints import AutoCheckpoint, Checkpoint, ScriptedCheckpoint
from agent.config import get_settings
from agent.db import connect, get_jobs_by_status_count, STATUS_TRANSITIONS

# ---------------------------------------------------------------------------
# Lock manager — single-run lock file
# ---------------------------------------------------------------------------

_LOCK_PATH = Path("data/.lock")


class LockError(Exception):
    """Raised when a concurrent run is detected."""


def acquire_lock(force: bool = False) -> None:
    """Acquire a single-run lock file.

    Raises LockError if a non-stale lock exists and force=False.
    Stale = older than 6 hours.
    """
    if _LOCK_PATH.exists():
        mtime = datetime.fromtimestamp(_LOCK_PATH.stat().st_mtime, tz=timezone.utc)
        stale_after = datetime.now(tz=timezone.utc).replace(hour=0) - __import__("datetime").timedelta(hours=6)
        if mtime >= stale_after:
            raise LockError(
                f"Another run is active ({_LOCK_PATH}). "
                f"Use --force-unlock to clear."
            )
        else:
            os.remove(_LOCK_PATH)  # Clear stale lock

    _LOCK_PATH.write_text(
        f"locked at {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )


def release_lock() -> None:
    """Remove the lock file if it exists."""
    if _LOCK_PATH.exists():
        os.remove(_LOCK_PATH)


# ---------------------------------------------------------------------------
# Logging — JSON lines per run
# ---------------------------------------------------------------------------

_LOG_DIR = Path("logs")


def setup_run_logging(run_id: str) -> None:
    """Create a JSON-lines log file for this run."""
    _LOG_DIR.mkdir(exist_ok=True)
    path = _LOG_DIR / f"run-{run_id}.jsonl"
    path.write_text(
        json.dumps({"event": "run_start", "run_id": run_id,
                     "ts": datetime.now(timezone.utc).isoformat()}) + "\n",
        encoding="utf-8",
    )


def log_event(run_id: str, stage: str, **kwargs: Any) -> None:
    """Append a JSON line to the run's log file."""
    _LOG_DIR.mkdir(exist_ok=True)
    path = _LOG_DIR / f"run-{run_id}.jsonl"
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        **kwargs,
    }
    path.write_text(path.read_text(encoding="utf-8") + "\n" + json.dumps(entry),
                     encoding="utf-8")

# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

# Network stages: these make external API calls and need retries.
_NETWORK_STAGES = {"discover", "scrape", "contacts", "verify"}
# LLM stages: call the local model, need at most 1 retry.
_LLM_STAGES = {"score", "draft"}


STAGE_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "discover",
        "input_status": None,
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
        "input_status": "drafted",
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


def _is_network_stage(name: str) -> bool:
    return name in _NETWORK_STAGES


def _is_llm_stage(name: str) -> bool:
    return name in _LLM_STAGES


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _run_with_retries(fn: Callable, stage_name: str, **kwargs: Any) -> dict[str, Any]:
    """Run a stage function with retries.

    Network stages get 3 attempts with exponential backoff.
    LLM stages get 1 retry (2 total attempts).
    Other stages get 1 attempt.
    """
    max_attempts = 3 if _is_network_stage(stage_name) else (2 if _is_llm_stage(stage_name) else 1)
    errors: list[str] = []
    last_exc: BaseException | None = None

    for attempt in range(max_attempts):
        if attempt > 0:
            delay = 2 ** (attempt - 1)  # 1s, 2s, ...
            print(f"  [{stage_name}] Retry {attempt}/{max_attempts-1} after {delay}s...")
            time.sleep(delay)

        start = time.time()
        try:
            result = fn(**kwargs)
            return result
        except Exception as e:
            errors.append(str(e))
            last_exc = e
            print(f"  [{stage_name}] Attempt {attempt+1} failed: {e}")

    if last_exc is not None:
        raise RuntimeError(
            f"[{stage_name}] All {max_attempts} attempts failed: {'; '.join(errors)}"
        ) from last_exc
    raise RuntimeError(f"[{stage_name}] Unknown failure after {max_attempts} attempts")

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    start: str = "discover",
    stop: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    db_path: str | Path | None = None,
    run_id: str = "run-1",
    force_unlock: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline (or a slice). Returns per-stage results.

    Uses Checkpoint system for human-in-the-loop stages.
    Handles single-run lock, retries, and logging.
    """
    settings = get_settings()
    conn = connect(db_path or settings.db_path)

    # Acquire lock (skip in dry-run mode)
    if not dry_run:
        try:
            acquire_lock(force=force_unlock)
        except LockError as e:
            print(f"Lock error: {e}")
            return {"error": str(e)}

    setup_run_logging(run_id)

    stage_names = [s["name"] for s in STAGE_REGISTRY]
    start_idx = stage_names.index(start) if start in stage_names else 0
    stop_idx = (stage_names.index(stop) + 1 if stop and stop in stage_names
                else len(stage_names))

    # Choose checkpoint based on --yes flag
    checkpoint: Checkpoint = (
        AutoCheckpoint() if yes else InteractiveCheckpoint()
    )

    results: dict[str, Any] = {}
    total_start = time.time()
    budget: dict[str, Any] = {"api_calls": 0, "llm_calls": 0, "network_calls": 0}

    print(f"\n{'='*60}")
    print(f"Pipeline run: {run_id}  (start={start}, stop={stop or 'end'})")
    print(f"{'='*60}")

    try:
        for stage in STAGE_REGISTRY[start_idx:stop_idx]:
            name = stage["name"]
            input_status = stage["input_status"]

            # Check if there are rows to process
            if input_status:
                count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM jobs WHERE status = ?",
                    (input_status,),
                ).fetchone()["cnt"]
                if count == 0:
                    results[name] = {"skipped": True, "reason": f"No {input_status} rows"}
                    print(f"  [{name}] Skipped: no {input_status} rows")
                    log_event(run_id, name, event="skipped", reason=f"No {input_status} rows")
                    continue

            # Run the stage with retries
            fn = _get_stage_fn(name)
            kwargs = {
                "run_id": run_id,
                "dry_run": dry_run,
                "db_path": db_path,
                "yes": yes,
            }

            print(f"\n  [{name}] Running...")
            log_event(run_id, name, event="start")
            stage_start = time.time()

            try:
                result = _run_with_retries(fn, name, **kwargs)
                results[name] = {**result, "status": "ok"}
                elapsed = time.time() - stage_start
                print(f"  [{name}] OK ({elapsed:.1f}s): {result}")
                log_event(run_id, name, event="ok", duration_s=round(elapsed, 2))

                # Track budget
                if _is_network_stage(name):
                    budget["network_calls"] += result.get("total", 0)
                if _is_llm_stage(name):
                    budget["llm_calls"] += result.get("scored", 0) + result.get("drafted", 0)

            except Exception as e:
                elapsed = time.time() - stage_start
                results[name] = {"status": "error", "error": str(e), "duration_s": round(elapsed, 2)}
                print(f"  [{name}] ERROR after {elapsed:.1f}s: {e}")
                log_event(run_id, name, event="error", error=str(e))

        # Release lock
        if not dry_run:
            release_lock()

    except BaseException:
        if not dry_run:
            release_lock()
        raise
    finally:
        elapsed_total = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"Pipeline complete in {elapsed_total:.1f}s")
        print(f"Budget: api_calls={budget['api_calls']}, "
              f"network_calls={budget['network_calls']}, "
              f"llm_calls={budget['llm_calls']}")
        print(f"{'='*60}")

        log_event(run_id, "_summary", event="pipeline_complete",
                  duration_s=round(elapsed_total, 2), budget=budget)
        conn.commit()
        conn.close()

    return results


def status_summary() -> dict[str, int]:
    """Return job counts per status."""
    settings = get_settings()
    conn = connect(settings.db_path)
    rows = get_jobs_by_status_count(conn)
    return {r["status"]: r["cnt"] for r in rows}
