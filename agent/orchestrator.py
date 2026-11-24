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

import fcntl
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.checkpoints import AutoCheckpoint, Checkpoint, InteractiveCheckpoint
from agent.config import get_settings
from agent.db import connect, get_jobs_by_status_count

# ---------------------------------------------------------------------------
# Lock manager — single-run lock file
# ---------------------------------------------------------------------------

_LOCK_PATH = Path("data/.lock")
_lock_directory_fd: int | None = None


class LockError(Exception):
    """Raised when a concurrent run is detected."""


def acquire_lock(force: bool = False) -> None:
    """Acquire a single-run lock file.

    Raises LockError if a non-stale lock exists and force=False.
    Stale = older than 6 hours.
    """
    global _lock_directory_fd
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(_LOCK_PATH.parent, os.O_RDONLY)
    try:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockError(f"Another run holds {_LOCK_PATH}.") from exc
        if _LOCK_PATH.exists():
            age = time.time() - _LOCK_PATH.stat().st_mtime
            if not force or age <= 6 * 3600:
                raise LockError(f"Found {_LOCK_PATH}. Use --force-unlock for a stale lock.")
            _LOCK_PATH.unlink()
        with _LOCK_PATH.open("x", encoding="utf-8") as lock:
            lock.write(f"locked at {datetime.now(UTC).isoformat()}\n")
    except (OSError, LockError):
        os.close(directory_fd)
        raise
    _lock_directory_fd = directory_fd


def release_lock() -> None:
    """Remove the sentinel before relinquishing the directory lock."""
    global _lock_directory_fd
    if _lock_directory_fd is not None:
        try:
            _LOCK_PATH.unlink(missing_ok=True)
        finally:
            os.close(_lock_directory_fd)
            _lock_directory_fd = None


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
                     "ts": datetime.now(UTC).isoformat()}) + "\n",
        encoding="utf-8",
    )


def log_event(run_id: str, stage: str, **kwargs: Any) -> None:
    """Append a JSON line to the run's log file."""
    _LOG_DIR.mkdir(exist_ok=True)
    path = _LOG_DIR / f"run-{run_id}.jsonl"
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "stage": stage,
        **kwargs,
    }
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")

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
        "input_status": ("picked", "scrape_failed"),
        "output_status": "scraped",
        "module": "agent.stages.scrape",
        "fn": "run",
    },
    {
        "name": "score",
        "input_status": ("scraped", "score_failed"),
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

        try:
            result = fn(**kwargs)
            return result
        except Exception as e:  # noqa: BLE001 - stages have independent failure boundaries.
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
    run_id: str | None = None,
    force_unlock: bool = False,
    window: str | None = None,
    checkpoint: Checkpoint | None = None,
) -> dict[str, Any]:
    """Run the full pipeline (or a slice). Returns per-stage results.

    Uses Checkpoint system for human-in-the-loop stages.
    Handles single-run lock, retries, and logging.
    """
    settings = get_settings()
    stage_names = [stage["name"] for stage in STAGE_REGISTRY]
    if start not in stage_names or (stop is not None and stop not in stage_names):
        raise ValueError(f"Unknown stage. Choose from {', '.join(stage_names)}.")
    start_idx = stage_names.index(start)
    stop_idx = stage_names.index(stop) + 1 if stop is not None else len(stage_names)
    if start_idx >= stop_idx:
        raise ValueError("The start stage must precede or equal the stop stage.")

    window = settings.search.window if window is None else window
    if window not in {"1d", "7d"}:
        raise ValueError("Search window must be 1d or 7d.")
    run_id = run_id or uuid4().hex
    db_path = db_path or ("data/dry-run.db" if dry_run else settings.db_path)
    if dry_run and Path(db_path).resolve() == Path(settings.db_path).resolve():
        raise ValueError("Dry runs require a database separate from the live database.")

    if not dry_run:
        acquire_lock(force=force_unlock)
    try:
        conn = connect(db_path)
        try:
            setup_run_logging(run_id)
            conn.execute("INSERT INTO runs (run_id, window) VALUES (?, ?)", (run_id, window))
            conn.commit()
            return _execute_stages(
                conn, start_idx, stop_idx, run_id, db_path, dry_run, yes, window,
                checkpoint or (AutoCheckpoint() if yes else InteractiveCheckpoint()),
            )
        finally:
            conn.close()
    finally:
        if not dry_run:
            release_lock()


def _execute_stages(conn, start_idx, stop_idx, run_id, db_path, dry_run, yes, window, checkpoint):
    results: dict[str, Any] = {}
    total_start = time.time()
    budget: dict[str, Any] = {"api_calls": 0, "llm_calls": 0, "network_calls": 0}

    print(f"\n{'='*60}")
    print(f"Pipeline run: {run_id}")
    print(f"{'='*60}")

    try:
        for stage in STAGE_REGISTRY[start_idx:stop_idx]:
            name = stage["name"]
            input_status = stage["input_status"]

            # Check if there are rows to process
            if input_status:
                statuses = (input_status,) if isinstance(input_status, str) else input_status
                placeholders = ", ".join("?" for _ in statuses)
                count = conn.execute(
                    f"SELECT COUNT(*) as cnt FROM jobs WHERE status IN ({placeholders})",
                    statuses,
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
                "window": window,
                "checkpoint": checkpoint,
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

            except Exception as e:  # noqa: BLE001 - keep subsequent stages observable.
                elapsed = time.time() - stage_start
                results[name] = {"status": "error", "error": str(e), "duration_s": round(elapsed, 2)}
                print(f"  [{name}] ERROR after {elapsed:.1f}s: {e}")
                log_event(run_id, name, event="error", error=str(e))

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

    return results


def status_summary() -> dict[str, int]:
    """Return job counts per status."""
    settings = get_settings()
    conn = connect(settings.db_path)
    rows = get_jobs_by_status_count(conn)
    return {r["status"]: r["cnt"] for r in rows}
