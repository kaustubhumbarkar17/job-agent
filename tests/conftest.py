"""Shared test configuration."""
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest


@pytest.fixture
def offline_pipeline(tmp_path, monkeypatch):
    """Isolate pipeline output and replace model calls with deterministic results."""
    from agent import config
    from agent.stages import score

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    copytree(root / "prompts", tmp_path / "prompts")
    copytree(root / "schemas", tmp_path / "schemas")
    settings = config.Settings.from_yaml(root / "config.yaml")
    settings.db_path = str(tmp_path / "data" / "jobs.db")
    monkeypatch.setattr(config, "_settings", settings)
    monkeypatch.setattr(score, "embed", lambda texts, **kwargs: np.ones((len(texts), 2)))
    monkeypatch.setattr(score, "chat_json", lambda *_, **kwargs: {
        "breakdown": {"skills": 0, "seniority": 25, "domain": 20, "location": 15},
        "matched_skills": [],
    })
