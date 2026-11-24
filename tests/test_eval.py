"""Evaluation metric and CLI tests using synthetic or mocked model scores."""
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from typer.testing import CliRunner

from agent.cli import app
from agent.eval import _spearman
from agent.stages import score


def test_spearman_uses_average_ranks_for_ties():
    assert _spearman(np.array([1, 2, 2, 3]), np.array([4, 1, 1, 0])) == pytest.approx(-1)
    assert _spearman(np.array([1, 2, 3]), np.array([1, 1, 2])) == pytest.approx(0.8660254)
    assert _spearman(np.array([1, 2, 3]), np.array([0, 0, 0])) is None


def test_synthetic_eval_never_contacts_model(monkeypatch):
    model = Mock(side_effect=AssertionError("Offline evaluation must not contact Ollama"))
    monkeypatch.setattr(score, "chat_json", model)
    result = CliRunner().invoke(app, ["eval", "--synthetic"])
    assert result.exit_code == 0, result.output
    assert "Spearman rank correlation: 1.000" in result.output
    assert "Top-3 overlap: 3/3" in result.output
    assert "Mean absolute error: 11.67 points" in result.output
    assert result.output.index("Platform Engineer") < result.output.index("Backend Engineer")
    model.assert_not_called()


@pytest.mark.usefixtures("offline_pipeline")
def test_labeled_eval_uses_shared_scorer_and_excludes_labels_from_prompt(monkeypatch):
    directory = Path("data/eval")
    directory.mkdir()
    Path("profile.md").write_text("Python", encoding="utf-8")
    for index in range(15):
        case = {
            "job": {"title": f"Engineer {index}"},
            "label_score": index,
            "label_note": "Private label explanation",
        }
        (directory / f"{index:02}.json").write_text(json.dumps(case), encoding="utf-8")

    def predict(prompt, schema, *, model, **kwargs):
        assert model == "fixture-model"
        assert "Private label explanation" not in prompt
        assert "label_score" not in prompt
        index = int(prompt.split('"title": "Engineer ')[1].split('"')[0])
        return {
            "score": 99,
            "breakdown": {"skills": index * 2, "seniority": 0, "domain": 0, "location": 0},
            "matched_skills": ["Python", "Hallucinated skill"],
        }

    model = Mock(side_effect=predict)
    monkeypatch.setattr(score, "chat_json", model)
    result = CliRunner().invoke(app, ["eval", "--model", "fixture-model"])
    assert result.exit_code == 0, result.output
    assert "MVP ranking targets" in result.output
    assert "PASS" in result.output
    assert "Top-5 overlap: 5/5" in result.output
    assert "Mean absolute error: 0.00 points" in result.output
    assert model.call_count == 15


def test_invalid_labels_are_rejected_before_scoring(tmp_path, monkeypatch):
    model = Mock()
    monkeypatch.setattr(score, "chat_json", model)
    for name, label in [("valid", 40), ("invalid", True)]:
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "job": {"title": "Engineer"}, "label_score": label, "label_note": "Label",
        }), encoding="utf-8")
    result = CliRunner().invoke(app, ["eval", "--eval-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "invalid.json" in result.output
    model.assert_not_called()
