"""Compare the rubric scorer with job fit labels collected independently."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pydantic import BaseModel, Field, ValidationError

from agent.config import get_settings
from agent.llm import LLMOutputError
from agent.stages.score import load_rubric, score_job

_SYNTHETIC_CASES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "eval"
FitScore = Annotated[int, Field(strict=True, ge=0, le=100)]


class LabeledJob(BaseModel):
    job: dict
    label_score: FitScore
    label_note: str = Field(min_length=1)
    synthetic_score: FitScore | None = None


def _load_cases(directory: Path) -> list[tuple[str, LabeledJob]]:
    paths = sorted(directory.glob("*.json"))
    if len(paths) < 2:
        raise ValueError(f"Add at least two labeled JSON files under {directory}.")
    cases = []
    for path in paths:
        try:
            case = LabeledJob.model_validate_json(path.read_text(encoding="utf-8"))
            title = case.job.get("title")
            if not isinstance(title, str) or not title.strip() or not case.label_note.strip():
                raise ValueError("Each case needs a job title and a label note.")
        except (OSError, ValidationError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        cases.append((path.stem, case))
    return cases


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks for equal values."""
    order = np.argsort(values, kind="stable")
    _, counts = np.unique(values, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat((starts + 1 + ends) / 2, counts)
    return ranks


def _spearman(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    if np.ptp(labels) == 0 or np.ptp(predictions) == 0:
        return None
    return float(np.clip(np.corrcoef(_ranks(labels), _ranks(predictions))[0, 1], -1, 1))


def _report(cases: list[tuple[str, LabeledJob]], predict: Callable[[LabeledJob], int]) -> None:
    predictions = []
    for name, case in cases:
        try:
            predictions.append(predict(case))
        except LLMOutputError as exc:
            raise ValueError(f"Scoring failed for {name}: {exc}") from exc

    labels = np.array([case.label_score for _, case in cases])
    predictions = np.array(predictions)
    errors = np.abs(labels - predictions)
    correlation = _spearman(labels, predictions)
    top_size = min(5, len(cases))
    expected_top = set(np.argsort(-labels, kind="stable")[:top_size])
    predicted_top = set(np.argsort(-predictions, kind="stable")[:top_size])
    overlap = len(expected_top & predicted_top)
    typer.echo(f"Jobs scored: {len(cases)}")
    typer.echo(
        f"Spearman rank correlation: {correlation:.3f}"
        if correlation is not None else "Spearman rank correlation: undefined (constant scores)"
    )
    typer.echo(f"Top-{top_size} overlap: {overlap}/{top_size}")
    typer.echo(f"Mean absolute error: {errors.mean():.2f} points")
    typer.echo("Largest disagreements")
    for index in np.argsort(-errors, kind="stable")[:5]:
        name, case = cases[index]
        typer.echo(
            f"  {case.job['title']} [{name}] · label {labels[index]} · model {predictions[index]}"
            f" · error {errors[index]} · {case.label_note}"
        )
    if len(cases) >= 15:
        passed = correlation is not None and correlation >= 0.7 and overlap >= 3
        typer.echo(f"MVP ranking targets (Spearman ≥ 0.70 and top-5 ≥ 3/5): {'PASS' if passed else 'FAIL'}")
    else:
        typer.echo("Use at least 15 independently labeled jobs for the MVP ranking gate.")


def evaluate(
    eval_dir: Annotated[Path, typer.Option("--eval-dir", help="Directory of labeled jobs.")] = Path("data/eval"),
    profile: Annotated[Path, typer.Option("--profile", help="Profile used for real scoring.")] = Path("profile.md"),
    model: str = typer.Option("", "--model", help="Override the configured Ollama chat model."),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use bundled synthetic scores offline."),
) -> None:
    """Evaluate fit scoring against independently labeled job descriptions."""
    typer.echo("Label real jobs before looking at model output.")
    try:
        if synthetic:
            cases = _load_cases(_SYNTHETIC_CASES)

            def predict(case: LabeledJob) -> int:
                if case.synthetic_score is None:
                    raise ValueError("Synthetic fixtures require a synthetic_score.")
                return case.synthetic_score

            typer.echo("Synthetic fixtures (no model calls)")
            _report(cases, predict)
            return

        cases = _load_cases(eval_dir)
        profile_md = profile.read_text(encoding="utf-8")
        if not profile_md.strip():
            raise ValueError(f"Add a candidate profile to {profile}.")
        rubric = load_rubric()
        model_name = model or get_settings().llm_model

        def score(case: LabeledJob) -> int:
            return score_job(case.job, profile_md, rubric, model=model_name)["score"]

        typer.echo(f"Ollama model: {model_name}")
        _report(cases, score)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    typer.run(evaluate)
