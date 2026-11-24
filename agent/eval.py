"""Compare the rubric scorer with job fit labels collected independently."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from pydantic import BaseModel, Field, ValidationError

from agent.config import get_settings
from agent.llm import LLMOutputError
from agent.stages.score import load_rubric, score_job

_SYNTHETIC_CASES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "eval"
FitScore = Annotated[int, Field(strict=True, ge=0, le=100)]
_CHECKBOX = re.compile(r"^\s*- \[([ xX])\] \[([^\]]+)\]\([^)]*\)")


class LabeledJob(BaseModel):
    job: dict
    label_score: FitScore | None = None
    label_note: str = ""
    synthetic_score: FitScore | None = None


def _load_cases(directory: Path, *, selection: bool = False) -> list[tuple[str, LabeledJob]]:
    paths = sorted(directory.glob("*.json"))
    if len(paths) < 2:
        raise ValueError(f"Add at least two labeled JSON files under {directory}.")
    cases = []
    for path in paths:
        try:
            case = LabeledJob.model_validate_json(path.read_text(encoding="utf-8"))
            title = case.job.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("Each case needs a job title.")
            if not selection and (case.label_score is None or not case.label_note.strip()):
                raise ValueError("Each numeric case needs a score and a label note.")
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


def _predict(
    cases: list[tuple[str, LabeledJob]], predict: Callable[[LabeledJob], dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for name, case in cases:
        try:
            results.append(predict(case))
        except LLMOutputError as exc:
            raise ValueError(f"Scoring failed for {name}: {exc}") from exc

    return results


def _report(cases: list[tuple[str, LabeledJob]], results: list[dict[str, Any]]) -> None:
    labels = np.array([case.label_score for _, case in cases])
    predictions = np.array([result["score"] for result in results])
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


def _read_selection(checklist: Path, cases: list[tuple[str, LabeledJob]]) -> set[str]:
    rows = [match.groups() for line in checklist.read_text(encoding="utf-8").splitlines()
            if (match := _CHECKBOX.match(line))]
    if len(rows) != len(cases) or {name for _, name in rows} != {name for name, _ in cases}:
        raise ValueError(f"{checklist} must contain exactly one checkbox for each job ID.")
    marked = {name for mark, name in rows if mark.lower() == "x"}
    if not marked:
        raise ValueError(f"Mark at least one job in {checklist} before scoring.")
    return marked


def _report_selection(
    cases: list[tuple[str, LabeledJob]], results: list[dict[str, Any]], marked: set[str],
) -> None:
    predictions = np.array([result["score"] for result in results])
    order = np.argsort(-predictions, kind="stable")
    top = order[:5]
    overlap = sum(cases[index][0] in marked for index in top)
    typer.echo(f"Jobs scored: {len(cases)} · You marked: {len(marked)}")
    typer.echo(f"Marked jobs in the model's top {len(top)}: {overlap}/{len(top)}")
    typer.echo(f"Your marked jobs found in the top {len(top)}: {overlap}/{len(marked)}")
    zero_marked = [name for (name, _), value in zip(cases, predictions, strict=True) if name in marked and value == 0]
    typer.echo(f"Marked jobs scored zero: {len(zero_marked)}/{len(marked)}")
    typer.echo("Model ranking")
    for rank, index in enumerate(order, 1):
        name, case = cases[index]
        status = "marked" if name in marked else "unmarked"
        typer.echo(f"  {rank}. [{name}] {case.job['title']} · model {predictions[index]} · {status}")
        result = results[index]
        breakdown = ", ".join(f"{key} {part}" for key, part in result["breakdown"].items())
        typer.echo(f"      Breakdown: {breakdown}")
        typer.echo(f"      Seniority fit: {result.get('seniority_fit') or 'Unspecified'}")
        typer.echo(f"      Matched skills: {', '.join(result.get('matched_skills', [])) or 'None listed'}")
        typer.echo(f"      Missing skills: {', '.join(result.get('missing_skills', [])) or 'None listed'}")
        typer.echo(f"      Dealbreaker: {result.get('dealbreaker') or 'None reported'}")
        typer.echo(f"      Reasoning: {result.get('reasoning') or 'None reported'}")


def evaluate(
    eval_dir: Annotated[Path, typer.Option("--eval-dir", help="Directory of labeled jobs.")] = Path("data/eval"),
    profile: Annotated[Path, typer.Option("--profile", help="Profile used for real scoring.")] = Path("profile.md"),
    model: str = typer.Option("", "--model", help="Override the configured Ollama chat model."),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use bundled synthetic scores offline."),
    selection: bool = typer.Option(False, "--selection", help="Read marked jobs from LABELING.md."),
) -> None:
    """Evaluate fit scoring against independently labeled job descriptions."""
    typer.echo("Label real jobs before looking at model output.")
    try:
        if synthetic:
            if selection:
                raise ValueError("Choose either --synthetic or --selection.")
            cases = _load_cases(_SYNTHETIC_CASES)

            def predict(case: LabeledJob) -> dict[str, int]:
                if case.synthetic_score is None:
                    raise ValueError("Synthetic fixtures require a synthetic_score.")
                return {"score": case.synthetic_score}

            typer.echo("Synthetic fixtures (no model calls)")
            _report(cases, _predict(cases, predict))
            return

        cases = _load_cases(eval_dir, selection=selection)
        marked = _read_selection(eval_dir / "LABELING.md", cases) if selection else set()
        profile_md = profile.read_text(encoding="utf-8")
        if not profile_md.strip():
            raise ValueError(f"Add a candidate profile to {profile}.")
        rubric = load_rubric()
        model_name = model or get_settings().llm_model

        def score(case: LabeledJob) -> dict[str, Any]:
            return score_job(case.job, profile_md, rubric, model=model_name)

        typer.echo(f"Ollama model: {model_name}")
        results = _predict(cases, score)
        if selection:
            _report_selection(cases, results, marked)
        else:
            _report(cases, results)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    typer.run(evaluate)
