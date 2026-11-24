"""Ollama client wrapper.

Provides a single method: chat_json that returns schema-validated JSON.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from jsonschema import ValidationError, validate

from agent.config import settings


class LLMOutputError(Exception):
    """Raised when the LLM returns invalid JSON twice."""


def chat_json(
    prompt: str,
    schema: dict[str, Any],
    system: str | None = None,
    temperature: float = 0.0,
    model: str | None = None,
) -> dict[str, Any]:
    """Call Ollama /api/chat and return a validated JSON dict.

    Retries once on invalid JSON. Raises LLMOutputError on second failure.
    """
    model_name = model or settings.llm_model
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": (
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            if system
            else [{"role": "user", "content": prompt}]
        ),
        "format": schema,
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
        },
        "stream": False,
        "think": False,
    }

    host = settings.ollama_host
    last_err = LLMOutputError("Unknown failure")

    for attempt in range(2):
        try:
            resp = httpx.post(f"{host}/api/chat", json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"].strip()
            parsed = json.loads(content)
            validate(parsed, schema)
            return parsed
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValidationError) as e:
            last_err = LLMOutputError(
                f"Attempt {attempt + 1}: Invalid JSON — {e}"
            )
            if attempt == 0:
                # Retry with a stronger nudge
                payload["messages"][-1]["content"] = "Return valid JSON only. " + prompt
            continue
        except httpx.HTTPError as e:
            raise LLMOutputError(f"Ollama unreachable: {e}")

    raise last_err


def chat_raw(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
) -> str:
    """Call Ollama and return raw text (no schema validation)."""
    model_name = model or settings.llm_model
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": (
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            if system
            else [{"role": "user", "content": prompt}]
        ),
        "options": {"temperature": 0.5, "num_ctx": 8192},
        "stream": False,
    }
    resp = httpx.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"]
