"""Tests for agent/llm.py using mocked HTTP."""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.llm import LLMOutputError, chat_json


class TestChatJson:
    """LLM JSON chat tests."""

    def test_valid_json_returned(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        with patch("agent.llm.httpx.post") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {"message": {"content": '{"ok": true}', "role": "assistant"}},
                raise_for_status=MagicMock(),
            )
            result = chat_json('Test', schema)
            assert result == {"ok": True}

    @pytest.mark.parametrize("invalid", ["not json", '{"answer": "wrong type"}'])
    def test_invalid_then_valid_retries(self, invalid):
        """First call returns bad JSON, second returns valid."""
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
        with patch("agent.llm.httpx.post") as mock:
            side_effects = [
                MagicMock(
                    status_code=200,
                    json=lambda: {"message": {"content": invalid, "role": "assistant"}},
                    raise_for_status=MagicMock(),
                ),
                MagicMock(
                    status_code=200,
                    json=lambda: {"message": {"content": '{"answer": 42}', "role": "assistant"}},
                    raise_for_status=MagicMock(),
                ),
            ]
            mock.side_effect = side_effects
            result = chat_json('Test', schema)
            assert result == {"answer": 42}
            assert mock.call_count == 2

    def test_invalid_twice_raises(self):
        """Two invalid calls raise LLMOutputError."""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        with patch("agent.llm.httpx.post") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {"message": {"content": '{invalid}', "role": "assistant"}},
                raise_for_status=MagicMock(),
            )
            with pytest.raises(LLMOutputError):
                chat_json('Test', schema)

    def test_connection_error_reported(self):
        """HTTP error produces clear message."""
        schema = {"type": "object", "properties": {}}
        with patch("agent.llm.httpx.post") as mock:
            mock.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(LLMOutputError) as exc_info:
                chat_json('Test', schema)
            assert "unreachable" in str(exc_info.value).lower()
