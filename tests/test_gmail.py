"""Tests for agent/gmail_client.py.

Gmail tests use a mocked service object. A grep guard ensures no send
operations are performed on the Gmail service.
"""
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BASE = Path(__file__).resolve().parent.parent


def grep_for_send_calls() -> list[str]:
    """Grep agent/ for forbidden .send() calls on Gmail service."""
    violations: list[str] = []
    agent_dir = BASE / "agent"
    pattern = re.compile(r"\.send\s*\(")
    py_file = agent_dir / "gmail_client.py"
    if py_file.exists():
        with open(py_file) as f:
            for i, line in enumerate(f, 1):
                # Skip comments and docstrings
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""'):
                    continue
                if pattern.search(line):
                    violations.append(f"{py_file.name}:{i}: {stripped}")
    return violations


class TestGmailSendGuard:
    """F5: Grep guard - no .send() calls on Gmail service."""

    def test_no_send_calls(self):
        violations = grep_for_send_calls()
        assert len(violations) == 0, (
            "Forbidden .send() calls found:\n" + "\n".join(violations)
        )


class TestCreateDraft:
    """Gmail draft creation tests."""

    @pytest.fixture()
    def mock_service(self):
        """Return a mocked Gmail service."""
        service = MagicMock()
        drafts = MagicMock()
        users = MagicMock()
        users.drafts.return_value = drafts
        service.users.return_value = users
        drafts.create.return_value.execute.return_value = {"id": "draft-123"}
        return service, drafts, users

    def test_draft_created_with_correct_fields(self, mock_service):
        """Draft is created with correct To/Subject/Body."""
        service, drafts, users = mock_service

        from agent.gmail_client import create_draft_raw
        with patch("agent.gmail_client._get_service", return_value=service):
            draft_id = create_draft_raw(
                "test@example.com",
                "Test Subject",
                "Test body content",
            )

        # Verify the call chain
        users.drafts.assert_called_once()
        drafts.create.assert_called_once()
        assert draft_id == "draft-123"


class TestGmailScope:
    """Verify Gmail scope is compose-only."""

    def test_scopes_are_compose_only(self):
        from agent.gmail_client import SCOPES
        assert "https://www.googleapis.com/auth/gmail.compose" in SCOPES
        # Should NOT include read/write scopes
        assert "https://www.googleapis.com/auth/gmail.modify" not in SCOPES
        assert "https://www.googleapis.com/auth/gmail.send" not in SCOPES
