"""Tests for agent/gmail_client.py.

Gmail tests use a mocked service object. A grep guard ensures no send
operations are performed on the Gmail service.
"""
import base64
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BASE = Path(__file__).resolve().parent.parent


def grep_for_send_calls() -> list[str]:
    """Grep agent/ for forbidden .send() calls on Gmail service."""
    violations: list[str] = []
    agent_dir = BASE / "agent"
    pattern = re.compile(r"\.send\s*\(")
    for py_file in agent_dir.rglob("*.py"):
        with py_file.open(encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                # Skip comments and docstrings
                stripped = line.strip()
                if stripped.startswith(("#", '"""')):
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

    def test_mime_message_carries_resume_attachment(self, mock_service, tmp_path):
        from agent.gmail_client import create_draft

        service, drafts, _ = mock_service
        resume = tmp_path / "resume.pdf"
        resume.write_bytes(b"resume content")
        with patch("agent.gmail_client._get_service", return_value=service):
            assert create_draft("alex@example.com", "Subject", "Body", str(resume)) == "draft-123"

        raw = drafts.create.call_args.kwargs["body"]["message"]["raw"]
        message = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(raw))
        assert message["To"] == "alex@example.com"
        assert message["Subject"] == "Subject"
        assert message.get_body(preferencelist=("plain",)).get_content().strip() == "Body"
        attachment = next(message.iter_attachments())
        assert attachment.get_filename() == "resume.pdf"
        assert attachment.get_payload(decode=True) == b"resume content"
        drafts.create.return_value.execute.assert_called_once_with(num_retries=0)

    def test_missing_gmail_id_is_uncertain(self, mock_service):
        from agent.gmail_client import DraftUncertain, create_draft

        service, drafts, _ = mock_service
        drafts.create.return_value.execute.return_value = {}
        with patch("agent.gmail_client._get_service", return_value=service), pytest.raises(DraftUncertain):
            create_draft("alex@example.com", "Subject", "Body")

    def test_desktop_oauth_caches_token(self, tmp_path):
        from agent.config import Settings
        from agent.gmail_client import SCOPES, _get_service

        token = tmp_path / "token.json"
        settings = Settings(google_token_path=str(token), google_credentials_path="desktop.json")
        with (
            patch("agent.gmail_client.get_settings", return_value=settings),
            patch("agent.gmail_client.InstalledAppFlow.from_client_secrets_file") as factory,
            patch("agent.gmail_client.build") as build,
        ):
            flow = factory.return_value
            credentials = flow.run_local_server.return_value
            credentials.to_json.return_value = '{"token":"fixture"}'
            assert _get_service() is build.return_value

        factory.assert_called_once_with("desktop.json", SCOPES)
        flow.run_local_server.assert_called_once_with(port=0)
        assert token.read_text(encoding="utf-8") == '{"token":"fixture"}'
        assert token.stat().st_mode & 0o777 == 0o600


class TestGmailScope:
    """Verify Gmail scope is compose-only."""

    def test_scopes_are_compose_only(self):
        from agent.gmail_client import SCOPES
        assert "https://www.googleapis.com/auth/gmail.compose" in SCOPES
        # Should NOT include read/write scopes
        assert "https://www.googleapis.com/auth/gmail.modify" not in SCOPES
        assert "https://www.googleapis.com/auth/gmail.send" not in SCOPES
