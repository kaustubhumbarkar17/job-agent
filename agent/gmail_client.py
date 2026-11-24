"""Create Gmail drafts through the compose scope."""
from __future__ import annotations

import base64
import mimetypes
import os
from email.message import EmailMessage
from pathlib import Path
from sqlite3 import Connection
from time import perf_counter
from typing import TYPE_CHECKING

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from httplib2 import HttpLib2Error
from oauthlib.oauth2 import OAuth2Error

from agent.config import get_settings
from agent.db import log_api_call

if TYPE_CHECKING:
    from googleapiclient.discovery import Resource


SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]


class DraftRejected(RuntimeError):
    """Draft preparation failed or Gmail explicitly rejected the request."""


class DraftUncertain(RuntimeError):
    """Gmail may have created a draft, but its ID is unavailable."""


def _get_service() -> Resource:
    """Build a Gmail service with cached desktop OAuth credentials."""
    settings = get_settings()
    token_path = Path(settings.google_token_path)
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES) if token_path.exists() else None
    if not creds or not creds.valid:
        if creds and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(settings.google_credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def create_draft(
    to: str, subject: str, body: str, attachment_path: str | None = None,
    *, conn: Connection | None = None, run_id: str = "",
) -> str:
    """Create a MIME draft and return Gmail's draft ID."""
    try:
        msg = EmailMessage()
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        if attachment_path:
            path = Path(attachment_path)
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            maintype, subtype = mime.split("/", 1)
            msg.add_attachment(path.read_bytes(), maintype, subtype, filename=path.name)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        request = _get_service().users().drafts().create(userId="me", body={"message": {"raw": raw}})
    except (OSError, ValueError, GoogleAuthError, HttpError, HttpLib2Error, OAuth2Error) as exc:
        raise DraftRejected(f"Could not prepare Gmail draft ({type(exc).__name__})") from exc

    started = perf_counter()
    ok = False
    try:
        draft = request.execute(num_retries=0)
        draft_id = draft.get("id") if isinstance(draft, dict) else None
        if not isinstance(draft_id, str) or not draft_id.strip():
            raise DraftUncertain("Gmail did not return a draft ID")
        ok = True
        return draft_id
    except HttpError as exc:
        if exc.resp.status in {400, 401, 403, 404, 422, 429}:
            raise DraftRejected(f"Gmail rejected the draft (HTTP {exc.resp.status})") from exc
        raise DraftUncertain(f"Gmail returned HTTP {exc.resp.status} without a draft ID") from exc
    except (OSError, ValueError, GoogleAuthError, HttpLib2Error) as exc:
        raise DraftUncertain("Gmail did not return a draft ID") from exc
    finally:
        if conn is not None:
            log_api_call(
                conn, "gmail", "drafts.create", latency_ms=int((perf_counter() - started) * 1000),
                ok=ok, run_id=run_id,
            )


def create_draft_raw(to: str, subject: str, body: str) -> str:
    """Create a text draft without an attachment."""
    return create_draft(to, subject, body)
