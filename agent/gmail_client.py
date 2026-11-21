"""Gmail client for creating drafts only.

IMPORTANT: This module ONLY calls gmail.compose scope.
It cannot read, modify, or send emails.
A test verifies that no send operations are performed on the Gmail service.
"""
from __future__ import annotations

import base64
import os
from email import message as email_module
from email.mime.application import MIMEApplication
from typing import TYPE_CHECKING, Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from agent.config import get_settings

if TYPE_CHECKING:
    from googleapiclient.discovery import Resource


SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]


def _get_service() -> "Resource":
    """Build a Gmail service with OAuth credentials."""
    creds: Optional[Credentials] = None
    cred_path = get_settings().google_credentials_path
    token_path = get_settings().google_token_path

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refreshable:
            creds.refresh(Request())
        else:
            if not os.path.exists(cred_path):
                raise FileNotFoundError(
                    f"Credentials file not found at {cred_path}. "
                    "Run OAuth flow first."
                )
            flow = InstalledAppFlow(
                client_file=cred_path,
                scopes=SCOPES,
            )
            creds = flow.run_console()

    return build("gmail", "v1", credentials=creds)


def create_draft(
    to: str,
    subject: str,
    body: str,
    attachment_path: Optional[str] = None,
) -> str:
    """Create a Gmail draft. Returns the draft ID."""
    service = _get_service()

    msg = email_module.EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    # Add attachment if provided
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            msg.attach(MIMEApplication(f.read(), Name=os.path.basename(attachment_path)))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    gmail_msg = {
        "message": {
            "raw": raw,
            "label_ids": ["DRAFT"],
        }
    }

    draft = service.users().drafts().create(userId="me", body=gmail_msg).execute()
    return draft["id"]


def create_draft_raw(
    to: str,
    subject: str,
    body: str,
) -> str:
    """Create a draft using a pre-built message. Simpler interface."""
    service = _get_service()

    msg_str = f"Subject: {subject}\nTo: {to}\nMIME-Version: 1.0\nContent-type: text/plain\n\n{body}"
    raw = base64.urlsafe_b64encode(msg_str.encode()).decode()

    gmail_msg = {"message": {"raw": raw, "label_ids": ["DRAFT"]}}
    draft = service.users().drafts().create(userId="me", body=gmail_msg).execute()
    return draft["id"]
