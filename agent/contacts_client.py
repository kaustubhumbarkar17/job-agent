"""Contacts client interface for email discovery.

Uses Hunter.io as the default provider via a Protocol-based abstraction.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

from pydantic import BaseModel


class Person(BaseModel):
    """A contact person found for a company."""
    name: str
    role: str
    company: str
    source_url: str  # Where we found this person


class EmailResult(BaseModel):
    """An email address with confidence score."""
    email: Optional[str] = None
    domain: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    confidence: float = 0.0
    verify_status: str = "unverified"  # valid|risky|invalid|unverified


class VerifyResult(BaseModel):
    """Verification result from an email verifier API."""
    email: str
    valid: bool
    status: str  # valid|risky|invalid
    confidence: float = 0.0


class HunterProtocol(Protocol):
    """Interface for email finder providers."""

    def find_people(self, company: str, domain: Optional[str] = None,
                    roles: list[str] | None = None) -> list[Person]:
        ...

    def find_email(self, first_name: str, last_name: str,
                   domain: str) -> EmailResult:
        ...

    def verify_email(self, email: str) -> VerifyResult:
        ...


class HunterClient:
    """Hunter.io implementation (placeholder - requires API key)."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.hunter.io/v2"

    def find_people(self, company: str, domain: Optional[str] = None,
                    roles: list[str] | None = None) -> list[Person]:
        """Search for people at a company. Returns empty list until API key is set."""
        if not self.api_key:
            return []
        # TODO: Implement actual Hunter API call
        return []

    def find_email(self, first_name: str, last_name: str,
                   domain: str) -> EmailResult:
        """Find email for a person at a domain."""
        if not self.api_key:
            return EmailResult(email=None, domain=domain)
        # TODO: Implement actual Hunter API call
        return EmailResult(email=None, domain=domain, confidence=0.0)

    def verify_email(self, email: str) -> VerifyResult:
        """Verify an email address."""
        if not self.api_key:
            return VerifyResult(email=email, valid=False, status="unverified")
        # TODO: Implement actual Hunter verification API call
        return VerifyResult(email=email, valid=False, status="unverified")


def get_client(api_key: str | None = None) -> HunterClient:
    """Return a HunterClient."""
    key = api_key or ""
    return HunterClient(key)
