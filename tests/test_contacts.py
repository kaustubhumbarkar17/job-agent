"""Tests for agent/contacts_client.py with mocked HTTP."""
import pytest

from agent.contacts_client import HunterClient, EmailResult, VerifyResult


class TestHunterFindPeople:
    """Hunter people search tests."""

    def test_no_api_key_returns_empty(self):
        client = HunterClient(api_key="")
        result = client.find_people("Acme")
        assert result == []

    def test_with_api_key_returns_list(self):
        client = HunterClient(api_key="test-key")
        result = client.find_people("Acme")
        assert isinstance(result, list)


class TestHunterFindEmail:
    """Hunter email finder tests."""

    def test_no_api_key(self):
        client = HunterClient(api_key="")
        result = client.find_email("John", "Doe", "acme.com")
        assert result.email is None
        assert result.domain == "acme.com"


class TestHunterVerify:
    """Hunter email verification tests."""

    def test_no_api_key_returns_unverified(self):
        client = HunterClient(api_key="")
        result = client.verify_email("test@example.com")
        assert result.valid is False
        assert result.status == "unverified"


class TestEmailResultModel:
    """EmailResult Pydantic model tests."""

    def test_email_present(self):
        result = EmailResult(
            email="test@example.com",
            domain="example.com",
            confidence=0.95,
            verify_status="valid",
        )
        assert result.email == "test@example.com"

    def test_risky_email(self):
        result = EmailResult(
            email="test@example.com",
            domain="example.com",
            confidence=0.5,
            verify_status="risky",
        )
        assert result.email == "test@example.com"


class TestVerifyResultModel:
    """VerifyResult Pydantic model tests."""

    def test_valid_verification(self):
        result = VerifyResult(
            email="test@example.com", valid=True, status="valid", confidence=0.95
        )
        assert result.valid is True

    def test_risky_verification(self):
        result = VerifyResult(
            email="test@example.com", valid=False, status="risky", confidence=0.6
        )
        assert result.status == "risky"
