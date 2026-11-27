"""Hunter client for sourced professional contacts and email verification."""
from __future__ import annotations

import re
from sqlite3 import Connection
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from agent.config import get_settings
from agent.db import log_api_call


class HunterError(RuntimeError):
    """A Hunter request failed or returned unusable data."""


class HunterSuppressed(HunterError):
    """Hunter declined to process the contact."""


class Person(BaseModel):
    name: str
    role: str
    company: str
    source_url: str
    email: str | None = None
    confidence: float = 0.0
    email_origin: str | None = None


class EmailResult(BaseModel):
    email: str | None = None
    domain: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    confidence: float = 0.0
    verify_status: str = "unverified"


class VerifyResult(BaseModel):
    email: str
    valid: bool
    status: str
    confidence: float = 0.0


class HunterProtocol(Protocol):
    def find_people(
        self, company: str, domain: str | None = None, roles: list[str] | None = None,
    ) -> list[Person]: ...

    def find_email(self, first: str, last: str, domain: str) -> EmailResult: ...

    def verify_email(self, email: str) -> VerifyResult: ...


def clean_email(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    name, separator, domain = value.rpartition("@")
    if (separator and name and "." in domain and "@" not in name
            and not any(char.isspace() or char in "<>,;" for char in value)):
        return value
    return None


def role_priority(position: str, roles: list[str]) -> int:
    title = " ".join(position.casefold().split())
    for index, role in enumerate(roles):
        target = " ".join(role.casefold().split())
        if target and re.search(rf"(?<!\w){re.escape(target)}(?!\w)", title):
            return index
    return len(roles)


def _confidence(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool) and 0 <= value <= 100:
        return value / 100
    return 0.0


def _source_url(sources: object) -> str | None:
    if not isinstance(sources, list):
        return None
    for source in sources:
        url = source.get("uri") if isinstance(source, dict) else None
        if isinstance(url, str):
            try:
                parsed = urlsplit(url)
                if parsed.scheme in {"http", "https"} and parsed.hostname:
                    return url
            except ValueError:
                continue
    return None


def is_eligible(email: str | None, status: str, confidence: float, threshold: float) -> bool:
    """Use the same eligibility rule when verifying and drafting."""
    return clean_email(email) is not None and (
        status == "valid" or (status == "risky" and confidence >= threshold)
    )


class HunterClient:
    """Hunter v2 domain search, email finder and verifier."""

    def __init__(
        self, api_key: str, *, conn: Connection | None = None, run_id: str = "",
    ):
        self.api_key = api_key
        self.base_url = "https://api.hunter.io/v2"
        self._conn = conn
        self._run_id = run_id

    def _get(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        if not self.api_key:
            raise HunterError("HUNTER_API_KEY is not set")
        started = perf_counter()
        ok = False
        credits = 0.0
        try:
            response = httpx.get(
                f"{self.base_url}/{endpoint}", params=params,
                headers={"X-API-KEY": self.api_key}, timeout=30,
            )
            try:
                credits = float(response.headers.get("X-Credits-Used", "0"))
            except ValueError:
                pass
            if response.status_code == 451:
                raise HunterSuppressed("Hunter declined to process this contact")
            if endpoint == "email-verifier" and response.status_code in {202, 222}:
                return {}
            if response.status_code >= 400:
                raise HunterError(f"Hunter {endpoint} returned HTTP {response.status_code}")
            envelope = response.json()
            data = envelope.get("data") if isinstance(envelope, dict) else None
            if not isinstance(data, dict):
                raise HunterError(f"Hunter {endpoint} returned invalid data")
            ok = True
            return data
        except httpx.HTTPError:
            raise HunterError(f"Hunter {endpoint} request failed") from None
        except ValueError:
            raise HunterError(f"Hunter {endpoint} returned invalid JSON") from None
        finally:
            if self._conn is not None:
                log_api_call(
                    self._conn, "hunter", f"/v2/{endpoint}", credits,
                    int((perf_counter() - started) * 1000), ok, self._run_id,
                )

    def find_people(
        self, company: str, domain: str | None = None, roles: list[str] | None = None,
    ) -> list[Person]:
        if not domain and not company:
            return []
        data = self._get("domain-search", {"domain": domain} if domain else {"company": company})
        emails = data.get("emails")
        if not isinstance(emails, list):
            raise HunterError("Hunter domain search returned invalid contacts")

        people = []
        for item in emails:
            if not isinstance(item, dict) or item.get("type") == "generic":
                continue
            parts = (item.get("first_name"), item.get("last_name"), item.get("position"))
            if any(not isinstance(part, str) or not part.strip() for part in parts):
                continue
            first, last, role = (part.strip() for part in parts)
            if roles and role_priority(role, roles) == len(roles):
                continue
            source = _source_url(item.get("sources"))
            if source is None:
                continue
            email = clean_email(item.get("value"))
            people.append(Person(
                name=f"{first} {last}", role=role, company=company, source_url=source,
                email=email, confidence=_confidence(item.get("confidence")),
                email_origin="api" if email else None,
            ))
        return sorted(people, key=lambda person: (role_priority(person.role, roles or []), -person.confidence))

    def find_email(self, first: str, last: str, domain: str) -> EmailResult:
        data = self._get("email-finder", {"first_name": first, "last_name": last, "domain": domain})
        return EmailResult(
            email=clean_email(data.get("email")), domain=domain, first_name=first, last_name=last,
            confidence=_confidence(data.get("score")),
        )

    def verify_email(self, email: str) -> VerifyResult:
        data = self._get("email-verifier", {"email": email})
        raw_status = data.get("status")
        returned = clean_email(data.get("email"))
        if data and (returned is None or returned.casefold() != email.casefold()):
            raise HunterError("Hunter verified a different address")
        if (raw_status in ("invalid", "disposable", "webmail")
                or data.get("disposable") or data.get("webmail")):
            status = "invalid"
        elif raw_status == "accept_all" or data.get("accept_all") is True:
            status = "risky"
        elif raw_status == "valid" and not data.get("block"):
            status = "valid"
        else:
            status = "unverified"
        return VerifyResult(
            email=email, valid=status == "valid", status=status,
            confidence=_confidence(data.get("score")),
        )


def get_client(
    api_key: str | None = None, *, conn: Connection | None = None, run_id: str = "",
) -> HunterProtocol:
    return HunterClient(
        get_settings().hunter_api_key if api_key is None else api_key, conn=conn, run_id=run_id,
    )
