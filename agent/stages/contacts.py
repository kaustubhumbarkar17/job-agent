"""Save sourced professional contacts for shortlisted jobs."""
from __future__ import annotations

import json
import re
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

from agent.config import get_settings
from agent.contacts_client import (
    HunterError,
    Person,
    clean_email,
    get_client,
    role_priority,
)
from agent.db import connect, jobs_by_status
from agent.firecrawl_client import BudgetExceeded, FirecrawlClient, FirecrawlError

LINKEDIN_POST_LIMIT = 5
POST_CONTACT_SCHEMA = {
    "type": "object",
    "description": (
        "Extract only the named author of this hiring post and the work email they explicitly "
        "publish for candidates. Ignore comments, related posts, and inferred email addresses. "
        "Leave fields empty if the post does not identify such a contact."
    ),
    "properties": {
        "name": {"type": "string", "description": "Full name of the post author"},
        "role": {"type": "string", "description": "The author's recruiting, HR, or hiring role"},
        "company": {"type": "string", "description": "Employer the author is hiring for"},
        "email": {"type": "string", "description": "Work email explicitly listed by the author for applicants"},
        "quote": {"type": "string", "description": "Verbatim excerpt where the author invites applicants to email"},
    },
    "required": ["name", "role", "company", "email", "quote"],
}


def _company_host(domain: str | None) -> str:
    if not isinstance(domain, str):
        return ""
    domain = domain.strip()
    return (urlsplit(domain if "://" in domain else f"//{domain}").hostname or "").removeprefix("www.")


def _post_text(markdown: str) -> str:
    text = re.sub(r"\[([^\]\n]+)\]\([^)]+\)", r"\1", markdown)
    return " ".join(text.casefold().split())


def _linkedin_contacts(client: FirecrawlClient, company: str, domain: str, roles: list[str]) -> list[Person]:
    query = f'"{company}" hiring site:linkedin.com/posts/'
    try:
        posts = client.search(query, limit=LINKEDIN_POST_LIMIT)
    except (FirecrawlError, BudgetExceeded):
        return []

    candidates = []
    for post in posts[:LINKEDIN_POST_LIMIT]:
        url = urlsplit(post.url)
        host = url.hostname or ""
        if (url.scheme != "https" or url.username is not None
                or not (host == "linkedin.com" or host.endswith(".linkedin.com"))
                or not url.path.startswith("/posts/")):
            continue
        try:
            contact, markdown = client.scrape_json_with_markdown(post.url, POST_CONTACT_SCHEMA)
        except FirecrawlError:
            continue

        name, role = contact["name"].strip(), contact["role"].strip()
        email = clean_email(contact["email"])
        if not name or role_priority(role, roles) == len(roles) or not email:
            continue
        email_domain = email.rsplit("@", 1)[1].casefold()
        if contact["company"].strip().casefold() != company.casefold() or email_domain != domain:
            continue
        source = _post_text(markdown)
        quote = _post_text(contact["quote"])
        email_pattern = rf"(?<![\w.+%-]){re.escape(email.casefold())}(?![\w-]|\.[\w-])"
        if not re.search(email_pattern, quote) or quote not in source or " ".join(name.casefold().split()) not in source:
            continue
        candidates.append(Person(
            name=name, role=role, company=company, source_url=post.url,
            email=email, email_origin="posting",
        ))
    return candidates


def run(
    db_path: str | Path | None = None, run_id: str = "", dry_run: bool = False, **kwargs,
) -> dict:
    """Collect published recruiting contacts before falling back to Hunter."""
    config = get_settings()
    settings = config.contacts
    with closing(connect(db_path)) as conn:
        jobs = jobs_by_status(conn, "shortlisted")
        client = None if dry_run else get_client(conn=conn, run_id=run_id)
        firecrawl = None
        if jobs and not dry_run:
            try:
                firecrawl = FirecrawlClient(conn=conn, run_id=run_id, search_limit=config.search.max_queries_per_run)
            except FirecrawlError:
                pass  # Hunter can still find contacts when Firecrawl is unavailable.
        published = {}
        found = no_contact = failed = 0
        for job in jobs:
            try:
                detail = json.loads(job["detail_json"] or "{}")
                if not isinstance(detail, dict):
                    raise TypeError("Job detail must be a JSON object")
                company = detail.get("company") or job["company"] or ""
                domain = detail.get("company_domain") or None
                candidates = []
                posting = detail.get("posting_contact")
                if isinstance(posting, dict):
                    name, role = posting.get("name"), posting.get("role")
                    if all(isinstance(value, str) and value.strip() for value in (name, role)):
                        email = clean_email(posting.get("email"))
                        candidates.append(Person(
                            name=name.strip(), role=role.strip(), company=company,
                            source_url=job["url"], email=email,
                            email_origin="posting" if email else None,
                        ))
                if len(candidates) < settings.max_per_job:
                    if dry_run:
                        candidates.append(Person(
                            name="Fixture Recruiter", role="Recruiter", company=company,
                            source_url=job["url"], email="fixture@example.com", email_origin="api",
                        ))
                    elif client is not None:
                        host = _company_host(domain)
                        if firecrawl is not None and company and host:
                            key = (company.casefold(), host)
                            if key not in published:
                                published[key] = _linkedin_contacts(firecrawl, company, host, settings.target_roles)
                            candidates.extend(published[key])
                        if not any(candidate.email for candidate in candidates):
                            candidates.extend(client.find_people(company, host or None, settings.target_roles))
            except (HunterError, TypeError, ValueError) as exc:
                conn.execute("UPDATE jobs SET status_reason = ? WHERE id = ?", (str(exc), job["id"]))
                conn.commit()
                failed += 1
                continue

            candidates.sort(key=lambda candidate: (
                candidate.email_origin != "posting", role_priority(candidate.role, settings.target_roles),
            ))
            names, addresses = set(), set()
            with conn:
                for candidate in candidates:
                    key = candidate.name.casefold()
                    address = candidate.email.casefold() if candidate.email else None
                    if key in names or address in addresses:
                        continue
                    if len(names) >= settings.max_per_job:
                        break
                    names.add(key)
                    if address:
                        addresses.add(address)
                    conn.execute(
                        """INSERT INTO contacts (job_id, company, name, role, source_url,
                                                email, email_origin, confidence)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(job_id, name) DO NOTHING""",
                        (job["id"], candidate.company, candidate.name, candidate.role,
                         candidate.source_url, candidate.email, candidate.email_origin,
                         candidate.confidence),
                    )
                status = "contacted" if names else "no_contact"
                conn.execute(
                    "UPDATE jobs SET status = ?, status_reason = ? WHERE id = ?",
                    (status, None if names else "No sourced contacts found", job["id"]),
                )
            found += bool(names)
            no_contact += not names

    return {"found": found, "no_contact": no_contact, "failed": failed, "total": len(jobs)}
