"""Hunter response mapping and contact-stage regression tests."""
from contextlib import closing
from unittest.mock import Mock

import httpx
import pytest

from agent.config import get_settings
from agent.contacts_client import (
    EmailResult,
    HunterClient,
    HunterError,
    Person,
    VerifyResult,
)
from agent.db import connect, upsert_job


def test_hunter_requires_credentials_before_sending_request(monkeypatch):
    get = Mock()
    monkeypatch.setattr("agent.contacts_client.httpx.get", get)
    with pytest.raises(HunterError, match="HUNTER_API_KEY"):
        HunterClient("").find_people("Acme")
    get.assert_not_called()


def test_domain_search_filters_unsourced_and_generic_contacts(monkeypatch, tmp_path):
    source = {"uri": "https://example.com/team"}
    rows = [
        {"first_name": "Mira", "last_name": "Shah", "position": "Engineering Manager",
         "sources": [source], "value": "mira@example.com", "confidence": 82},
        {"first_name": "Rae", "last_name": "Diaz", "position": "Recruiter",
         "sources": [source], "value": "rae@example.com", "confidence": 92},
        {"first_name": "Ari", "last_name": "Patel", "position": "Co-Founder & CTO",
         "sources": [source], "value": "ari@example.com", "confidence": 80},
        {"first_name": "Sam", "last_name": "Rivera", "position": "Co-CTO",
         "sources": [source], "value": "sam@example.com", "confidence": 78},
        {"first_name": "Managing", "last_name": "Director", "position": "Managing Director",
         "sources": [source], "value": "managing@example.com", "confidence": 100},
        {"first_name": "Solutions", "last_name": "Director", "position": "Director of Solutions Architecture",
         "sources": [source], "value": "solutions@example.com", "confidence": 100},
        {"first_name": "No", "last_name": "Source", "position": "Recruiter", "sources": []},
        {"first_name": "Generic", "last_name": "Inbox", "position": "Recruiter",
         "sources": [source], "type": "generic"},
    ]
    get = Mock(return_value=httpx.Response(
        200, headers={"X-Credits-Used": "1"}, json={"data": {"emails": rows}},
    ))
    monkeypatch.setattr("agent.contacts_client.httpx.get", get)
    with closing(connect(tmp_path / "jobs.db")) as conn:
        results = HunterClient("test-key", conn=conn, run_id="run-1").find_people(
            "Acme", "example.com", ["Recruiter", "Engineering Manager", "cto"],
        )
        assert [person.name for person in results] == ["Rae Diaz", "Mira Shah", "Ari Patel", "Sam Rivera"]
        assert results[0].source_url == source["uri"]
        assert results[0].confidence == 0.92
        assert get.call_args.kwargs["params"] == {"domain": "example.com"}
        assert get.call_args.kwargs["headers"] == {"X-API-KEY": "test-key"}
        assert tuple(conn.execute(
            "SELECT service, endpoint, credits, ok, run_id FROM api_calls",
        ).fetchone()) == ("hunter", "/v2/domain-search", 1, 1, "run-1")


@pytest.mark.parametrize(("raw", "expected"), [
    ("valid", "valid"), ("invalid", "invalid"), ("accept_all", "risky"), ("unknown", "unverified"),
])
def test_finder_and_verifier_response_mapping(monkeypatch, raw, expected):
    email = "rae@example.com"
    get = Mock(side_effect=[
        httpx.Response(200, json={"data": {"email": email, "score": 84}}),
        httpx.Response(200, json={"data": {"email": email, "status": raw, "score": 80}}),
    ])
    monkeypatch.setattr("agent.contacts_client.httpx.get", get)
    client = HunterClient("test-key")
    assert client.find_email("Rae", "Diaz", "example.com").confidence == 0.84
    verified = client.verify_email(email)
    assert (verified.status, verified.valid, verified.confidence) == (
        expected, expected == "valid", 0.8,
    )


@pytest.mark.usefixtures("offline_pipeline")
def test_linkedin_contacts_reach_verifier_without_hunter_lookup(tmp_path, monkeypatch):
    from agent.stages import contacts, verify

    settings = get_settings()
    settings.firecrawl_base_url = "http://localhost:3002"
    settings.search.max_queries_per_run = 1
    recruiter_url = "https://www.linkedin.com/posts/recruiter_hiring-1"
    manager_url = "https://www.linkedin.com/posts/manager_hiring-2"
    duplicate_url = "https://www.linkedin.com/posts/recruiter_hiring-3"
    fabricated_url = "https://www.linkedin.com/posts/recruiter_hiring-4"
    unsupported_url = "https://example.org/posts/elsewhere"

    def response(data):
        return httpx.Response(
            200, json={"success": True, "data": data}, request=httpx.Request("POST", "http://localhost"),
        )

    def post(author, role, email, markdown):
        return response({
            "json": {"name": author, "role": role, "email": email, "company": "Acme", "quote": f"Apply at {email}"},
            "markdown": markdown,
        })

    firecrawl = Mock(side_effect=[
        response({"web": [{"title": "Hiring", "url": url} for url in
                          (unsupported_url, recruiter_url, manager_url, duplicate_url, fabricated_url)]}),
        post("Ari Patel", "Talent Acquisition Specialist", "ari@acme.example", "Ari Patel: Apply at ari@acme.example."),
        post("Sam Rivera", "Engineering Manager", "sam@acme.example",
             "Sam Rivera: Apply at [sam@acme.example](mailto:sam@acme.example)."),
        post("Ari Patel", "Talent Acquisition Specialist", "ari@acme.example", "Ari Patel: Apply at ari@acme.example."),
        post("Pat Lee", "Hiring Manager", "pat@acme.example", "Pat Lee: Apply on our website."),
    ])
    hunter = Mock()
    hunter.verify_email.side_effect = lambda email: VerifyResult(email=email, valid=True, status="valid")
    monkeypatch.setattr("agent.firecrawl_client.httpx.post", firecrawl)
    monkeypatch.setattr(contacts, "get_client", lambda **kwargs: hunter)
    monkeypatch.setattr(verify, "get_client", lambda **kwargs: hunter)

    db_path = tmp_path / "public.db"
    with closing(connect(db_path)) as conn, conn:
        job_id = upsert_job(conn, "https://acme.example/jobs/1", company="Acme")
        conn.execute("UPDATE jobs SET status = 'shortlisted', detail_json = ? WHERE id = ?",
                     ('{"company_domain":"https://www.acme.example/"}', job_id))
        conn.execute("INSERT INTO api_calls (service, endpoint, run_id) VALUES ('firecrawl', '/v2/search', 'one')")
    assert contacts.run(db_path=db_path, run_id="one")["found"] == 1
    assert verify.run(db_path=db_path, run_id="one")["verified"] == 1
    hunter.find_people.assert_not_called()
    hunter.find_email.assert_not_called()
    assert [call.args[0] for call in hunter.verify_email.call_args_list] == ["ari@acme.example", "sam@acme.example"]
    assert [call.args[0] for call in firecrawl.call_args_list] == ["http://localhost:3002/v2/search"] + [
        "http://localhost:3002/v2/scrape",
    ] * 4
    assert [call.kwargs["json"]["url"] for call in firecrawl.call_args_list[1:]] == [
        recruiter_url, manager_url, duplicate_url, fabricated_url,
    ]
    with closing(connect(db_path)) as conn:
        assert [tuple(row) for row in conn.execute("SELECT source_url, email_origin FROM contacts ORDER BY id")] == [
            (recruiter_url, "posting"), (manager_url, "posting"),
        ]


@pytest.mark.usefixtures("offline_pipeline")
def test_linkedin_search_falls_back_to_hunter_if_unavailable(tmp_path, monkeypatch):
    from agent.stages import contacts

    get_settings().firecrawl_base_url = "http://localhost:3002"
    search = Mock(return_value=httpx.Response(403, request=httpx.Request("POST", "http://localhost")))
    hunter = Mock()
    hunter.find_people.return_value = [Person(
        name="Ari Patel", role="Recruiter", company="Acme", source_url="https://acme.example/about",
    )]
    monkeypatch.setattr("agent.firecrawl_client.httpx.post", search)
    monkeypatch.setattr(contacts, "get_client", lambda **kwargs: hunter)
    db_path = tmp_path / "fallback.db"
    with closing(connect(db_path)) as conn, conn:
        job_id = upsert_job(conn, "https://acme.example/jobs/1", company="Acme")
        conn.execute("UPDATE jobs SET status = 'shortlisted', detail_json = ? WHERE id = ?",
                     ('{"company_domain":"acme.example"}', job_id))

    assert contacts.run(db_path=db_path)["found"] == 1
    search.assert_called_once()
    hunter.find_people.assert_called_once_with("Acme", "acme.example", get_settings().contacts.target_roles)


@pytest.mark.usefixtures("offline_pipeline")
def test_stages_persist_provenance_and_gate_risky_contacts(tmp_path, monkeypatch):
    from agent.stages import contacts, verify

    client = Mock()
    client.find_people.side_effect = [
        [Person(name="Pat Grey", role="Recruiter", company="Company", source_url="https://example.com/team")],
        [],
    ]
    client.find_email.return_value = EmailResult(email="pat@example.com", confidence=0.9)
    client.verify_email.side_effect = [
        VerifyResult(email="pat@example.com", valid=False, status="risky", confidence=0.8),
        VerifyResult(email="low@example.com", valid=False, status="risky", confidence=0.79),
        VerifyResult(email="invalid@example.com", valid=False, status="invalid", confidence=1.0),
        VerifyResult(email="valid@example.com", valid=True, status="valid", confidence=0.4),
    ]
    monkeypatch.setattr(contacts, "get_client", lambda **kwargs: client)
    monkeypatch.setattr(verify, "get_client", lambda **kwargs: client)
    db_path = tmp_path / "stages.db"

    with closing(connect(db_path)) as conn:
        first = upsert_job(conn, "https://example.com/job", company="Company")
        empty = upsert_job(conn, "https://example.com/empty", company="Company")
        conn.execute("UPDATE jobs SET status = 'shortlisted', detail_json = ?", ('{"company_domain":"example.com"}',))
        conn.commit()
        assert contacts.run(db_path=db_path)["found"] == 1
        assert contacts.run(db_path=db_path)["total"] == 0
        assert conn.execute("SELECT status FROM jobs WHERE id = ?", (empty,)).fetchone()[0] == "no_contact"

        for name in ("low", "invalid", "valid"):
            job_id = upsert_job(conn, f"https://example.com/{name}", company="Company")
            conn.execute("UPDATE jobs SET status = 'contacted' WHERE id = ?", (job_id,))
            conn.execute(
                "INSERT INTO contacts (job_id, name, role, source_url, email) VALUES (?, ?, ?, ?, ?)",
                (job_id, name, "Recruiter", "https://example.com/team", f"{name}@example.com"),
            )
        conn.commit()

        results = verify.run(db_path=db_path)
        assert (results["verified"], results["unverified"]) == (2, 2)
        assert dict(conn.execute("SELECT url, status FROM jobs")) == {
            "https://example.com/job": "verified", "https://example.com/empty": "no_contact",
            "https://example.com/low": "contacted", "https://example.com/invalid": "contacted",
            "https://example.com/valid": "verified",
        }
        stored = conn.execute(
            "SELECT name, source_url, email, email_origin, verify_status, confidence FROM contacts WHERE job_id = ?",
            (first,),
        ).fetchone()
        assert tuple(stored) == ("Pat Grey", "https://example.com/team", "pat@example.com", "api", "risky", 0.8)
        verify.run(db_path=db_path)
        assert client.verify_email.call_count == 4
        assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 0


@pytest.mark.usefixtures("offline_pipeline")
def test_verify_provider_failure_reuses_found_address_on_rerun(tmp_path, monkeypatch):
    from agent.stages import verify

    email = "candidate@example.com"
    client = Mock()
    client.find_email.return_value = EmailResult(email=email)
    client.verify_email.side_effect = [
        HunterError("Temporary verifier failure"),
        VerifyResult(email=email, valid=True, status="valid"),
    ]
    monkeypatch.setattr(verify, "get_client", lambda **kwargs: client)
    db_path = tmp_path / "recovery.db"
    with closing(connect(db_path)) as conn:
        job_id = upsert_job(conn, "https://example.com/job")
        with conn:
            conn.execute(
                "UPDATE jobs SET status = 'contacted', detail_json = ? WHERE id = ?",
                ('{"company_domain":"example.com"}', job_id),
            )
            conn.execute(
                "INSERT INTO contacts (job_id, name, role, source_url) VALUES (?, ?, ?, ?)",
                (job_id, "Candidate Recruiter", "Recruiter", "https://example.com/team"),
            )
        assert verify.run(db_path=db_path)["failed"] == 1
        assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "contacted"
        assert verify.run(db_path=db_path)["verified"] == 1
        client.find_email.assert_called_once_with("Candidate", "Recruiter", "example.com")
        assert client.verify_email.call_count == 2


@pytest.mark.usefixtures("offline_pipeline")
def test_cli_dry_run_persists_verified_fixture_contacts_without_hunter(monkeypatch):
    from typer.testing import CliRunner

    from agent.cli import app

    get = Mock(side_effect=AssertionError("Dry-run must never contact Hunter"))
    monkeypatch.setattr("agent.contacts_client.httpx.get", get)
    result = CliRunner().invoke(app, ["run", "--dry-run", "--yes", "--to", "verify"])
    assert result.exit_code == 0, result.output
    get.assert_not_called()
    with closing(connect("data/dry-run.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'verified'").fetchone()[0] > 0
        assert conn.execute(
            "SELECT COUNT(*) FROM contacts WHERE email = 'fixture@example.com' AND verify_status = 'valid'",
        ).fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 0
