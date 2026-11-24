"""Configuration: YAML defaults + .env overlay.

Priority: env vars > YAML > hardcoded defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    titles: List[str] = ["Full Stack Engineer", "Backend Engineer", "AI Engineer", "SDE 2"]
    locations: List[str] = ["Chennai", "Bangalore", "Remote India"]
    window: str = "7d"
    max_queries_per_run: int = 12
    results_per_query: int = 10
    blocklist_title_words: List[str] = []
    blocklist_companies: List[str] = []


class ScoringConfig(BaseModel):
    embed_keep_fraction: float = 0.4
    min_score_for_shortlist: int = 65
    weight_must_have_skills: int = 40
    weight_seniority: int = 25
    weight_domain: int = 20
    weight_location: int = 15

    @property
    def weights_dict(self) -> dict[str, int]:
        return {
            "must_have_skills": self.weight_must_have_skills,
            "seniority": self.weight_seniority,
            "domain": self.weight_domain,
            "location": self.weight_location,
        }

    def total_weight(self) -> int:
        return sum(self.weights_dict.values())


class ContactsConfig(BaseModel):
    target_roles: List[str] = [
        "Talent Acquisition", "Recruiter",
        "Engineering Manager", "Head of Engineering", "CTO",
    ]
    max_per_job: int = 3
    min_email_confidence: float = 0.8


class EmailConfig(BaseModel):
    max_drafts_per_day: int = 12
    max_words: int = 120
    resume_path: str = ""


class Settings(BaseModel):
    """Top-level settings. Loaded from config.yaml + .env overlay."""
    firecrawl_api_key: str = ""
    firecrawl_base_url: str = ""
    firecrawl_self_hosted_api_key: str = ""
    hunter_api_key: str = ""
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen3:14b"
    embed_model: str = "nomic-embed-text"
    google_credentials_path: str = "./credentials.json"
    google_token_path: str = "./token.json"
    db_path: str = "./data/jobs.db"

    search: SearchConfig = SearchConfig()
    scoring: ScoringConfig = ScoringConfig()
    contacts: ContactsConfig = ContactsConfig()
    email: EmailConfig = EmailConfig()

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Settings":
        """Load defaults from YAML file."""
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        kwargs = {}
        for section in ("search", "scoring", "contacts", "email"):
            if section in data:
                kwargs[section] = eval_section(section, data[section])
        return cls(**kwargs)

    @classmethod
    def from_yaml_and_env(cls, path: str = "config.yaml") -> "Settings":
        """Load YAML, then overlay env vars."""
        load_dotenv()
        settings = cls.from_yaml(path)
        # Overlay env vars
        overrides = {}
        for key in ("firecrawl_api_key", "firecrawl_base_url", "firecrawl_self_hosted_api_key",
                      "hunter_api_key", "ollama_host",
                      "llm_model", "embed_model", "google_credentials_path",
                      "google_token_path", "db_path"):
            env_val = os.environ.get(key.upper())
            if env_val:
                overrides[key] = env_val
        # List fields from .env are comma-separated
        for key in ("search_titles", "search_locations", "search_blocklist_title_words",
                      "search_blocklist_companies", "contacts_target_roles"):
            env_key = key.upper()
            env_val = os.environ.get(env_key)
            if env_val:
                overrides[key] = [v.strip() for v in env_val.split(",") if v.strip()]
        # Numeric overrides
        for key in ("search_max_queries_per_run", "search_results_per_query",
                      "scoring_embed_keep_fraction", "scoring_min_score_for_shortlist",
                      "contacts_max_per_job", "contacts_min_email_confidence",
                      "email_max_drafts_per_day", "email_max_words"):
            env_key = key.upper()
            env_val = os.environ.get(env_key)
            if env_val:
                overrides[key] = _parse_numeric(key, env_val)
        return cls(**overrides)


def _parse_numeric(field: str, value: str) -> float | int:
    """Parse a numeric env var, returning int or float."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def eval_section(name: str, data: dict) -> BaseModel:
    """Create a config sub-model from a YAML section dict."""
    cls_map = {
        "search": SearchConfig,
        "scoring": ScoringConfig,
        "contacts": ContactsConfig,
        "email": EmailConfig,
    }
    return cls_map[name](**data)


# Global singleton (loaded lazily on first access)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a fresh Settings instance, loading from config.yaml + .env."""
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml_and_env()
    return _settings


# Convenience alias
settings = get_settings()
