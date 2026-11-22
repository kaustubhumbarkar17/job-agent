# Job-Hunt Agent

Discover, score, and draft personalized emails for job applications — powered by a local LLM.

## Quickstart

```bash
cd job-agent
source .venv/bin/activate

# 1. Set up your environment
cp .env.example .env
# Edit .env: add FIRECRAWL_API_KEY, HUNTER_API_KEY, etc.

# 2. Initialize the database
python -m agent.cli db-init

# 3. Create your profile
cp profile.template.md profile.md
# Edit profile.md with your skills, experience, preferences

# 4. Run the pipeline
python -m agent.cli run --dry-run --yes   # Full dry run (no external calls)
python -m agent.cli discover              # Just discover jobs
python -m agent.cli score                # Score discovered jobs
python -m agent.cli draft                # Draft emails (requires Gmail OAuth)
```

## Commands

| Command | Description |
|---|---|
| `run [--dry-run] [--yes]` | Full pipeline with checkpoints |
| `discover` | Search for job postings |
| `pick [--yes]` | Select jobs to deep-scrape |
| `scrape` | Extract structured data from postings |
| `score` | Rank jobs against your profile |
| `shortlist [--yes]` | Select jobs to pursue |
| `contacts` | Find hiring contacts |
| `draft` | Generate email drafts as Gmail drafts |
| `status` | Show job counts per status |
| `db-init` | Initialize the database |

## Architecture

```
Discover → Pick → Scrape → Score → Shortlist → Contacts → Draft → Gmail
```

Each stage is an idempotent function that reads from and writes to SQLite. The LLM is called only for judgment tasks (scoring, drafting). Nothing is sent automatically — you review and send from Gmail Drafts.

## Configuration

Edit `config.yaml` for search terms, scoring weights, and draft limits.
Secrets go in `.env` (gitignored).

## Development

```bash
pip install -e ".[dev]"
pytest -v          # Run all tests
ruff check .       # Lint
```

## Gmail Setup

1. Create a Google Cloud project
2. Enable the Gmail API
3. Download an OAuth 2.0 Desktop client `credentials.json`
4. Place it at `credentials.json` in this directory
5. Run `python -m agent.cli draft` — it opens a browser for consent
