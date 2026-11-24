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
| `reconcile-draft <id>` | Resolve an interrupted Gmail draft creation after reviewing Gmail Drafts |
| `status` | Show job counts per status |
| `mark-sent <draft-id>` | Record a send after sending manually in Gmail |
| `mark-replied <draft-id>` | Record a reply and remove the draft from follow-ups |
| `followups [--days 5]` | List manually sent drafts at least this old with no recorded reply |
| `eval [--synthetic]` | Compare model rankings with independently labeled jobs |
| `db-init` | Initialize the database |

## Architecture

```
Discover → Pick → Scrape → Score → Shortlist → Contacts → Draft → Gmail
```

Each stage is an idempotent function that reads from and writes to SQLite. The LLM is called only for judgment tasks (scoring, drafting). Nothing is sent automatically — you review and send from Gmail Drafts.

## Configuration

Edit `config.yaml` for search terms, scoring weights, and draft limits.
Secrets go in `.env` (gitignored).

For Firecrawl Cloud, leave `FIRECRAWL_BASE_URL` blank and set `FIRECRAWL_API_KEY`. To use your own Firecrawl server for discovery and scraping, set `FIRECRAWL_BASE_URL=http://localhost:3002` in `.env`. Use the server's base URL, without `/v2`. Set `FIRECRAWL_SELF_HOSTED_API_KEY` only if that server requires a bearer token. The selected endpoint must support v2 search and structured JSON scraping. A self-hosted request failure stays on that endpoint, and the cloud key is reserved for Firecrawl Cloud.

## Scoring evaluation

Run `python -m agent.cli eval --synthetic` to exercise the harness offline. For a real evaluation, save 15–20 job descriptions as separate JSON files in `data/eval/`. Each file contains a `job` object with a `title` and the relevant posting details, a `label_score` integer from 0 to 100, and a short `label_note` explaining your judgment. Label the cases before seeing any model output.

With your completed `profile.md` and Ollama running, use `python -m agent.cli eval`. Use `--model` to compare another installed chat model. The report scores every label with the pipeline rubric and shows Spearman correlation, top-five overlap, mean absolute error, and the five largest disagreements. Equal scores share an average Spearman rank. The top-five comparison breaks ties in filename order. The MVP targets are Spearman ≥ 0.70 and at least three shared top-five jobs.

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

If drafting reports an uncertain Gmail outcome, compare the local copy in `out/drafts/<id>.txt` with Gmail Drafts. If you find a match, run `python -m agent.cli reconcile-draft <id> --gmail-draft-id <gmail-id>`. If you have confirmed that no matching draft exists, use `--confirmed-missing` to allow the next drafting run to retry it.
