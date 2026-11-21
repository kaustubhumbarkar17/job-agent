PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
  run_id     TEXT PRIMARY KEY,
  started_at TEXT DEFAULT CURRENT_TIMESTAMP,
  window     TEXT,
  notes      TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  url_hash      TEXT UNIQUE NOT NULL,
  url           TEXT NOT NULL,
  title         TEXT,
  company       TEXT,
  location      TEXT,
  source        TEXT,
  posted_at     TEXT,
  discovered_at TEXT DEFAULT CURRENT_TIMESTAMP,
  status        TEXT NOT NULL DEFAULT 'discovered',
  status_reason TEXT,
  detail_json   TEXT,
  embed_sim     REAL,
  score         INTEGER,
  score_json    TEXT,
  manual_note   TEXT,
  run_id        TEXT  -- soft reference; app code ensures runs row exists
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_company_title ON jobs(company, title, location);

CREATE TABLE IF NOT EXISTS embeddings (
  kind      TEXT NOT NULL,           -- 'profile_chunk' | 'job'
  ref_id    INTEGER NOT NULL,
  model     TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  vector    BLOB NOT NULL,           -- float32 bytes
  PRIMARY KEY (kind, ref_id, model)
);

CREATE TABLE IF NOT EXISTS contacts (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id        INTEGER NOT NULL REFERENCES jobs(id),
  company       TEXT,
  name          TEXT,
  role          TEXT,
  source_url    TEXT,
  email         TEXT,
  email_origin  TEXT,              -- api|pattern|posting
  verify_status TEXT DEFAULT 'unverified',
  confidence    REAL DEFAULT 0,
  UNIQUE(job_id, name)
);

CREATE TABLE IF NOT EXISTS drafts (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id         INTEGER NOT NULL REFERENCES jobs(id),
  contact_id     INTEGER NOT NULL REFERENCES contacts(id),
  subject        TEXT,
  body           TEXT,
  gmail_draft_id TEXT,
  status         TEXT DEFAULT 'created',
  created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
  sent_at        TEXT,
  UNIQUE(job_id, contact_id)
);

CREATE TABLE IF NOT EXISTS api_calls (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  ts          TEXT DEFAULT CURRENT_TIMESTAMP,
  service     TEXT,
  endpoint    TEXT,
  credits     REAL,
  latency_ms  INTEGER,
  ok          INTEGER,
  run_id      TEXT
);
