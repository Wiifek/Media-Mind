from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data/recordings.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key       TEXT NOT NULL UNIQUE,   -- hash of source content/URL, see db_keys.py
    source_name     TEXT NOT NULL,          -- filename or URL label; audio, video, or meeting recording
    transcript      TEXT,
    title           TEXT,
    summary         TEXT,
    action_items    TEXT,                   -- stored as JSON array
    key_decisions   TEXT,                   -- stored as JSON array
    open_questions  TEXT,                   -- stored as JSON array
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_source_name ON recordings(source_name);
"""


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the table (and its parent folder) if they don't exist yet."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_connection(db_path: str | Path = DEFAULT_DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()