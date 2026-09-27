from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from db.schema import DEFAULT_DB_PATH, get_connection, init_db
from db.keys import make_key
from db.models import RecordingRecord


class RecordingDB:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        init_db(self.db_path)

    # Re-exposed here so callers only need one import for the common case:
    # RecordingDB.make_key(...) instead of also importing from db_keys.
    make_key = staticmethod(make_key)

    def exists(self, dedup_key: str) -> bool:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM recordings WHERE dedup_key = ?", (dedup_key,)
            ).fetchone()
        return row is not None

    def get(self, dedup_key: str) -> Optional[RecordingRecord]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM recordings WHERE dedup_key = ?", (dedup_key,)
            ).fetchone()
        return RecordingRecord.from_row(dict(row)) if row else None

    def get_by_name(self, source_name: str) -> list[RecordingRecord]:
        """Look up by name instead of hash — e.g. to show a user's re-uploads."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM recordings WHERE source_name = ? ORDER BY created_at DESC",
                (source_name,),
            ).fetchall()
        return [RecordingRecord.from_row(dict(r)) for r in rows]

    def list_all(self, limit: int = 50) -> list[dict]:
        """Lightweight listing (no transcript/summary) for a history view."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, source_name, title, created_at FROM recordings "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save(self, record: RecordingRecord) -> None:
        """Insert a new recording, or overwrite if this dedup_key already exists."""
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO recordings
                    (dedup_key, source_name, transcript, title, summary,
                     action_items, key_decisions, open_questions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dedup_key) DO UPDATE SET
                    source_name=excluded.source_name,
                    transcript=excluded.transcript,
                    title=excluded.title,
                    summary=excluded.summary,
                    action_items=excluded.action_items,
                    key_decisions=excluded.key_decisions,
                    open_questions=excluded.open_questions
                """,
                (
                    record.dedup_key,
                    record.source_name,
                    record.transcript,
                    record.title,
                    record.summary,
                    json.dumps(record.action_items),
                    json.dumps(record.key_decisions),
                    json.dumps(record.open_questions),
                    record.created_at or datetime.now(timezone.utc).isoformat(),
                ),
            )

    def delete(self, dedup_key: str) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute("DELETE FROM recordings WHERE dedup_key = ?", (dedup_key,))