"""
The shape of a stored recording (audio file, video, or meeting — the
schema doesn't care which). One object, used consistently on both the
write path (build a RecordingRecord, pass it to db.save()) and the
read path (db.get() hands you back a RecordingRecord, not a raw dict) —
so there's only ever one way this data looks, instead of loose
positional arguments on the way in and a dict on the way out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RecordingRecord:
    dedup_key: str
    source_name: str           # filename or URL label — audio, video, or meeting
    transcript: str
    title: str
    summary: str
    action_items: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    created_at: Optional[str] = None   # set automatically on save if left blank
    id: Optional[int] = None           # set by the database, ignore when creating

    @classmethod
    def from_row(cls, row: dict) -> "RecordingRecord":
        """Build a RecordingRecord from a raw sqlite3.Row (as a dict)."""
        return cls(
            id=row["id"],
            dedup_key=row["dedup_key"],
            source_name=row["source_name"],
            transcript=row["transcript"],
            title=row["title"],
            summary=row["summary"],
            action_items=json.loads(row["action_items"] or "[]"),
            key_decisions=json.loads(row["key_decisions"] or "[]"),
            open_questions=json.loads(row["open_questions"] or "[]"),
            created_at=row["created_at"],
        )