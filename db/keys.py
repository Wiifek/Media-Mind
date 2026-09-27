"""
One job: turn a source (audio file, video, meeting recording — whatever
it is) into a stable "dedup key" so the same content is never
transcribed or summarized twice, even if it gets renamed or
re-uploaded under a different filename.

Kept separate from the database code because it's a pure function
(no I/O to a DB, easy to unit test on its own) and conceptually
distinct: this is a hashing *strategy*, not storage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


def make_key(source: str | Path, source_name: Optional[str] = None) -> str:
    """
    - Local file that exists on disk -> hash its bytes. Robust to
      renames; two different files that happen to share a name get
      different keys.
    - URL (e.g. a YouTube link) or a path that doesn't exist locally
      -> hash the string itself, with source_name as a tiebreaker.
    """
    path = Path(source) if not str(source).startswith("http") else None

    if path is not None and path.exists():
        return _hash_file(path)

    raw = f"{source}::{source_name or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()