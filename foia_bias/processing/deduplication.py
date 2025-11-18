"""Simple text dedup helpers."""
from __future__ import annotations

import hashlib
from typing import Iterable, Set


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_records(records: Iterable[dict], field: str = "text") -> list[dict]:
    seen: Set[str] = set()
    unique = []
    for record in records:
        value = record.get(field, "")
        if not value:
            unique.append(record)
            continue
        h = text_hash(value)
        if h in seen:
            continue
        seen.add(h)
        unique.append(record)
    return unique
