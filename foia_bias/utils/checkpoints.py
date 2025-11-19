"""Lightweight helpers for persisting ingestion checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Return the parsed checkpoint or an empty dict if unavailable."""

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        # If the checkpoint cannot be parsed (e.g., interrupted write), start
        # from scratch instead of crashing the pipeline run.
        return {}


def save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    """Persist the checkpoint atomically to avoid partial writes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data))
    tmp_path.replace(path)
