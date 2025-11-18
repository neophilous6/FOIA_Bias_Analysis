"""Common interfaces for ingestion sources."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from foia_bias.utils.logging_utils import get_logger


@dataclass
class DocumentRecord:
    """Normalized representation passed between ingestion + labeling layers."""
    source: str
    request_id: str
    agency: Optional[str]
    title: Optional[str]
    description: Optional[str]
    date_submitted: Optional[str]
    date_done: Optional[str]
    requester: Optional[str]
    files: List[Dict[str, str]]


class BaseIngestor:
    """All ingestion clients inherit from this class."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    def fetch(self) -> Iterator[DocumentRecord]:
        raise NotImplementedError

    def ensure_dir(self, path: str | Path) -> Path:
        """Utility helper so subclasses get consistent directory creation."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
