"""Download FOIA.gov annual report data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator

import requests

from foia_bias.data_sources.base import BaseIngestor, DocumentRecord


class FOIAGovClient(BaseIngestor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.output_dir = self.ensure_dir(config.get("output_dir", "data/foia_gov"))

    def fetch_year(self, base_url: str, year: int) -> Path:
        url = f"{base_url}?year={year}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        path = self.output_dir / f"foia_gov_{year}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def fetch(self) -> Iterator[DocumentRecord]:
        base_url = self.config.get("base_url")
        years = self.config.get("years", [])
        if not base_url:
            raise ValueError("Missing FOIA.gov base_url in config")
        for year in years:
            path = self.fetch_year(base_url, year)
            yield DocumentRecord(
                source="foia_gov",
                request_id=str(year),
                agency="FOIA.gov",
                title=f"FOIA.gov annual data {year}",
                description=base_url,
                date_submitted=None,
                date_done=None,
                requester=None,
                files=[{"path": str(path)}],
            )
