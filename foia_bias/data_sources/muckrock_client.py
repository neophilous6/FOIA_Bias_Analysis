"""MuckRock ingestion client."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import requests
from muckrock import MuckRockClient

from foia_bias.data_sources.base import BaseIngestor, DocumentRecord
from foia_bias.utils.logging_utils import get_logger


class MuckRockIngestor(BaseIngestor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        token_env = config.get("api_token_env_var", "MUCKROCK_API_TOKEN")
        token = os.getenv(token_env)
        if not token:
            raise RuntimeError(f"Missing MuckRock API token in env var {token_env}")
        self.client = MuckRockClient(token)
        self.download_dir = self.ensure_dir(config.get("download_dir", "data/muckrock/raw"))
        self.max_requests = config.get("max_requests", 1000)
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")

    def fetch(self) -> Iterator[DocumentRecord]:
        params: Dict[str, Any] = {
            "status": "done",
            "has_files": True,
            "page_size": 100,
        }
        if self.start_date:
            params["updated_after"] = self.start_date
        count = 0
        for req in self.client.requests.search(**params):
            if self.end_date and req.get("date_done") and req["date_done"] > self.end_date:
                continue
            files = req.get("files", [])
            if not files:
                continue
            yield DocumentRecord(
                source="muckrock",
                request_id=str(req["id"]),
                agency=req.get("agency_name"),
                title=req.get("title"),
                description=req.get("short_description"),
                date_submitted=req.get("date_submitted"),
                date_done=req.get("date_done"),
                requester=req.get("user_name"),
                files=files,
            )
            count += 1
            if count >= self.max_requests:
                break

    def download_files_for_record(self, record: DocumentRecord) -> list[Path]:
        paths: list[Path] = []
        for f in record.files:
            url = f.get("url")
            if not url:
                continue
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            filename = f"{record.request_id}_{f.get('id', 'file')}.pdf"
            path = self.download_dir / filename
            path.write_bytes(resp.content)
            paths.append(path)
        return paths
