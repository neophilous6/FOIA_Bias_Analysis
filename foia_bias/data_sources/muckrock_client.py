"""MuckRock ingestion client."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import requests

from foia_bias.data_sources.base import BaseIngestor, DocumentRecord
from foia_bias.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from muckrock.client import MuckRock as SDKMuckRock  # type: ignore
except Exception:  # pragma: no cover - library optional
    SDKMuckRock = None


class SimpleMuckRockClient:
    """Thin wrapper over the public MuckRock REST API using token auth."""

    def __init__(self, token: Optional[str], base_url: str = "https://www.muckrock.com/api_v2"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Token {token}"

    def iter_requests(self, **params: Any) -> Iterator[Dict[str, Any]]:
        url = f"{self.base_url}/requests/"
        params = {k: v for k, v in params.items() if v is not None}
        while url:
            resp = self.session.get(url, params=params if "?" not in url else None, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            for row in payload.get("results", []):
                yield row
            url = payload.get("next")
            params = None  # after first request, pagination URLs include params


class MuckRockIngestor(BaseIngestor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        token_env = config.get("api_token_env_var", "MUCKROCK_API_TOKEN")
        username_env = config.get("username_env_var", "MUCKROCK_USERNAME")
        password_env = config.get("password_env_var", "MUCKROCK_PASSWORD")
        token = os.getenv(token_env)
        username = os.getenv(username_env)
        password = os.getenv(password_env)
        self.download_dir = self.ensure_dir(config.get("download_dir", "data/muckrock/raw"))
        self.max_requests = config.get("max_requests", 1000)
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        base_url = config.get("base_url", "https://www.muckrock.com/api_v2")

        if username and password and SDKMuckRock is not None:
            self.client_mode = "sdk"
            self.client = SDKMuckRock(username=username, password=password, base_uri=base_url + "/")
            logger.info("Initialized python-muckrock SDK client for authenticated ingestion")
        else:
            if not token:
                raise RuntimeError(
                    "Missing MuckRock credentials: set MUCKROCK_USERNAME/MUCKROCK_PASSWORD or "
                    f"provide a token via {token_env}"
                )
            self.client_mode = "token"
            self.client = SimpleMuckRockClient(token=token, base_url=base_url)
            logger.info("Initialized token-based HTTP client for MuckRock ingestion")

    def fetch(self) -> Iterator[DocumentRecord]:
        params: Dict[str, Any] = {
            "status": "done",
            "has_files": True,
            "page_size": 100,
        }
        if self.start_date:
            params["updated_after"] = self.start_date
        count = 0
        for req in self._iter_requests(params):
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

    def _iter_requests(self, params: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        if self.client_mode == "sdk":
            results = self.client.requests.list(**params)
            for req in results:
                yield self._request_to_dict(req)
        else:
            yield from self.client.iter_requests(**params)

    @staticmethod
    def _request_to_dict(req: Any) -> Dict[str, Any]:
        if isinstance(req, dict):
            return req
        data = {k: v for k, v in vars(req).items() if not k.startswith("_")}
        return data

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
