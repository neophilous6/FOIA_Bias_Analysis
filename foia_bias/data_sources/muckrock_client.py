"""MuckRock ingestion client."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import requests
from requests import HTTPError

from foia_bias.data_sources.base import BaseIngestor, DocumentRecord
from foia_bias.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SimpleMuckRockClient:
    """Thin wrapper over the public MuckRock REST API."""

    def __init__(
        self,
        token: str | None,
        base_url: str = "https://www.muckrock.com/api_v2",
        documents_base_url: str | None = None,
        rate_limit_seconds: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.documents_base_url = (
            documents_base_url.rstrip("/") if documents_base_url else self.base_url
        )
        self.session = requests.Session()
        self.rate_limit_seconds = rate_limit_seconds
        if token:
            # If a token is present, MuckRock lets us send requests at full
            # speed. Otherwise the caller should set a rate limit.
            self.session.headers["Authorization"] = f"Token {token}"

    def _paged_get(self, url: str, params: Dict[str, Any] | None = None) -> Iterator[Dict[str, Any]]:
        """Shared pagination helper for /requests/ and /documents/ endpoints."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        page_idx = 1
        while url:
            if self.rate_limit_seconds > 0:
                # Respect the configured rate limit to avoid 429s for
                # unauthenticated scrapes.
                time.sleep(self.rate_limit_seconds)
            logger.info("Requesting %s page %s with params=%s", url, page_idx, params or "{}")
            resp = self.session.get(url, params=params if "?" not in url else None, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            logger.info("Received %s results from %s page %s", len(results), url, page_idx)
            for row in results:
                yield row
            url = payload.get("next")
            params = None  # after first request, pagination URLs include params
            page_idx += 1

    def iter_requests(self, **params: Any) -> Iterator[Dict[str, Any]]:
        """Stream paginated request objects from the REST API."""
        yield from self._paged_get(f"{self.base_url}/requests/", params)

    def iter_documents(self, request_id: str) -> Iterator[Dict[str, Any]]:
        """Iterate through released documents with multi-endpoint fallbacks."""

        attempts: list[tuple[str, Dict[str, Any] | None]] = []
        seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()

        def _push(url: str | None, params: Dict[str, Any] | None = None) -> None:
            if not url:
                return
            key = (url, tuple(sorted((params or {}).items())))
            if key in seen:
                return
            seen.add(key)
            attempts.append((url, params))

        # Try the dedicated documents API (v1) first, followed by the v2 path.
        _push(f"{self.documents_base_url}/requests/{request_id}/documents/")
        if self.documents_base_url != self.base_url:
            _push(f"{self.base_url}/requests/{request_id}/documents/")

        # Fallback to the search endpoint that accepts ?request=<id> filters.
        _push(f"{self.documents_base_url}/documents/", {"request": request_id})
        if self.documents_base_url != self.base_url:
            _push(f"{self.base_url}/documents/", {"request": request_id})

        for url, params in attempts:
            try:
                docs = list(self._paged_get(url, params))
            except HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status in {401, 403, 404}:
                    logger.info(
                        "Documents endpoint %s returned %s for request %s; trying fallback",
                        url,
                        status,
                        request_id,
                    )
                    continue
                raise
            if not docs:
                logger.info(
                    "Documents endpoint %s returned zero files for request %s; trying fallback",
                    url,
                    request_id,
                )
                continue
            logger.info(
                "Retrieved %s documents for request %s via %s",
                len(docs),
                request_id,
                url,
            )
            for doc in docs:
                yield doc
            return
        logger.warning("Unable to retrieve documents for request %s", request_id)

    def get_request(self, request_id: str) -> Dict[str, Any]:
        """Fetch the detail view for a request, including embedded documents."""
        if self.rate_limit_seconds > 0:
            time.sleep(self.rate_limit_seconds)
        url = f"{self.base_url}/requests/{request_id}/"
        logger.info("Requesting detail for request %s", request_id)
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()


class MuckRockIngestor(BaseIngestor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        token_env = config.get("api_token_env_var", "MUCKROCK_API_TOKEN")
        token = os.getenv(token_env)
        self.download_dir = self.ensure_dir(config.get("download_dir", "data/muckrock/raw"))
        self.max_requests = config.get("max_requests", 1000)
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        base_url = config.get("base_url", "https://www.muckrock.com/api_v2")
        documents_base_url = config.get("documents_base_url", "https://www.muckrock.com/api_v1")
        default_rate = 0.0 if token else 1.0
        self.rate_limit_seconds = float(config.get("rate_limit_seconds", default_rate))

        if token:
            logger.info("Initialized authenticated MuckRock client with API token")
        else:
            logger.warning(
                "Running MuckRock client without API token; defaulting to %.2fs between requests",
                self.rate_limit_seconds,
            )

        self.client = SimpleMuckRockClient(
            token=token,
            base_url=base_url,
            documents_base_url=documents_base_url,
            rate_limit_seconds=self.rate_limit_seconds,
        )

    def fetch(self) -> Iterator[DocumentRecord]:
        """Yield completed requests that already have releasable files."""
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
            documents = self._extract_documents(req)
            if not documents:
                logger.info("Request %s has no released documents; skipping", req.get("id"))
                continue
            logger.info(
                "Fetched request %s (%s) with %d file(s) from %s",
                req.get("id"),
                req.get("title"),
                len(documents),
                req.get("agency_name"),
            )
            yield DocumentRecord(
                source="muckrock",
                request_id=str(req["id"]),
                agency=req.get("agency_name"),
                title=req.get("title"),
                description=req.get("short_description"),
                date_submitted=req.get("date_submitted"),
                date_done=req.get("date_done"),
                requester=req.get("user_name"),
                files=documents,
            )
            count += 1
            if count >= self.max_requests:
                break

    def _iter_requests(self, params: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        yield from self.client.iter_requests(**params)

    def download_files_for_record(self, record: DocumentRecord) -> list[Path]:
        """Download every PDF referenced in the record and persist it locally."""
        paths: list[Path] = []
        for idx, f in enumerate(record.files, start=1):
            url = (
                f.get("url")
                or f.get("document_url")
                or f.get("file", {}).get("url")
                or f.get("document", {}).get("url")
            )
            if not url:
                logger.warning(
                    "Skipping file %s for request %s because no URL was present", f.get("id"), record.request_id
                )
                continue
            logger.info(
                "Downloading file %d/%d for request %s from %s",
                idx,
                len(record.files),
                record.request_id,
                url,
            )
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            suffix = self._infer_suffix(url, f)
            filename = f.get("filename") or f"{record.request_id}_{f.get('id', idx)}{suffix}"
            path = self.download_dir / filename
            path.write_bytes(resp.content)
            logger.info(
                "Stored %s bytes for request %s file %s at %s",
                len(resp.content),
                record.request_id,
                f.get("id", "file"),
                path,
            )
            paths.append(path)
        return paths

    def _extract_documents(self, request_row: Dict[str, Any]) -> list[Dict[str, Any]]:
        """Ensure we always return the list of released documents for a request."""
        documents = request_row.get("documents") or request_row.get("files") or []
        if documents:
            return documents
        request_id = request_row.get("id")
        if not request_id:
            return []
        logger.info("Request %s missing embedded documents; fetching detail endpoint", request_id)
        try:
            detail = self.client.get_request(str(request_id))
        except HTTPError as exc:
            logger.warning(
                "Failed to fetch request detail for %s (%s); falling back to documents API",
                request_id,
                exc,
            )
            return list(self.client.iter_documents(str(request_id)))

        documents = (
            detail.get("documents")
            or detail.get("files")
            or detail.get("attachments")
            or []
        )
        if documents:
            return documents
        logger.info("Detail endpoint lacked documents for %s; using documents API fallback", request_id)
        return list(self.client.iter_documents(str(request_id)))

    @staticmethod
    def _infer_suffix(url: str, payload: Dict[str, Any]) -> str:
        """Pick a reasonable file extension for the downloaded artifact."""
        parsed = Path(url.split("?")[0])
        if parsed.suffix:
            return parsed.suffix
        filetype = payload.get("filetype") or payload.get("content_type")
        if filetype:
            mapping = {
                "pdf": ".pdf",
                "application/pdf": ".pdf",
                "text/plain": ".txt",
                "html": ".html",
            }
            return mapping.get(filetype.lower(), ".bin")
        return ".bin"
