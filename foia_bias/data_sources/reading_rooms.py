"""Simple HTML paginated reading-room scraper."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

import requests
from bs4 import BeautifulSoup

from foia_bias.data_sources.base import BaseIngestor, DocumentRecord


class ReadingRoomScraper(BaseIngestor):
    """Collect PDFs listed in agency reading rooms with basic HTML parsing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.download_dir = self.ensure_dir(config.get("download_dir", "data/reading_rooms/raw"))

    def fetch_endpoint(self, endpoint: Dict[str, Any]) -> Iterator[DocumentRecord]:
        """Walk every paginated HTML page and emit PDFs discovered there."""
        base_url = endpoint["base_url"]
        max_pages = endpoint.get("max_pages", 1)
        param = endpoint.get("pagination_param", "page")
        for page in range(1, max_pages + 1):
            params = {param: page}
            resp = requests.get(base_url, params=params, timeout=120)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.select("a")
            for link in links:
                href = link.get("href")
                if not href or not href.lower().endswith(".pdf"):
                    continue
                title = link.text.strip() or "reading-room-doc"
                path = self.download_pdf(href, title, base_url)
                yield DocumentRecord(
                    source="reading_room",
                    request_id=f"{endpoint['id']}-{page}-{path.stem}",
                    agency=endpoint.get("name"),
                    title=title,
                    description=href,
                    date_submitted=None,
                    date_done=None,
                    requester=None,
                    files=[{"path": str(path)}],
                )

    def download_pdf(self, url: str, title: str, base_url: str) -> Path:
        """Fetch a PDF to the configured cache, handling relative links."""
        if not url.lower().startswith("http"):
            url = requests.compat.urljoin(base_url, url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        filename = title.replace(" ", "_")[:100] + ".pdf"
        path = self.download_dir / filename
        path.write_bytes(resp.content)
        return path

    def fetch(self) -> Iterator[DocumentRecord]:
        for endpoint in self.config.get("endpoints", []):
            if not endpoint.get("enabled", True):
                continue
            # Each endpoint may have drastically different markup, so we keep
            # the scraping logic intentionally simple and paginated.
            yield from self.fetch_endpoint(endpoint)
