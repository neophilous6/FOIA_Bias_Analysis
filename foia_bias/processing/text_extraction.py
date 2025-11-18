"""PDF to text utilities."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import pdfplumber

from foia_bias.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def extract_text_from_pdf(path: str | Path, min_len_for_no_ocr: int = 1000) -> str:
    """Extract text from a PDF, falling back to OCR when necessary."""

    path = Path(path)
    text_parts: list[str] = []

    # Step 1: try to use the embedded text layer, which is much faster and
    # preserves layout well for digital PDFs.
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")

    text = "\n".join(text_parts).strip()
    if len(text) >= min_len_for_no_ocr:
        return text

    # Step 2: if the PDF is mostly scanned images, fall back to running
    # Tesseract. We keep the log at DEBUG so the normal run output stays clean.
    LOGGER.debug("Running OCR fallback for %s", path)
    ocr_text = subprocess.check_output([
        "tesseract",
        str(path),
        "stdout",
        "--psm",
        "1",
    ], text=True)
    return ocr_text.strip()
