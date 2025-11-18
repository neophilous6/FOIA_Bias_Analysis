"""High level orchestration for ingestion + labeling."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from foia_bias.analysis.aggregate import prepare_for_analysis
from foia_bias.analysis.models import run_favorability_model, run_wrongdoing_model
from foia_bias.data_sources.base import DocumentRecord
from foia_bias.data_sources.foia_gov_client import FOIAGovClient
from foia_bias.data_sources.logs_downloader import FOIALogsDownloader
from foia_bias.data_sources.muckrock_client import MuckRockIngestor
from foia_bias.data_sources.reading_rooms import ReadingRoomScraper
from foia_bias.llm.classifiers import classify_document
from foia_bias.processing.admin_mapping import get_admin_for_date
from foia_bias.processing.politics_filter import PARTY_KEYWORDS, is_potentially_partisan
from foia_bias.processing.text_extraction import extract_text_from_pdf
from foia_bias.utils.logging_utils import configure_logging, get_logger


class Pipeline:
    """Coordinate ingestion, labeling, and analysis workstreams."""

    def __init__(self, config: Dict[str, Any]):
        # Persist the loaded configuration so every sub-component sees the
        # same knobs (sources, LLM settings, etc.).
        self.config = config
        configure_logging(config)
        self.logger = get_logger("Pipeline")

        # All labeled Parquet files are written under this directory. Keeping
        # the path on the instance makes it easy to reuse across steps.
        self.storage_dir = Path(config.get("storage", {}).get("labeled_output_dir", "data/processed"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- ingestion runners -----------------------------
    def run_all(self) -> None:
        """Execute every enabled ingestion source in the configured order."""
        for source in self.config.get("sources", {}).get("processing_priority", []):
            handler = getattr(self, f"process_{source}", None)
            if handler is None:
                self.logger.warning("No handler for source %s", source)
                continue
            enabled = self.config.get("sources", {}).get(source, {}).get("enabled", True)
            if not enabled:
                self.logger.info("Skipping disabled source %s", source)
                continue
            handler()

    def process_muckrock(self) -> None:
        """Download, extract, and label MuckRock responses."""
        source_cfg = self.config["sources"]["muckrock"]
        ingestor = MuckRockIngestor(source_cfg)
        self.logger.info("Starting MuckRock ingestion (max %s requests)", ingestor.max_requests)
        records = []
        for idx, record in enumerate(tqdm(ingestor.fetch(), desc="MuckRock requests"), start=1):
            self.logger.info(
                "Processing MuckRock request %s (%s) [%d]", record.request_id, record.title, idx
            )
            paths = ingestor.download_files_for_record(record)
            if not paths:
                self.logger.info("Request %s did not yield any downloadable files", record.request_id)
                continue
            text = self.combine_texts(paths)
            labeled = self.label_text(text, record)
            if labeled:
                records.append(labeled)
                self.logger.info("Finished labeling request %s", record.request_id)
            else:
                self.logger.info("Skipping request %s because no text was extracted", record.request_id)
        self.logger.info("Completed MuckRock ingestion with %s labeled records", len(records))
        self.save_records(records, "muckrock")

    def process_agency_logs(self) -> None:
        """Iterate through normalized agency logs row-by-row for labeling."""
        source_cfg = self.config["sources"]["agency_logs"]
        ingestor = FOIALogsDownloader(source_cfg)
        self.logger.info("Starting agency log ingestion")
        records = []
        for record in tqdm(list(ingestor.fetch()), desc="Agency logs"):
            parquet_path = Path(record.files[0]["path"])
            if not parquet_path.exists():
                self.logger.warning("Agency log parquet missing at %s", parquet_path)
                continue
            df = pd.read_parquet(parquet_path)
            for idx, row in df.iterrows():
                self.logger.info(
                    "Labeling agency log %s row %s (%s entries)",
                    record.request_id,
                    idx,
                    len(row.index),
                )
                text = self.render_log_row_text(row)
                if not text.strip():
                    self.logger.info("Skipping empty log row %s for %s", idx, record.request_id)
                    continue
                row_record = DocumentRecord(
                    source=record.source,
                    request_id=f"{record.request_id}_{idx}",
                    agency=record.agency,
                    title=self.infer_log_row_title(row, record.title, idx),
                    description=None,
                    date_submitted=None,
                    date_done=self.infer_log_row_date(row),
                    requester=None,
                    files=record.files,
                )
                labeled = self.label_text(text, row_record)
                if labeled:
                    records.append(labeled)
        self.logger.info("Completed agency logs ingestion with %s labeled rows", len(records))
        self.save_records(records, "agency_logs")

    def process_reading_rooms(self) -> None:
        """Scrape PDFs from agency reading rooms and label their contents."""
        source_cfg = self.config["sources"]["reading_rooms"]
        ingestor = ReadingRoomScraper(source_cfg)
        self.logger.info("Starting reading-room ingestion")
        records = []
        for record in tqdm(list(ingestor.fetch()), desc="Reading rooms"):
            text = self.combine_texts([Path(record.files[0]["path"])])
            labeled = self.label_text(text, record)
            if labeled:
                records.append(labeled)
        self.logger.info("Completed reading-room ingestion with %s labeled records", len(records))
        self.save_records(records, "reading_rooms")

    def process_foia_gov_annual(self) -> None:
        """Load FOIA.gov annual datasets and treat them as metadata only."""
        source_cfg = self.config["sources"]["foia_gov_annual"]
        ingestor = FOIAGovClient(source_cfg)
        self.logger.info("Starting FOIA.gov annual ingestion")
        records = []
        for record in tqdm(list(ingestor.fetch()), desc="FOIA.gov annual"):
            text = Path(record.files[0]["path"]).read_text(encoding="utf-8")
            labeled = self.label_text(text, record, treat_as_metadata=True)
            if labeled:
                records.append(labeled)
        self.logger.info("Completed FOIA.gov ingestion with %s labeled records", len(records))
        self.save_records(records, "foia_gov")

    # ----------------------------- labeling helpers -----------------------------
    def combine_texts(self, paths: Iterable[Path]) -> str:
        """OCR + concatenate multiple PDF files into a single blob."""
        text_cfg = self.config.get("processing", {}).get("text_extraction", {})
        min_len = text_cfg.get("min_text_length_for_no_ocr", 1000)
        # Each PDF is independently extracted (with OCR fallback) so we can
        # concatenate them into a single prompt string per request.
        parts = [extract_text_from_pdf(p, min_len_for_no_ocr=min_len) for p in paths if Path(p).exists()]
        if not parts:
            return ""
        return "\n\n".join(parts)

    def label_text(self, text: str, record, treat_as_metadata: bool = False) -> Optional[dict]:
        """Apply the pre-filter + classifier to a single document record."""
        if not text:
            self.logger.info("No text extracted for record %s; skipping", record.request_id)
            return None
        if treat_as_metadata:
            # Metadata-only sources (e.g., FOIA.gov stats) have no free-form
            # documents. We short-circuit to a neutral label so they can still
            # flow through the downstream analytics code.
            classification = self.default_non_political_label("Metadata source; classifier skipped.")
        else:
            if not self.should_run_classifier(text):
                # The cheap keyword / NER filter determined the content is not
                # obviously partisan, so we avoid the expensive LLM call.
                classification = self.default_non_political_label(
                    "Pre-filter classified document as non-political; full classifier not called."
                )
                self.logger.info(
                    "Record %s skipped by prefilter; labeling as non-political", record.request_id
                )
            else:
                # This is the main path: run the structured classification
                # prompt and capture its JSON response.
                self.logger.info("Invoking classifier for record %s", record.request_id)
                classification = classify_document(text, record.request_id, self.config)
        admin_info = get_admin_for_date(record.date_done, self.config.get("processing", {}).get("admin_mapping", {}).get("mark_transition_period_months", 0))
        return {
            "source": record.source,
            "request_id": record.request_id,
            "agency": record.agency,
            "title": record.title,
            "date_done": record.date_done,
            "admin_name": admin_info.get("admin_name"),
            "admin_party": admin_info.get("admin_party"),
            "is_transition": admin_info.get("is_transition"),
            "political_relevance": classification["political_relevance"],
            "targets": classification["main_partisan_targets"],
            "wrongdoing_D": classification["wrongdoing_assessment"]["wrongdoing_by_party"]["D"],
            "wrongdoing_R": classification["wrongdoing_assessment"]["wrongdoing_by_party"]["R"],
            "fav_score_D": classification["favorability_assessment"]["favorability_scores"]["D"],
            "fav_score_R": classification["favorability_assessment"]["favorability_scores"]["R"],
            "raw_classification": json.dumps(classification),
        }

    def default_non_political_label(self, note: str) -> dict:
        """Return a schema-compatible stub used for skipped documents."""
        return {
            "political_relevance": "none",
            "main_partisan_targets": [],
            "wrongdoing_assessment": {
                "overall_wrongdoing_probability": 0.0,
                "wrongdoing_by_party": {"D": 0.0, "R": 0.0},
            },
            "favorability_assessment": {
                "overall_valence_party": {"D": "none", "R": "none"},
                "favorability_scores": {"D": 0.0, "R": 0.0},
            },
            "notes": note,
        }

    def should_run_classifier(self, text: str) -> bool:
        """Determine whether the expensive LLM call is warranted."""
        pre_cfg = self.config.get("prefilter", {})
        keyword_threshold = pre_cfg.get("keyword_threshold", 1)
        try:
            if is_potentially_partisan(text, keyword_threshold=keyword_threshold):
                return True
        except RuntimeError as exc:
            self.logger.warning("NER filter unavailable (%s); falling back to keywords only.", exc)
            if keyword_threshold <= 0:
                return True
            lowered = text.lower()
            return any(kw in lowered for kw in PARTY_KEYWORDS)
        if pre_cfg.get("use_embedding_filter"):
            self.logger.warning("Embedding prefilter enabled but no trained classifier is provided. Skipping.")
        return False

    def render_log_row_text(self, row: pd.Series) -> str:
        """Convert a heterogeneous log row into a classifier-friendly string."""
        parts: List[str] = []
        for column, value in row.items():
            if pd.isna(value):
                continue
            value_str = str(value).strip()
            if not value_str:
                continue
            parts.append(f"{column}: {value_str}")
        rendered = "\n".join(parts)
        self.logger.debug("Rendered log row with %d columns", len(parts))
        return rendered

    def infer_log_row_date(self, row: pd.Series) -> Optional[str]:
        """Best-effort extraction of a decision date from agency logs."""
        date_columns = [
            "date",
            "closed",
            "completed",
            "response",
            "decision",
            "released",
        ]
        for column, value in row.items():
            if not isinstance(column, str):
                continue
            lowered = column.lower()
            if not any(token in lowered for token in date_columns):
                continue
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                continue
            return parsed.date().isoformat()
        return None

    def infer_log_row_title(self, row: pd.Series, fallback: Optional[str], idx: int) -> Optional[str]:
        """Pick a human-friendly title per row to aid downstream analysis."""
        title_hints = ["subject", "summary", "title", "description", "records", "topic"]
        for column, value in row.items():
            if not isinstance(column, str):
                continue
            lowered = column.lower()
            if not any(hint in lowered for hint in title_hints):
                continue
            value_str = str(value).strip()
            if value_str:
                return value_str
        if fallback:
            return f"{fallback} (row {idx})"
        return f"Agency log row {idx}"

    def save_records(self, records: List[dict], source: str) -> None:
        """Write labeled data to Parquet so later analysis can reload it."""
        if not records:
            self.logger.info("No records to save for %s", source)
            return
        df = pd.DataFrame(records)
        # Serialize the nested list so Parquet stays schema-stable. When
        # reloading we JSON-decode this column.
        df["targets"] = df["targets"].apply(json.dumps)
        path = self.storage_dir / self.config.get("storage", {}).get("labeled_file_pattern", "labeled_{source}.parquet").format(source=source)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self.logger.info("Saved %s records to %s", len(df), path)

    # ----------------------------- analysis entrypoints -----------------------------
    def load_labeled_data(self, source: Optional[str] = None) -> pd.DataFrame:
        """Load one or more labeled Parquet files and JSON-decode columns."""
        files = list(self.storage_dir.glob("labeled_*.parquet")) if source is None else [
            self.storage_dir / self.config.get("storage", {}).get("labeled_file_pattern", "labeled_{source}.parquet").format(source=source)
        ]
        frames = []
        for path in files:
            if path.exists():
                df = pd.read_parquet(path)
                # Undo the serialization that happened inside ``save_records``.
                df["targets"] = df["targets"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                df["raw_classification"] = df["raw_classification"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                frames.append(df)
        if not frames:
            raise FileNotFoundError("No labeled parquet files found")
        return pd.concat(frames, ignore_index=True)

    def analyze_wrongdoing(self, source: Optional[str] = None):
        """Run the wrongdoing hypothesis regression and return statsmodels text."""
        df = self.load_labeled_data(source)
        df = prepare_for_analysis(df, self.config)
        model = run_wrongdoing_model(
            df,
            include_agency_fe=self.config["analysis"]["regression"].get("include_agency_fixed_effects", True),
            include_year_fe=self.config["analysis"]["regression"].get("include_year_fixed_effects", True),
        )
        return model.summary().as_text()

    def analyze_favorability(self, source: Optional[str] = None):
        """Run the favorability regression and return statsmodels text."""
        df = self.load_labeled_data(source)
        df = prepare_for_analysis(df, self.config)
        model = run_favorability_model(
            df,
            include_agency_fe=self.config["analysis"]["regression"].get("include_agency_fixed_effects", True),
            include_year_fe=self.config["analysis"]["regression"].get("include_year_fixed_effects", True),
        )
        return model.summary().as_text()
