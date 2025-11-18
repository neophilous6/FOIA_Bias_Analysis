"""Keyword + NER prefilter for partisan content."""
from __future__ import annotations

from functools import lru_cache
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
import spacy

LOGGER = logging.getLogger(__name__)

PARTY_KEYWORDS = [
    "democrat",
    "democrats",
    "democratic party",
    "republican",
    "republicans",
    "republican party",
    "gop",
    "dnc",
    "rnc",
    "campaign",
    "election",
    "senator",
    "congressional",
    "president",
]

# URLs and cache locations for the GovTrack people index, which contains
# every Senator/Representative in U.S. history.  The dataset is updated
# periodically, so we cache the downloaded file locally and refresh on
# demand when the user deletes the cache.
GOVTRACK_INDEX_URL = os.environ.get(
    "GOVTRACK_PEOPLE_INDEX_URL", "https://www.govtrack.us/data/us/people/index.json"
)
GOVTRACK_CACHE_PATH = Path(
    os.environ.get("GOVTRACK_PEOPLE_CACHE", "data/cache/govtrack_people_index.json")
)
CONGRESS_MIN_YEAR = 1993
CONGRESS_MAX_YEAR = 2025

# A couple of non-person entities we still want to match quickly.
STATIC_ACTORS = {
    "dnc": "D",
    "democratic national committee": "D",
    "rnc": "R",
    "republican national committee": "R",
}


def _ensure_cache_dir() -> None:
    GOVTRACK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _download_govtrack_people() -> Dict:
    """Download the GovTrack index and persist it locally."""

    LOGGER.info("Downloading GovTrack people index from %s", GOVTRACK_INDEX_URL)
    resp = requests.get(GOVTRACK_INDEX_URL, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    try:
        _ensure_cache_dir()
        GOVTRACK_CACHE_PATH.write_text(json.dumps(data))
    except OSError as exc:  # pragma: no cover - IO heavy
        LOGGER.warning("Unable to cache GovTrack index at %s: %s", GOVTRACK_CACHE_PATH, exc)
    return data


def _load_people_index() -> Dict:
    """Load the GovTrack index from cache or download it on demand."""

    if GOVTRACK_CACHE_PATH.exists():
        try:
            return json.loads(GOVTRACK_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            LOGGER.warning("Cached GovTrack index is corrupt; re-downloading")
        except OSError as exc:
            LOGGER.warning("Unable to read GovTrack cache %s: %s", GOVTRACK_CACHE_PATH, exc)
    try:
        return _download_govtrack_people()
    except requests.RequestException as exc:
        LOGGER.warning(
            "Failed to download GovTrack data (%s). Political entity detection will be limited.",
            exc,
        )
        return {}


def _safe_year(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value[:4])
    except (ValueError, TypeError):
        return None


def _normalize_party(value: str | None) -> str | None:
    if not value:
        return None
    first = value.strip().upper()[:1]
    if first in {"D", "R"}:
        return first
    return None


SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalize_tokens(name: str) -> List[str]:
    cleaned = re.sub(r"[^A-Za-z\s-]", " ", name)
    cleaned = cleaned.replace("-", " ")
    tokens = [tok for tok in cleaned.lower().split() if tok]
    return [tok for tok in tokens if tok not in SUFFIXES]


def _name_variants(name: str) -> List[str]:
    tokens = _normalize_tokens(name)
    if not tokens:
        return []
    variants = {" ".join(tokens)}
    if len(tokens) >= 2:
        first = tokens[0]
        last = tokens[-1]
        variants.add(f"{first} {last}")
        # Include nicknames that appear as quoted first tokens (e.g., "Mike")
    return sorted(variants)


def _collect_name_strings(person: Dict) -> List[str]:
    """Gather every plausible string representation of a member's name."""

    names: List[str] = []
    direct_fields = [
        "name",
        "name_full",
        "name_long",
        "name_sort",
        "official_full",
        "display_name",
        "fullname",
    ]
    for field in direct_fields:
        value = person.get(field)
        if isinstance(value, str):
            names.append(value)

    name_block = person.get("name")
    if isinstance(name_block, dict):
        for maybe in name_block.values():
            if isinstance(maybe, str):
                names.append(maybe)
        first = name_block.get("first") or name_block.get("given")
        last = name_block.get("last") or name_block.get("family")
        if isinstance(first, str) and isinstance(last, str):
            names.append(f"{first} {last}")
        nickname = name_block.get("nickname")
        if isinstance(nickname, str) and isinstance(last, str):
            names.append(f"{nickname} {last}")

    # Legacy GovTrack data exposes firstname/lastname fields at the top level.
    first = person.get("firstname") or person.get("first_name") or person.get("first")
    last = person.get("lastname") or person.get("last_name") or person.get("last")
    nickname = person.get("nickname")
    if isinstance(first, str) and isinstance(last, str):
        names.append(f"{first} {last}")
    if isinstance(nickname, str) and isinstance(last, str):
        names.append(f"{nickname} {last}")

    other_names = person.get("other_names") or person.get("othernames")
    if isinstance(other_names, list):
        for entry in other_names:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict):
                val = entry.get("name") or entry.get("official_full")
                if isinstance(val, str):
                    names.append(val)
                first = entry.get("first")
                last = entry.get("last")
                if isinstance(first, str) and isinstance(last, str):
                    names.append(f"{first} {last}")

    return names


def load_known_actors(
    min_year: int = CONGRESS_MIN_YEAR, max_year: int = CONGRESS_MAX_YEAR
) -> Dict[str, str]:
    """Build a dictionary of known partisan actors from GovTrack data."""

    people_index = _load_people_index()
    if not people_index:
        LOGGER.warning("GovTrack data unavailable; returning static actor list only")
        return STATIC_ACTORS.copy()

    if isinstance(people_index, dict):
        people_iterable = people_index.values()
    else:
        people_iterable = people_index

    actors: Dict[str, str] = dict(STATIC_ACTORS)
    for person in people_iterable:
        if not isinstance(person, dict):
            continue
        names = _collect_name_strings(person)
        if not names:
            continue
        roles = person.get("roles") or person.get("terms") or []
        for role in roles:
            if not isinstance(role, dict):
                continue
            party = _normalize_party(role.get("party"))
            if not party:
                continue
            start = _safe_year(role.get("startdate") or role.get("start"))
            end = _safe_year(role.get("enddate") or role.get("end"))
            if end is None:
                end = max_year
            if start is None:
                start = min_year
            if end < min_year or start > max_year:
                continue
            for name in names:
                for variant in _name_variants(name):
                    if len(variant.split()) < 2:
                        continue
                    existing = actors.get(variant)
                    if existing and existing != party:
                        actors[variant] = "mixed"
                    else:
                        actors[variant] = party
    LOGGER.info("Loaded %s partisan actors from GovTrack", len(actors))
    return actors


KNOWN_ACTORS = load_known_actors()


@lru_cache(maxsize=1)
def get_spacy_model():  # pragma: no cover - heavy dependency
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:  # model missing
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. Run 'python -m spacy download en_core_web_sm'."
        ) from exc


def keyword_score(text: str) -> int:
    """Count obvious partisan keywords to short-circuit LLM calls."""
    lowered = text.lower()
    return sum(1 for kw in PARTY_KEYWORDS if kw in lowered)


def extract_entities(text: str) -> List[str]:
    """Run a lightweight NER pass to find proper nouns in the text."""
    nlp = get_spacy_model()
    doc = nlp(text[:5000])
    ents = [ent.text.lower() for ent in doc.ents if ent.label_ in {"PERSON", "ORG"}]
    return ents


def match_partisan_entities(entities: Iterable[str]) -> List[Tuple[str, str]]:
    """Match NER hits to a curated list of well-known partisan actors."""
    hits = []
    for ent in entities:
        key = ent.lower()
        if key in KNOWN_ACTORS:
            hits.append((ent, KNOWN_ACTORS[key]))
    return hits


def is_potentially_partisan(text: str, keyword_threshold: int = 1) -> bool:
    """Return True if the cheap filters think the text mentions US politics."""
    if keyword_score(text) >= keyword_threshold:
        return True
    ents = extract_entities(text)
    return len(match_partisan_entities(ents)) > 0
