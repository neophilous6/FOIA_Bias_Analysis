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

# URLs for the open-source Congress dataset mirrored on GitHub. GovTrack now
# throttles anonymous bulk downloads from Codespaces-like environments, so we
# default to the unitedstates/congress-legislators repository instead and cache
# the merged payload locally.
CONGRESS_DATA_SOURCES = [
    os.environ.get(
        "CONGRESS_LEGISLATORS_CURRENT_URL",
        "https://raw.githubusercontent.com/unitedstates/congress-legislators/master/legislators-current.json",
    ),
    os.environ.get(
        "CONGRESS_LEGISLATORS_HISTORICAL_URL",
        "https://raw.githubusercontent.com/unitedstates/congress-legislators/master/legislators-historical.json",
    ),
]
CONGRESS_CACHE_PATH = Path(
    os.environ.get("CONGRESS_LEGISLATORS_CACHE", "data/cache/congress_legislators.json")
)
HTTP_USER_AGENT = os.environ.get(
    "FOIA_HTTP_USER_AGENT",
    "foia-bias-analysis/1.0 (+https://github.com/neophilous6/FOIA_Bias_Analysis)",
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

# Presidents and vice presidents do not appear in the congressional dataset,
# so we seed them manually and merge them into the auto-generated dictionary.
MANUAL_ACTORS = [
    # Presidents
    ("bill clinton", "D"),
    ("george w bush", "R"),
    ("barack obama", "D"),
    ("donald trump", "R"),
    ("joe biden", "D"),
    # Vice Presidents (Joe Biden appears twice on purpose; duplicates collapse)
    ("al gore", "D"),
    ("dick cheney", "R"),
    ("joe biden", "D"),
    ("mike pence", "R"),
    ("kamala harris", "D"),
]


def _ensure_cache_dir() -> None:
    CONGRESS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _download_congress_people() -> list[Dict]:
    """Download Congress membership from the GitHub mirror and cache it."""

    aggregated: list[Dict] = []
    for url in CONGRESS_DATA_SOURCES:
        if not url:
            continue
        try:
            LOGGER.info("Downloading congressional roster from %s", url)
            resp = requests.get(url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list):
                aggregated.extend(payload)
            elif isinstance(payload, dict):
                aggregated.append(payload)
        except requests.RequestException as exc:
            LOGGER.warning("Unable to download %s (%s)", url, exc)
    if not aggregated:
        raise RuntimeError("Congressional roster download failed for all sources")
    try:
        _ensure_cache_dir()
        CONGRESS_CACHE_PATH.write_text(json.dumps(aggregated))
    except OSError as exc:  # pragma: no cover - IO heavy
        LOGGER.warning("Unable to cache Congress roster at %s: %s", CONGRESS_CACHE_PATH, exc)
    return aggregated


def _load_people_index() -> Dict:
    """Load the congressional roster from cache or download on demand."""

    if CONGRESS_CACHE_PATH.exists():
        try:
            return json.loads(CONGRESS_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            LOGGER.warning("Cached congressional roster is corrupt; re-downloading")
        except OSError as exc:
            LOGGER.warning("Unable to read Congress cache %s: %s", CONGRESS_CACHE_PATH, exc)
    try:
        return _download_congress_people()
    except RuntimeError as exc:
        LOGGER.warning(
            "Failed to download congressional data (%s). Political entity detection will be limited.",
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
    """Build a dictionary of known partisan actors from congressional data."""

    people_index = _load_people_index()
    if not people_index:
        LOGGER.warning("Congress roster unavailable; returning static actor list only")
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
    # Add presidents/vice presidents explicitly so we always recognize them.
    for name, party in MANUAL_ACTORS:
        for variant in _name_variants(name):
            actors[variant] = party

    LOGGER.info("Loaded %s partisan actors from congressional data", len(actors))
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
