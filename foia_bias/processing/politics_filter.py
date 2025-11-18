"""Keyword + NER prefilter for partisan content."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple

import spacy

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

KNOWN_ACTORS = {
    "joe biden": "D",
    "joseph r. biden": "D",
    "donald trump": "R",
    "barack obama": "D",
    "hillary clinton": "D",
    "mitch mcconnell": "R",
    "nancy pelosi": "D",
    "kevin mccarthy": "R",
    "dnc": "D",
    "rnc": "R",
}


@lru_cache(maxsize=1)
def get_spacy_model():  # pragma: no cover - heavy dependency
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:  # model missing
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. Run 'python -m spacy download en_core_web_sm'."
        ) from exc


def keyword_score(text: str) -> int:
    lowered = text.lower()
    return sum(1 for kw in PARTY_KEYWORDS if kw in lowered)


def extract_entities(text: str) -> List[str]:
    nlp = get_spacy_model()
    doc = nlp(text[:5000])
    ents = [ent.text.lower() for ent in doc.ents if ent.label_ in {"PERSON", "ORG"}]
    return ents


def match_partisan_entities(entities: Iterable[str]) -> List[Tuple[str, str]]:
    hits = []
    for ent in entities:
        key = ent.lower()
        if key in KNOWN_ACTORS:
            hits.append((ent, KNOWN_ACTORS[key]))
    return hits


def is_potentially_partisan(text: str, keyword_threshold: int = 1) -> bool:
    if keyword_score(text) >= keyword_threshold:
        return True
    ents = extract_entities(text)
    return len(match_partisan_entities(ents)) > 0
