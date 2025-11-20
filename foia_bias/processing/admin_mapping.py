"""Map dates to presidential administrations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

ADMIN_PERIODS = [
    ("Clinton", "D", date(1993, 1, 20), date(2001, 1, 20)),
    ("Bush", "R", date(2001, 1, 20), date(2009, 1, 20)),
    ("Obama", "D", date(2009, 1, 20), date(2017, 1, 20)),
    ("Trump", "R", date(2017, 1, 20), date(2021, 1, 20)),
    ("Biden", "D", date(2021, 1, 20), date(2025, 1, 20)),
]


def parse_date(value: str | None) -> Optional[date]:
    """Parse ISO8601 date strings while tolerating missing values."""
    if not value:
        return None
    return datetime.fromisoformat(value).date()


def get_admin_for_date(value: str | date | None, transition_months: int = 0) -> dict[str, str | None]:
    """Return the administration + optional transition flag for a date."""
    if isinstance(value, str):
        parsed = parse_date(value)
    else:
        parsed = value
    if not parsed:
        return {"admin_name": None, "admin_party": None, "is_transition": False}
    for name, party, start, end in ADMIN_PERIODS:
        # Optional transition windows allow us to keep track of liminal
        # periods around inaugurations (useful for robustness checks).
        adj_start = start - timedelta(days=30 * transition_months)
        adj_end = end + timedelta(days=30 * transition_months)
        if adj_start <= parsed < adj_end:
            is_transition = not (start <= parsed < end)
            return {"admin_name": name, "admin_party": party, "is_transition": is_transition}
    return {"admin_name": None, "admin_party": None, "is_transition": False}
