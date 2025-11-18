"""Aggregate labeled data into modeling-ready frames."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


PARTISAN_PARTIES = {"D", "R"}


def infer_party_target(targets: list[dict]) -> str:
    if not targets or not isinstance(targets, list):
        return "none"
    parties = [t.get("party") for t in targets if t.get("party") in PARTISAN_PARTIES]
    if not parties:
        return "unknown"
    return max(set(parties), key=parties.count)


def prepare_for_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    df["party_target"] = df["targets"].apply(infer_party_target)
    df = df[df["party_target"].isin(PARTISAN_PARTIES)]
    df["same_party"] = (df["party_target"] == df["admin_party"]).astype(int)
    df["wrongdoing_any"] = (
        (df["wrongdoing_D"] > 0.5) | (df["wrongdoing_R"] > 0.5)
    ).astype(int)
    df["fav_diff"] = df["fav_score_D"] - df["fav_score_R"]
    min_year = config.get("analysis", {}).get("min_year", 0)
    max_year = config.get("analysis", {}).get("max_year", 9999)
    df["year"] = pd.to_datetime(df["date_done"], errors="coerce").dt.year
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
    return df
