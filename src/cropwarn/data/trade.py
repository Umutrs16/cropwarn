"""Bilateral trade ingestion and lagged export-share matrices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from cropwarn.constants import CROP_HS6

FAOSTAT_TM = "https://fenixservices.fao.org/faostat/api/v1/en/data/TM"


def download_faostat_trade(year: int, timeout: int = 120) -> pd.DataFrame:
    params = {
        "element": "5610,5910",  # import / export quantity
        "year": str(year),
        "output_type": "objects",
    }
    resp = requests.get(FAOSTAT_TM, params=params, timeout=timeout)
    resp.raise_for_status()
    return pd.DataFrame(resp.json().get("data", []))


def export_share_matrix(trade: pd.DataFrame, year: int, crop: str) -> pd.DataFrame:
    """W(i->j, c) from the previous complete trade year only."""

    sub = trade[(trade["year"] == year) & (trade["crop"] == crop)].copy()
    if sub.empty:
        return sub
    totals = sub.groupby("importer")["export_share"].transform("sum")
    sub["export_share"] = sub["export_share"] / totals.replace(0, 1.0)
    return sub


def map_baci_hs6(baci: pd.DataFrame, hs_col: str = "k") -> pd.DataFrame:
    rows = []
    for crop, codes in CROP_HS6.items():
        part = baci[baci[hs_col].astype(str).str.zfill(6).isin(codes)].copy()
        if part.empty:
            continue
        part["crop"] = crop
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else baci.iloc[0:0]


def write_trade_readme(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Place FAOSTAT Detailed Trade Matrix or CEPII BACI extracts here.\n"
        "Early-warning runs must use the previous complete trade year only.\n",
        encoding="utf-8",
    )
