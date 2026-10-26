"""FAOSTAT production and yield downloaders (no secrets required)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

from cropwarn.config import Config
from cropwarn.constants import COUNTRY_BY_ISO3, FAOSTAT_ELEMENTS, FAOSTAT_ITEMS

QCL_BULK = "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
DATA_API = "https://fenixservices.fao.org/faostat/api/v1/en/data/QCL"


def download_qcl_bulk(dest: Path, timeout: int = 180) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(QCL_BULK, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def parse_qcl_zip(zip_path: Path, crops: list[str] | None = None) -> pd.DataFrame:
    crops = crops or list(FAOSTAT_ITEMS)
    item_codes = {FAOSTAT_ITEMS[c] for c in crops}
    elements = set(FAOSTAT_ELEMENTS.values())
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            df = pd.read_csv(fh, encoding="utf-8", low_memory=False)
    df = df[df["Item Code"].isin(item_codes) & df["Element Code"].isin(elements)].copy()
    code_to_crop = {v: k for k, v in FAOSTAT_ITEMS.items()}
    df["crop"] = df["Item Code"].map(code_to_crop)
    return df


def faostat_api_panel(cfg: Config, year_start: int, year_end: int) -> pd.DataFrame:
    """Smaller API pull for the study countries/crops when bulk download is heavy."""

    areas = ",".join(str(c.fao_code) for c in COUNTRY_BY_ISO3.values())
    items = ",".join(str(FAOSTAT_ITEMS[c]) for c in cfg.crops)
    elements = ",".join(str(v) for v in FAOSTAT_ELEMENTS.values())
    params = {
        "area": areas,
        "item": items,
        "element": elements,
        "year": ",".join(str(y) for y in range(year_start, year_end + 1)),
        "output_type": "objects",
    }
    resp = requests.get(DATA_API, params=params, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    records = payload.get("data") or payload
    df = pd.DataFrame(records)
    return df


def tidy_production(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert FAOSTAT long tables to country-crop-year yield / area / production."""

    colmap = {
        "Area Code": "faostat_code",
        "area_code": "faostat_code",
        "Area": "country",
        "area": "country",
        "Item Code": "item_code",
        "item_code": "item_code",
        "Element Code": "element_code",
        "element_code": "element_code",
        "Year": "year",
        "year": "year",
        "Value": "value",
        "value": "value",
        "crop": "crop",
    }
    df = raw.rename(columns={k: v for k, v in colmap.items() if k in raw.columns}).copy()
    if "crop" not in df.columns:
        inv = {v: k for k, v in FAOSTAT_ITEMS.items()}
        df["crop"] = df["item_code"].map(inv)
    inv_el = {v: k for k, v in FAOSTAT_ELEMENTS.items()}
    df["element"] = df["element_code"].map(inv_el)
    wide = (
        df.dropna(subset=["crop", "element"])
        .pivot_table(
            index=["faostat_code", "country", "crop", "year"],
            columns="element",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    if "yield" in wide:
        # FAOSTAT yield is hg/ha = 0.1 t/ha? 5419 is hg/ha, so /10000 * 1? hg/ha / 10000 = t/ha.
        wide["yield_tha"] = wide["yield"] / 1000.0
    if "area_harvested" in wide:
        wide["harvested_ha"] = wide["area_harvested"]
    iso = {c.fao_code: c.iso3 for c in COUNTRY_BY_ISO3.values()}
    region = {c.fao_code: c.region for c in COUNTRY_BY_ISO3.values()}
    wide["iso3"] = wide["faostat_code"].map(iso)
    wide["region"] = wide["faostat_code"].map(region)
    return wide


def save_manifest(path: Path, source: str, extra: dict | None = None) -> None:
    payload = {"source": source, **(extra or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
