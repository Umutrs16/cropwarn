"""TROPOMI observed SIF at crop points. Independent of GOSIF reconstruction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cropwarn.constants import GEE_PROJECT
from cropwarn.gee.crop_extract import _parse_point_table, _points_fc
from cropwarn.gee.export import init_ee

TROPOMI_SIF = "COPERNICUS/S5P/OFFL/L2_SIF"
TROPOMI_START_YEAR = 2018


def extract_tropomi_year(ee, year: int, crop: str, pts: pd.DataFrame, dest: Path) -> pd.DataFrame:
    cache = dest / f"tropomi_{crop}_{year}.csv"
    if cache.exists() and cache.stat().st_size > 200:
        return pd.read_csv(cache, parse_dates=["date"])
    sub = pts[pts["crop"] == crop]
    if sub.empty or year < TROPOMI_START_YEAR:
        return pd.DataFrame()
    fc = _points_fc(ee, sub)
    col = ee.ImageCollection(TROPOMI_SIF).select("SIF_743nm")
    clean = []
    for month in range(1, 13):
        if year == TROPOMI_START_YEAR and month < 5:
            continue
        start = ee.Date.fromYMD(year, month, 1)
        month_col = col.filterDate(start, start.advance(1, "month"))
        img = ee.Image(
            ee.Algorithms.If(
                month_col.size().gt(0),
                month_col.mean().rename(f"m{month:02d}_tropomi_sif"),
                ee.Image.constant(0).updateMask(0).rename(f"m{month:02d}_tropomi_sif"),
            )
        )
        clean.append(img)
    if not clean:
        return pd.DataFrame()
    table = ee.Image.cat(clean).reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=7000, tileScale=2)
    out = _parse_point_table(table.getInfo(), year, "era5")
    if out.empty:
        return out
    if "tropomi_sif" not in out.columns:
        return pd.DataFrame()
    out = out[["iso3", "crop", "date", "year", "tropomi_sif"]].dropna(subset=["tropomi_sif"])
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def extract_tropomi(root: Path, years: list[int], pts: pd.DataFrame, crops: list[str] | None = None) -> pd.DataFrame:
    ee = init_ee(GEE_PROJECT)
    dest = root / "data_raw" / "tropomi"
    dest.mkdir(parents=True, exist_ok=True)
    crops = crops or sorted(pts["crop"].unique())
    frames = []
    for crop in crops:
        for year in years:
            if year < TROPOMI_START_YEAR:
                continue
            try:
                part = extract_tropomi_year(ee, year, crop, pts, dest)
                if len(part):
                    frames.append(part)
                    print(f"TROPOMI {crop} {year} rows={len(part)}", flush=True)
            except Exception as exc:
                print(f"SKIP TROPOMI {crop} {year}: {exc}", flush=True)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    interim = root / "data_interim" / "tropomi_crop.csv"
    out.to_csv(interim, index=False)
    return out


def load_tropomi(root: Path) -> pd.DataFrame:
    path = root / "data_interim" / "tropomi_crop.csv"
    if path.exists() and path.stat().st_size > 50:
        return pd.read_csv(path, parse_dates=["date"])
    raw = root / "data_raw" / "tropomi"
    if not raw.exists():
        return pd.DataFrame()
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in sorted(raw.glob("tropomi_*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
