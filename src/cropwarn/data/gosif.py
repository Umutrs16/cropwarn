"""Download GOSIF v2 monthly grids and sample cropland points.

GOSIF is reconstructed from OCO-2 + MODIS + reanalysis (Li & Xiao 2019).
It is a true SIF product, not MOD17 GPP. Still add TROPOMI/GOME-2 for the
camera-ready independence check.
"""

from __future__ import annotations

import gzip
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from cropwarn.constants import BREADBASKET_POINTS, TYPICAL_AREA_MHA

GOSIF_MONTHLY_BASE = "https://data.globalecology.unh.edu/data/GOSIF_v2/Monthly"
GOSIF_SCALE = 0.0001
GOSIF_FILL = 32767


def gosif_filename(year: int, month: int) -> str:
    return f"GOSIF_{year}.M{month:02d}.tif.gz"


def gosif_url(year: int, month: int) -> str:
    return f"{GOSIF_MONTHLY_BASE}/{gosif_filename(year, month)}"


def download_gosif_monthly(root: Path, years: list[int], timeout: int = 120) -> list[Path]:
    dest = root / "data_raw" / "gosif"
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in years:
        for month in range(1, 13):
            if year == 2000 and month < 3:
                continue
            name = gosif_filename(year, month)
            path = dest / name
            if path.exists() and path.stat().st_size > 100_000:
                paths.append(path)
                continue
            url = gosif_url(year, month)
            tmp = path.with_suffix(path.suffix + ".part")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "cropwarn/nature-food"})
                with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        fh.write(chunk)
                tmp.replace(path)
                paths.append(path)
                print(f"GOSIF {name}", flush=True)
            except Exception as exc:
                if tmp.exists():
                    tmp.unlink()
                print(f"SKIP GOSIF {name}: {exc}", flush=True)
    return paths


def _sample_points(root: Path) -> pd.DataFrame:
    crop_pts = root / "data_interim" / "crop_points.csv"
    if crop_pts.exists():
        pts = pd.read_csv(crop_pts)
        if {"iso3", "lon", "lat"} <= set(pts.columns) and len(pts) >= 10:
            return pts
    cached = root / "data_interim" / "cropland_points.csv"
    if cached.exists():
        pts = pd.read_csv(cached)
        if {"iso3", "lon", "lat"} <= set(pts.columns) and len(pts) >= 10:
            return pts[["iso3", "lon", "lat"]]
    rows = []
    iso3s = sorted({iso for crop in TYPICAL_AREA_MHA.values() for iso in crop})
    for iso3 in iso3s:
        if iso3 not in BREADBASKET_POINTS:
            continue
        lat, lon = BREADBASKET_POINTS[iso3]
        rows.append({"iso3": iso3, "lon": lon, "lat": lat})
        for dlat, dlon in ((0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4), (0.25, 0.25), (-0.25, -0.25)):
            rows.append({"iso3": iso3, "lon": lon + dlon, "lat": lat + dlat})
    return pd.DataFrame(rows)


def _sample_tif_bytes(raw: bytes, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    from rasterio.io import MemoryFile

    with MemoryFile(raw) as mem, mem.open() as src:
        samples = np.array([v[0] for v in src.sample(zip(lons.tolist(), lats.tolist()))], dtype=float)
    samples[(samples >= GOSIF_FILL - 1) | (samples < -9000)] = np.nan
    return samples * GOSIF_SCALE


def sample_gosif_national(root: Path, years: list[int]) -> pd.DataFrame:
    """Country mean of GOSIF at cropland / breadbasket sample points."""

    pts = _sample_points(root)
    dest = root / "data_raw" / "gosif"
    rows = []
    for year in years:
        for month in range(1, 13):
            path = dest / gosif_filename(year, month)
            if not path.exists() or path.stat().st_size < 100_000:
                continue
            try:
                with gzip.open(path, "rb") as fh:
                    raw = fh.read()
                vals = _sample_tif_bytes(raw, pts["lon"].to_numpy(), pts["lat"].to_numpy())
            except Exception as exc:
                print(f"SKIP sample {path.name}: {exc}", flush=True)
                continue
            tmp = pts.copy()
            tmp["sif"] = vals
            keys = ["iso3", "crop"] if "crop" in tmp.columns else ["iso3"]
            agg = tmp.groupby(keys, as_index=False)["sif"].mean()
            agg["date"] = pd.Timestamp(year=year, month=month, day=15)
            rows.append(agg)
            print(f"sampled {path.name}", flush=True)
    if not rows:
        return pd.DataFrame(columns=["iso3", "date", "sif"])
    out = pd.concat(rows, ignore_index=True)
    cache = root / "data_interim" / ("gosif_crop.csv" if "crop" in out.columns else "gosif_national.csv")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def sample_gosif_points(root: Path, pts: pd.DataFrame, years: list[int], cache_name: str) -> pd.DataFrame:
    """Sample already-downloaded GOSIF monthly grids at an arbitrary point table."""

    dest = root / "data_raw" / "gosif"
    if pts.empty or not {"lon", "lat"} <= set(pts.columns):
        return pd.DataFrame()
    rows = []
    for year in years:
        for month in range(1, 13):
            path = dest / gosif_filename(year, month)
            if not path.exists() or path.stat().st_size < 100_000:
                continue
            try:
                with gzip.open(path, "rb") as fh:
                    raw = fh.read()
                vals = _sample_tif_bytes(raw, pts["lon"].to_numpy(), pts["lat"].to_numpy())
            except Exception as exc:
                print(f"SKIP sample {path.name}: {exc}", flush=True)
                continue
            tmp = pts.copy()
            tmp["sif"] = vals
            keys = [c for c in ("iso3", "unit_id", "crop") if c in tmp.columns]
            agg = tmp.groupby(keys, as_index=False)["sif"].mean()
            agg["date"] = pd.Timestamp(year=year, month=month, day=15)
            rows.append(agg)
            print(f"sampled {path.name} -> {cache_name}", flush=True)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cache = root / "data_interim" / cache_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out
