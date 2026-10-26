"""VODCA v2 L-band canopy VOD (Moesinger et al.; TU Wien t74ty-tcx62).

10-daily 0.25° grids, 2010–2021. This is microwave VOD, not SMAP vegetation
water content. Prefer it over the 2015+ SMAP VWC fallback when both exist.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

VODCA_L_URL = "https://researchdata.tuwien.at/records/t74ty-tcx62/files/VODCA_L.zip?download=1"
VODCA_VAR = "VODCA_L"


def vodca_zip_path(root: Path) -> Path:
    return root / "data_raw" / "vodca" / "VODCA_L.zip"


def _point_table(root: Path) -> pd.DataFrame:
    frames = []
    for name in ("crop_points.csv", "india_rice_points.csv", "india_rice_district_points.csv", "china_early_rice_points.csv", "usa_maize_points.csv"):
        path = root / "data_interim" / name
        if path.exists() and path.stat().st_size > 50:
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    pts = pd.concat(frames, ignore_index=True)
    if "unit_id" not in pts.columns:
        pts["unit_id"] = pts["iso3"]
    else:
        pts["unit_id"] = pts["unit_id"].fillna(pts["iso3"])
    return pts.dropna(subset=["lon", "lat"])


def _nearest_index(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.abs(coord.reshape(-1, 1) - values.reshape(1, -1)).argmin(axis=0)


def extract_vodca_zip(root: Path) -> Path:
    dest = root / "data_raw" / "vodca"
    out = dest / "daily_images_VODCA_L"
    existing = list(out.rglob("*.nc")) if out.exists() else []
    if len(existing) >= 400:
        return out
    zip_path = vodca_zip_path(root)
    print(f"extracting {zip_path} -> {dest}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return out


def _sample_nc_path(path: Path, ilat: np.ndarray, ilon: np.ndarray) -> tuple[pd.Timestamp | None, np.ndarray]:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if VODCA_VAR not in ds:
            return None, np.full(len(ilat), np.nan)
        arr = np.asarray(ds[VODCA_VAR].values, dtype=float)
        if arr.ndim == 3:
            arr = arr[0]
        date = pd.Timestamp(np.asarray(ds["time"].values)[0]) if "time" in ds else None
        vals = arr[ilat, ilon]
        vals[~np.isfinite(vals)] = np.nan
        vals[(vals < 0) | (vals > 3)] = np.nan
        return date, vals


def sample_vodca_crop(root: Path) -> pd.DataFrame:
    """Monthly mean VOD at crop / India-rice sample points."""

    zip_path = vodca_zip_path(root)
    cache = root / "data_interim" / "vodca_crop.csv"
    if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
        return pd.DataFrame()
    pts = _point_table(root)
    if pts.empty:
        return pd.DataFrame()

    extracted = extract_vodca_zip(root)
    names = sorted(extracted.rglob("*.nc"))
    if not names:
        return pd.DataFrame()
    import xarray as xr

    with xr.open_dataset(names[0]) as ds:
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
    ilat = _nearest_index(lat, pts["lat"].to_numpy(dtype=float))
    ilon = _nearest_index(lon, pts["lon"].to_numpy(dtype=float))

    rows = []
    for i, path in enumerate(names):
        try:
            date, vals = _sample_nc_path(path, ilat, ilon)
        except Exception as exc:
            print(f"SKIP VODCA {path.name}: {exc}", flush=True)
            continue
        if date is None:
            continue
        tmp = pts[["iso3", "crop", "unit_id"]].copy()
        tmp["date"] = date
        tmp["vod"] = vals
        rows.append(tmp)
        if (i + 1) % 36 == 0:
            print(f"VODCA sampled {i + 1}/{len(names)}", flush=True)

    if not rows:
        return pd.DataFrame()
    raw = pd.concat(rows, ignore_index=True)
    raw["month"] = pd.to_datetime(raw["date"]).dt.to_period("M")
    keys = ["iso3", "crop", "unit_id", "month"]
    out = raw.groupby(keys, as_index=False)["vod"].mean()
    out["date"] = out["month"].dt.to_timestamp(how="start") + pd.Timedelta(days=14)
    out = out.drop(columns=["month"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"VODCA rows={len(out)} units={out['unit_id'].nunique()}", flush=True)
    return out


def load_vodca_crop(root: Path) -> pd.DataFrame:
    path = root / "data_interim" / "vodca_crop.csv"
    if not path.exists() or path.stat().st_size < 50:
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])
