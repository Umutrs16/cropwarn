"""Observed TROPOSIF L2B (Guanter et al. 2021) sampled at crop points.

SRON public L2B covers May 2018–2021. Mid-month soundings are streamed,
sampled, then deleted. This is observed TROPOMI SIF, not a MODIS reconstruction.
"""

from __future__ import annotations

import re
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

SRON_BASE = "https://ftp.sron.nl/open-access-data-2/TROPOMI/tropomi/sif/v2.1/l2b"
UA = {"User-Agent": "cropwarn/nature-food"}


def _list_days(year: int, month: int) -> list[str]:
    url = f"{SRON_BASE}/{year}/{month:02d}/"
    try:
        html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read().decode("utf-8", "replace")
    except Exception:
        return []
    return sorted(set(re.findall(r"TROPOSIF_L2B_(\d{4}-\d{2}-\d{2})\.nc", html)))


def _pick_midmonth(days: list[str]) -> str | None:
    if not days:
        return None
    target = 15
    return min(days, key=lambda d: abs(int(d[-2:]) - target))


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
    return dest


def _sample_file(path: Path, pts: pd.DataFrame, bin_deg: float = 0.5) -> pd.DataFrame:
    import netCDF4

    ds = netCDF4.Dataset(path)
    prod = ds.groups["PRODUCT"]
    lat = np.asarray(prod.variables["latitude"][:], dtype=float)
    lon = np.asarray(prod.variables["longitude"][:], dtype=float)
    sif = np.asarray(prod.variables["SIF_Corr_743"][:], dtype=float)
    ds.close()
    ok = np.isfinite(sif) & np.isfinite(lat) & np.isfinite(lon) & (np.abs(sif) < 5)
    lat, lon, sif = lat[ok], lon[ok], sif[ok]
    blat = np.round(lat / bin_deg) * bin_deg
    blon = np.round(lon / bin_deg) * bin_deg
    grid = pd.DataFrame({"blat": blat, "blon": blon, "sif": sif}).groupby(["blat", "blon"], as_index=False)["sif"].mean()
    p = pts.copy()
    p["blat"] = np.round(p["lat"].to_numpy(dtype=float) / bin_deg) * bin_deg
    p["blon"] = np.round(p["lon"].to_numpy(dtype=float) / bin_deg) * bin_deg
    out = p.merge(grid, on=["blat", "blon"], how="left")
    out = out.rename(columns={"sif": "tropomi_sif"})
    keep = [c for c in ("iso3", "crop", "lat", "lon", "tropomi_sif") if c in out.columns]
    return out[keep]


def sample_tropomi_sron(root: Path, years: list[int], pts: pd.DataFrame) -> pd.DataFrame:
    dest = root / "data_raw" / "tropomi"
    dest.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in years:
        for month in range(1, 13):
            if year == 2018 and month < 5:
                continue
            days = _list_days(year, month)
            day = _pick_midmonth(days)
            if day is None:
                print(f"SKIP TROPOSIF {year}-{month:02d}: no days", flush=True)
                continue
            cache_month = dest / f"tropomi_month_{day}.csv"
            if cache_month.exists() and cache_month.stat().st_size > 50:
                part = pd.read_csv(cache_month)
            else:
                url = f"{SRON_BASE}/{year}/{month:02d}/TROPOSIF_L2B_{day}.nc"
                nc = dest / f"TROPOSIF_L2B_{day}.nc"
                try:
                    print(f"TROPOSIF download {day}", flush=True)
                    _download(url, nc)
                    part = _sample_file(nc, pts)
                    part["date"] = pd.Timestamp(day)
                    part["year"] = year
                    part.to_csv(cache_month, index=False)
                except Exception as exc:
                    print(f"SKIP TROPOSIF {day}: {exc}", flush=True)
                    if nc.exists():
                        nc.unlink()
                    continue
                if nc.exists():
                    nc.unlink()
            if "date" not in part.columns:
                part["date"] = pd.Timestamp(day)
            if "year" not in part.columns:
                part["year"] = year
            frames.append(part)
            print(f"TROPOSIF {day} rows={len(part)}", flush=True)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    keys = [c for c in ("iso3", "crop", "date") if c in raw.columns]
    out = raw.groupby(keys, as_index=False)["tropomi_sif"].mean()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    interim = root / "data_interim" / "tropomi_crop.csv"
    out.to_csv(interim, index=False)
    return out
