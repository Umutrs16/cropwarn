"""Leakage-safe ENSO baseline: OND ONI of year t−1 (fully preseason)."""

from __future__ import annotations

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"


def download_oni(root: Path, timeout: int = 60) -> Path:
    dest = root / "data_raw" / "enso" / "oni.ascii.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 200:
        return dest
    req = urllib.request.Request(ONI_URL, headers={"User-Agent": "cropwarn/nature-food"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dest.write_bytes(resp.read())
    return dest


def parse_oni(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    rows = []
    for line in raw[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            season, year, oni = parts[0], int(parts[1]), float(parts[3] if len(parts) > 3 else parts[2])
        except ValueError:
            continue
        # CPC file is SEAS  YR   TOTAL   ANOM  — season first.
        if parts[0].isalpha():
            season, year, oni = parts[0], int(parts[1]), float(parts[-1])
        rows.append({"season": season, "year": year, "oni": oni})
    return pd.DataFrame(rows)


def preseason_oni(root: Path) -> pd.DataFrame:
    """ONI for forecast year t = OND mean of year t−1 (known before sowing in most NH systems)."""

    path = download_oni(root)
    oni = parse_oni(path)
    ond = oni[oni["season"].str.upper().eq("OND")][["year", "oni"]].rename(columns={"oni": "oni_ond"})
    out = ond.copy()
    out["year"] = out["year"] + 1
    out = out.rename(columns={"oni_ond": "oni_preseason"})
    return out


def enso_baseline_predictions(panel: pd.DataFrame, oni: pd.DataFrame, test_start: int = 2016) -> pd.DataFrame:
    """Causal linear yield_anomaly ~ ONI_preseason. Fit on years < t only."""

    keys = [c for c in ("iso3", "crop", "year", "yield_anomaly") if c in panel.columns]
    p = panel[keys].drop_duplicates().merge(oni, on="year", how="left")
    rows = []
    for (iso3, crop), g in p.groupby(["iso3", "crop"], sort=False):
        g = g.sort_values("year")
        for i, rec in g.iterrows():
            year = int(rec["year"])
            if year < test_start:
                continue
            hist = g[(g["year"] < year) & g["yield_anomaly"].notna() & g["oni_preseason"].notna()]
            if len(hist) < 6 or not np.isfinite(rec.get("oni_preseason")):
                pred = np.nan
            else:
                x = hist["oni_preseason"].to_numpy(dtype=float)
                y = hist["yield_anomaly"].to_numpy(dtype=float)
                coef = np.polyfit(x, y, 1)
                pred = float(np.polyval(coef, float(rec["oni_preseason"])))
            rows.append(
                {
                    "iso3": iso3,
                    "crop": crop,
                    "year": year,
                    "yield_anomaly": rec["yield_anomaly"],
                    "pred": pred,
                    "oni_preseason": rec.get("oni_preseason"),
                    "experiment": "ENSO_preseason",
                }
            )
    return pd.DataFrame(rows)
