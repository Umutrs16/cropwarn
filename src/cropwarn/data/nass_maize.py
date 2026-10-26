"""US NASS county maize (corn grain) yields for the 2012 drought belt.

FAO national USA maize mixes the whole country and does not couple to
CROPGRIDS points. County grain yield does: 192-day leads are late March
(no canopy), while mid-season SIF/VOD can see the 2012 flash drought.
Source: NASS Quick Stats via productofamerica/usda-county-yields.
"""

from __future__ import annotations

from pathlib import Path
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from cropwarn.data.cropgrids import nc_path

# Corn grain: 56 lb/bu, acre -> hectare.
BU_ACRE_TO_T_HA = 56.0 * 0.45359237 / 1000.0 / 0.40468564224

MAIZE_CALENDAR = (120, 280)

# Corn-belt states that took the 2012 hit (plus MN/WI/SD for contrast).
CORN_BELT = {
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "27": "MN",
    "29": "MO",
    "31": "NE",
    "39": "OH",
    "46": "SD",
    "55": "WI",
}

CORN_JSON_URLS = (
    "https://cdn.jsdelivr.net/gh/productofamerica/usda-county-yields@main/data/states/{fips}/crops/corn.json",
    "https://raw.githubusercontent.com/productofamerica/usda-county-yields/main/data/states/{fips}/crops/corn.json",
)
GAZETTEER_ZIP = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip"


def _fetch(url: str, dest: Path, timeout: int = 180) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 200:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "cropwarn/nature-food"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    tmp.replace(dest)
    return dest


def download_county_json(root: Path, fips: str) -> Path:
    dest = root / "data_raw" / "nass_maize" / f"{fips}_corn.json"
    last = None
    for url in CORN_JSON_URLS:
        try:
            return _fetch(url.format(fips=fips), dest)
        except Exception as exc:
            last = exc
    raise last


def _canonical_yield_series(county: dict) -> dict[int, float]:
    for series in county.get("series") or []:
        if not series.get("canonical"):
            continue
        if str(series.get("statistic", "")).upper() != "YIELD":
            continue
        util = str(series.get("util_practice", "")).upper()
        if util and util not in {"GRAIN", "ALL UTILIZATION PRACTICES"}:
            continue
        unit = str(series.get("unit", "")).upper()
        if "BU" not in unit:
            continue
        out = {}
        for year, val in (series.get("values") or {}).items():
            try:
                y = int(year)
                v = float(val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                out[y] = v
        if out:
            return out
    return {}


def parse_state_corn_json(path: Path, state_fips: str, year_start: int = 2001, year_end: int = 2024) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    abbr = CORN_BELT[state_fips]
    rows = []
    for cnty, rec in (raw.get("counties") or {}).items():
        series = _canonical_yield_series(rec)
        if not series:
            continue
        name = str(rec.get("name") or cnty).title()
        for year, bu in series.items():
            if year < year_start or year > year_end:
                continue
            rows.append(
                {
                    "iso3": "USA",
                    "unit_id": f"USA-{abbr}-{str(cnty).zfill(3)}",
                    "state_fips": state_fips,
                    "state_code": abbr,
                    "county_fips": str(cnty).zfill(3),
                    "geoid": f"{state_fips}{str(cnty).zfill(3)}",
                    "county": name,
                    "crop": "maize",
                    "year": int(year),
                    "yield_bu_acre": float(bu),
                    "yield_tha": float(bu) * BU_ACRE_TO_T_HA,
                    "source": "nass_usda_county_yields",
                }
            )
    return pd.DataFrame(rows)


def load_gazetteer(root: Path) -> pd.DataFrame:
    dest = root / "data_raw" / "nass_maize" / "2024_Gaz_counties_national.txt"
    if not dest.exists() or dest.stat().st_size < 1000:
        zpath = dest.with_suffix(".zip")
        _fetch(GAZETTEER_ZIP, zpath, timeout=120)
        with zipfile.ZipFile(zpath) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
            dest.write_bytes(zf.read(name))
    gaz = pd.read_csv(dest, sep="\t", dtype=str)
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["geoid"] = gaz["GEOID"].astype(str).str.zfill(5)
    gaz["lat"] = pd.to_numeric(gaz["INTPTLAT"], errors="coerce")
    gaz["lon"] = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")
    return gaz[["geoid", "lat", "lon", "USPS", "NAME"]].dropna(subset=["lat", "lon"])


def _select_counties(labels: pd.DataFrame, n_per_state: int = 5, min_years: int = 16) -> list[str]:
    """Keep established corn counties with a 2012 observation."""

    pre = labels[labels["year"].between(2008, 2011)].groupby("unit_id")["yield_tha"].mean()
    y2012 = labels[labels["year"].eq(2012)].groupby("unit_id")["yield_tha"].mean()
    stats = labels.groupby("unit_id").agg(n=("year", "nunique"), state_code=("state_code", "first")).reset_index()
    stats["pre"] = stats["unit_id"].map(pre)
    stats["y2012"] = stats["unit_id"].map(y2012)
    stats["has_2012"] = stats["y2012"].notna()
    stats = stats[stats["n"].ge(min_years) & stats["has_2012"] & stats["pre"].notna()].copy()
    keep = []
    for _, g in stats.groupby("state_code", sort=False):
        g = g.sort_values("pre", ascending=False)
        keep.extend(g["unit_id"].head(n_per_state).tolist())
    return keep


def parse_usa_maize(root: Path, year_start: int = 2001, year_end: int = 2024, n_per_state: int = 5) -> pd.DataFrame:
    cache = root / "data_interim" / "usa_maize.csv"
    frames = []
    for fips in CORN_BELT:
        path = download_county_json(root, fips)
        frames.append(parse_state_corn_json(path, fips, year_start, year_end))
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw
    keep = _select_counties(raw, n_per_state=n_per_state)
    out = raw[raw["unit_id"].isin(keep)].copy()
    out["harvested_ha"] = np.nan
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def state_unit_id(unit_id) -> str:
    parts = str(unit_id).split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else str(unit_id)


def load_all_county_maize(root: Path, year_start: int = 2001, year_end: int = 2024) -> pd.DataFrame:
    frames = []
    for fips in CORN_BELT:
        path = root / "data_raw" / "nass_maize" / f"{fips}_corn.json"
        if not path.exists() or path.stat().st_size < 200:
            continue
        frames.append(parse_state_corn_json(path, fips, year_start, year_end))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _county_cropgrids_weight(root: Path, counties: pd.DataFrame) -> pd.Series:
    """One CROPGRIDS maize cell at each county centroid — area proxy for weighting."""

    import xarray as xr

    path = nc_path(root, "maize")
    if not path.exists() or counties.empty:
        return pd.Series(dtype=float)
    gaz = load_gazetteer(root)
    u = counties[["unit_id", "geoid"]].drop_duplicates()
    u["geoid"] = u["geoid"].astype(str).str.zfill(5)
    u = u.merge(gaz, on="geoid", how="left").dropna(subset=["lat", "lon"])
    ds = xr.open_dataset(path)
    ha = ds["harvarea"]
    weights = {}
    for rec in u.itertuples(index=False):
        try:
            val = float(ha.sel(lon=float(rec.lon), lat=float(rec.lat), method="nearest").values)
        except Exception:
            val = np.nan
        weights[rec.unit_id] = val if np.isfinite(val) and val > 0 else np.nan
    ds.close()
    return pd.Series(weights, name="cg_ha")


def parse_usa_maize_state(root: Path, year_start: int = 2001, year_end: int = 2024) -> pd.DataFrame:
    """Statewide maize yield: CROPGRIDS-weighted county mean, else median."""

    raw = load_all_county_maize(root, year_start, year_end)
    if raw.empty:
        return raw
    w = _county_cropgrids_weight(root, raw)
    raw = raw.copy()
    raw["cg_ha"] = raw["unit_id"].map(w)
    rows = []
    for (code, year), g in raw.groupby(["state_code", "year"], sort=False):
        ww = g["cg_ha"].to_numpy(dtype=float)
        yy = g["yield_tha"].to_numpy(dtype=float)
        ok = np.isfinite(yy)
        if ok.sum() < 8:
            continue
        if np.isfinite(ww).sum() >= 8 and np.nansum(ww[ok]) > 0:
            yld = float(np.nansum(yy[ok] * ww[ok]) / np.nansum(ww[ok]))
            src = "nass_county_cropgrids_weighted"
        else:
            yld = float(np.nanmedian(yy[ok]))
            src = "nass_county_median"
        rows.append(
            {
                "iso3": "USA",
                "unit_id": f"USA-{code}",
                "state_code": code,
                "crop": "maize",
                "year": int(year),
                "yield_tha": yld,
                "harvested_ha": float(np.nansum(ww)) if np.isfinite(ww).any() else np.nan,
                "n_counties": int(ok.sum()),
                "source": src,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    counts = out.groupby("unit_id")["year"].nunique()
    out = out[out["unit_id"].isin(counts[counts >= 16].index)].copy()
    cache = root / "data_interim" / "usa_maize_state.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def load_usa_maize_state(root: Path) -> pd.DataFrame:
    path = root / "data_interim" / "usa_maize_state.csv"
    if not path.exists() or path.stat().st_size < 200:
        return pd.DataFrame()
    return pd.read_csv(path)


def rollup_to_state(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "unit_id" not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out["unit_id"] = out["unit_id"].map(state_unit_id)
    keys = [c for c in ("iso3", "unit_id", "crop", "date", "year") if c in out.columns]
    num = [c for c in out.columns if c not in keys and np.issubdtype(out[c].dtype, np.number)]
    if not keys or not num:
        return pd.DataFrame()
    return out.groupby(keys, as_index=False)[num].mean()


def load_usa_maize(root: Path) -> pd.DataFrame:
    path = root / "data_interim" / "usa_maize.csv"
    if not path.exists() or path.stat().st_size < 200:
        return pd.DataFrame()
    out = pd.read_csv(path, dtype={"geoid": str, "county_fips": str, "state_fips": str})
    if "geoid" in out.columns:
        out["geoid"] = out["geoid"].astype(str).str.zfill(5)
    if "county_fips" in out.columns:
        out["county_fips"] = out["county_fips"].astype(str).str.zfill(3)
    return out


def usa_maize_points(root: Path, n_per: int = 3, radius_deg: float = 0.40, min_ha: float = 20.0) -> pd.DataFrame:
    import xarray as xr

    labels = load_usa_maize(root)
    if labels.empty:
        return pd.DataFrame()
    path = nc_path(root, "maize")
    if not path.exists():
        return pd.DataFrame()
    gaz = load_gazetteer(root)
    units = labels[["unit_id", "state_code", "geoid", "county"]].drop_duplicates()
    units["geoid"] = units["geoid"].astype(str).str.zfill(5)
    gaz = gaz.copy()
    gaz["geoid"] = gaz["geoid"].astype(str).str.zfill(5)
    units = units.merge(gaz, on="geoid", how="left")
    ds = xr.open_dataset(path)
    ha = ds["harvarea"]
    lat_desc = float(ha.lat[0]) > float(ha.lat[-1])
    rows = []
    for rec in units.itertuples(index=False):
        if not np.isfinite(getattr(rec, "lat", np.nan)):
            continue
        lon0, lat0 = float(rec.lon), float(rec.lat)
        if lat_desc:
            sub = ha.sel(lon=slice(lon0 - radius_deg, lon0 + radius_deg), lat=slice(lat0 + radius_deg, lat0 - radius_deg))
        else:
            sub = ha.sel(lon=slice(lon0 - radius_deg, lon0 + radius_deg), lat=slice(lat0 - radius_deg, lat0 + radius_deg))
        vals = np.asarray(sub.values, dtype=float)
        lats = sub["lat"].values
        lons = sub["lon"].values
        if vals.size == 0:
            continue
        vals[~np.isfinite(vals)] = 0
        flat = vals.ravel()
        if flat.max() <= 0:
            continue
        k = min(n_per, int((flat >= min_ha).sum()) or int((flat > 0).sum()) or 1)
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]
        width = vals.shape[1]
        for i in idx:
            r, c = divmod(int(i), width)
            w = float(flat[i])
            if w <= 0:
                continue
            rows.append(
                {
                    "iso3": "USA",
                    "unit_id": rec.unit_id,
                    "state_code": rec.state_code,
                    "crop": "maize",
                    "lon": float(lons[c]),
                    "lat": float(lats[r]),
                    "weight": w,
                    "source": "cropgrids2020",
                }
            )
    ds.close()
    out = pd.DataFrame(rows)
    cache = root / "data_interim" / "usa_maize_points.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
        labels = labels.copy()
        w = out.groupby("unit_id")["weight"].sum()
        labels["harvested_ha"] = labels["unit_id"].map(w) * 100.0
        labels.to_csv(root / "data_interim" / "usa_maize.csv", index=False)
    return out
