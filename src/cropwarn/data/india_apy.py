"""India DES/APY seasonal rice yields (kharif / Aman), state level.

FAO annual rice is 2–3 seasons stacked and is not a valid early-warning
target. State kharif/Aman yield is the monsoon crop that matches calendar
(170, 320). Production in some APY mirrors is corrupt; use the Yield column
(t/ha) area-weighted to the state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cropwarn.data.cropgrids import nc_path

# Agricultural-year 2019-20 -> harvest year 2019 for Kharif/Aman.
KHARIF_CALENDAR = (170, 320)

# Main monsoon rice season in the APY table (not rabi/boro/summer).
MAIN_MONSOON_SEASON = {
    "west bengal": "winter",
    "odisha": "winter",
    "assam": "winter",
    "bihar": "winter",
    "jharkhand": "winter",
}

HEADLINE_STATES = {
    "uttar pradesh": "UP",
    "chhattisgarh": "CG",
    "andhra pradesh": "AP",
    "punjab": "PB",
    "west bengal": "WB",
    "odisha": "OD",
    "bihar": "BR",
    "assam": "AS",
    "tamil nadu": "TN",
    "haryana": "HR",
    "madhya pradesh": "MP",
    "karnataka": "KA",
    "jharkhand": "JH",
}


def _norm_state(name: str) -> str:
    return " ".join(str(name).lower().replace("&", "and").split())


def _year_from_apy(value) -> int | None:
    text = str(value).strip()
    if not text or text == "nan":
        return None
    token = text[:4]
    if token.isdigit():
        return int(token)
    return None


def _find_apy_csv(root: Path) -> Path | None:
    candidates = [
        root / "data_raw" / "india_apy" / "India Agriculture Crop Production.csv",
        root / "data_raw" / "india_apy" / "repo" / "data" / "India Agriculture Crop Production.csv",
        root / "data_raw" / "india_apy" / "crop_production.csv",
        root / "data_raw" / "india_apy" / "india_apy.csv",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 100_000:
            return path
    return None


def parse_india_kharif_rice(root: Path, year_start: int = 2001, year_end: int = 2019) -> pd.DataFrame:
    path = _find_apy_csv(root)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def col(*names: str) -> str | None:
        for name in names:
            if name in cols:
                return cols[name]
        return None

    state_c = col("state", "state_name")
    crop_c = col("crop")
    year_c = col("year", "crop_year")
    season_c = col("season")
    area_c = col("area")
    yield_c = col("yield")
    if not all([state_c, crop_c, year_c, season_c, area_c, yield_c]):
        return pd.DataFrame()

    rice = df[df[crop_c].astype(str).str.strip().str.lower().eq("rice")].copy()
    rice["state_key"] = rice[state_c].map(_norm_state)
    rice["season_key"] = rice[season_c].astype(str).str.strip().str.lower()
    rice["year"] = rice[year_c].map(_year_from_apy)
    rice["area_ha"] = pd.to_numeric(rice[area_c], errors="coerce")
    rice["yield_tha"] = pd.to_numeric(rice[yield_c], errors="coerce")
    rice = rice[rice["state_key"].isin(HEADLINE_STATES)]
    rice = rice[rice["year"].between(year_start, year_end)]

    keep = []
    for state_key, code in HEADLINE_STATES.items():
        season = MAIN_MONSOON_SEASON.get(state_key, "kharif")
        sub = rice[(rice["state_key"] == state_key) & (rice["season_key"] == season)]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["state_code"] = code
        keep.append(sub)
    if not keep:
        return pd.DataFrame()
    rice = pd.concat(keep, ignore_index=True)
    rice = rice.dropna(subset=["year", "area_ha", "yield_tha"])
    rice = rice[(rice["area_ha"] > 0) & (rice["yield_tha"] > 0.2) & (rice["yield_tha"] < 8)]

    rows = []
    for (code, year), g in rice.groupby(["state_code", "year"]):
        w = g["area_ha"].to_numpy(dtype=float)
        y = g["yield_tha"].to_numpy(dtype=float)
        m = np.isfinite(w) & np.isfinite(y) & (w > 0)
        if m.sum() < 3:
            continue
        rows.append(
            {
                "iso3": "IND",
                "unit_id": f"IND-{code}",
                "state_code": code,
                "crop": "rice",
                "year": int(year),
                "season": "kharif",
                "yield_tha": float(np.average(y[m], weights=w[m])),
                "harvested_ha": float(w[m].sum()),
            }
        )
    out = pd.DataFrame(rows)
    # Drop units with too few years for rolling-origin tests.
    counts = out.groupby("unit_id")["year"].nunique()
    keep_ids = set(counts[counts >= 12].index)
    out = out[out["unit_id"].isin(keep_ids)].copy()
    cache = root / "data_interim" / "india_kharif_rice.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out


def _state_polygons(root: Path) -> dict[str, object]:
    import json
    from shapely.geometry import shape

    path = root / "data_raw" / "india_apy" / "repo" / "data" / "states_india.geojson"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    wanted = {_norm_state(k): v for k, v in HEADLINE_STATES.items()}
    # geojson uses 'Arunanchal' etc.; match after normalize
    alias = {
        "orissa": "odisha",
        "nct of delhi": "delhi",
    }
    out = {}
    for feat in raw.get("features", []):
        name = _norm_state(feat.get("properties", {}).get("st_nm", ""))
        name = alias.get(name, name)
        code = wanted.get(name)
        if not code:
            continue
        out[code] = shape(feat["geometry"])
    return out


def india_rice_points_from_cropgrids(root: Path, n_per_state: int = 10, min_ha: float = 20.0) -> pd.DataFrame:
    import xarray as xr

    path = nc_path(root, "rice")
    if not path.exists():
        return pd.DataFrame()
    polys = _state_polygons(root)
    if not polys:
        return pd.DataFrame()
    ds = xr.open_dataset(path)
    ha = ds["harvarea"]
    rows = []
    for code, geom in polys.items():
        minx, miny, maxx, maxy = geom.bounds
        if float(ha.lat[0]) > float(ha.lat[-1]):
            sub = ha.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))
        else:
            sub = ha.sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
        vals = np.asarray(sub.values, dtype=float)
        lats = sub["lat"].values
        lons = sub["lon"].values
        if vals.size == 0:
            continue
        vals[~np.isfinite(vals)] = 0
        from affine import Affine
        from rasterio.features import geometry_mask

        res_x = float(lons[1] - lons[0]) if len(lons) > 1 else 0.05
        res_y = float(lats[1] - lats[0]) if len(lats) > 1 else 0.05
        transform = Affine.translation(float(lons[0]) - res_x / 2, float(lats[0]) - res_y / 2) * Affine.scale(res_x, res_y)
        inside = geometry_mask([geom.__geo_interface__], out_shape=vals.shape, transform=transform, invert=True)
        vals = np.where(inside, vals, 0)
        flat = vals.ravel()
        if flat.max() <= 0:
            continue
        k = min(n_per_state, int((flat >= min_ha).sum()) or int((flat > 0).sum()) or 1)
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
                    "iso3": "IND",
                    "unit_id": f"IND-{code}",
                    "state_code": code,
                    "crop": "rice",
                    "lon": float(lons[c]),
                    "lat": float(lats[r]),
                    "weight": w,
                    "source": "cropgrids2020",
                }
            )
    ds.close()
    out = pd.DataFrame(rows)
    cache = root / "data_interim" / "india_rice_points.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out


def _tokens(name: str) -> frozenset[str]:
    text = _norm_state(name)
    for junk in ("district", "zila", "zilla"):
        text = text.replace(junk, " ")
    text = text.replace("-", " ").replace(".", " ")
    drop = {"the", "and", "of"}
    return frozenset(t for t in text.split() if t and t not in drop)


def _name_key(name: str) -> str:
    return " ".join(sorted(_tokens(name)))


DISTRICT_ALIASES = {
    "medinipur west": "paschim medinipur",
    "medinipur east": "purba medinipur",
    "24 paraganas south": "south 24 parganas",
    "24 paraganas north": "north 24 parganas",
    "south twenty four parganas": "south 24 parganas",
    "north twenty four parganas": "north 24 parganas",
    "purba bardhaman": "purba bardhaman",
    "bardhaman": "purba bardhaman",
    "barddhaman": "purba bardhaman",
    "hooghly": "hugli",
    "hugli": "hugli",
    "howrah": "haora",
    "jalpaiguri": "jalpaiguri",
    "west godavari": "west godavari",
    "east godavari": "east godavari",
    "janjgir-champa": "janjgir champa",
    "janjgir champa": "janjgir champa",
}


def _district_polygons(root: Path) -> dict[tuple[str, str], object]:
    """(state_code, name_key) -> polygon, using ADM2 clipped to headline states."""

    import json
    from shapely.geometry import shape

    path = root / "data_raw" / "india_apy" / "IND_ADM2_simplified.geojson"
    if not path.exists():
        return {}
    states = _state_polygons(root)
    if not states:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for feat in raw.get("features", []):
        geom = shape(feat["geometry"])
        if geom.is_empty:
            continue
        c = geom.centroid
        code = next((sc for sc, sg in states.items() if sg.contains(c) or sg.intersects(c)), None)
        if not code:
            continue
        raw_name = str(feat.get("properties", {}).get("shapeName", ""))
        alias = DISTRICT_ALIASES.get(_norm_state(raw_name), raw_name)
        key = _name_key(alias)
        out[(code, key)] = geom
        out.setdefault((code, _name_key(raw_name)), geom)
    return out


def _best_district_geom(polys: dict, state_code: str, district: str):
    alias = DISTRICT_ALIASES.get(_norm_state(district), district)
    key = _name_key(alias)
    hit = polys.get((state_code, key))
    if hit is not None:
        return hit, key
    tokens = _tokens(alias)
    best, best_j = None, 0.0
    for (sc, pk), geom in polys.items():
        if sc != state_code:
            continue
        other = set(pk.split())
        j = len(tokens & other) / max(len(tokens | other), 1)
        if j > best_j:
            best, best_j = geom, j
    if best_j >= 0.6:
        return best, key
    return None, key


def parse_india_district_rice(root: Path, year_start: int = 2001, year_end: int = 2019, top_n: int = 80) -> pd.DataFrame:
    """Largest monsoon-rice districts with a long enough yield record."""

    path = _find_apy_csv(root)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def col(*names: str) -> str | None:
        for name in names:
            if name in cols:
                return cols[name]
        return None

    state_c, crop_c, year_c = col("state", "state_name"), col("crop"), col("year", "crop_year")
    season_c, area_c, yield_c, dist_c = col("season"), col("area"), col("yield"), col("district", "district_name")
    if not all([state_c, crop_c, year_c, season_c, area_c, yield_c, dist_c]):
        return pd.DataFrame()

    rice = df[df[crop_c].astype(str).str.strip().str.lower().eq("rice")].copy()
    rice["state_key"] = rice[state_c].map(_norm_state)
    rice["season_key"] = rice[season_c].astype(str).str.strip().str.lower()
    rice["year"] = rice[year_c].map(_year_from_apy)
    rice["area_ha"] = pd.to_numeric(rice[area_c], errors="coerce")
    rice["yield_tha"] = pd.to_numeric(rice[yield_c], errors="coerce")
    rice["district"] = rice[dist_c].astype(str).str.strip()
    rice = rice[rice["state_key"].isin(HEADLINE_STATES) & rice["year"].between(year_start, year_end)]
    rice = rice.dropna(subset=["year", "area_ha", "yield_tha", "district"])
    rice = rice[(rice["area_ha"] > 500) & (rice["yield_tha"].between(0.5, 7))]

    keep = []
    for state_key, code in HEADLINE_STATES.items():
        season = MAIN_MONSOON_SEASON.get(state_key, "kharif")
        sub = rice[(rice["state_key"] == state_key) & (rice["season_key"] == season)].copy()
        if sub.empty:
            continue
        sub["state_code"] = code
        keep.append(sub)
    if not keep:
        return pd.DataFrame()
    rice = pd.concat(keep, ignore_index=True)
    stats = (
        rice.groupby(["state_code", "district"], as_index=False)
        .agg(n=("year", "nunique"), area=("area_ha", "mean"))
    )
    stats = stats[stats["n"] >= 12].sort_values("area", ascending=False).head(top_n)
    wanted = set(zip(stats["state_code"], stats["district"]))
    rice = rice[rice.apply(lambda r: (r["state_code"], r["district"]) in wanted, axis=1)]

    rows = []
    for (code, district, year), g in rice.groupby(["state_code", "district", "year"]):
        slug = "".join(ch for ch in _norm_state(district) if ch.isalnum() or ch == " ").replace(" ", "")[:18].upper()
        rows.append(
            {
                "iso3": "IND",
                "unit_id": f"IND-{code}-{slug}",
                "state_code": code,
                "district": district,
                "crop": "rice",
                "year": int(year),
                "season": "kharif",
                "yield_tha": float(g["yield_tha"].mean()),
                "harvested_ha": float(g["area_ha"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    cache = root / "data_interim" / "india_kharif_rice_district.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out


def load_india_kharif_rice(root: Path) -> pd.DataFrame:
    district = root / "data_interim" / "india_kharif_rice_district.csv"
    if district.exists() and district.stat().st_size > 50:
        return pd.read_csv(district)
    cache = root / "data_interim" / "india_kharif_rice.csv"
    if cache.exists() and cache.stat().st_size > 50:
        return pd.read_csv(cache)
    return parse_india_kharif_rice(root)


def india_rice_district_points(root: Path, labels: pd.DataFrame | None = None, n_per: int = 4, min_ha: float = 15.0) -> pd.DataFrame:
    import xarray as xr

    if labels is None:
        labels = parse_india_district_rice(root)
    if labels.empty:
        return pd.DataFrame()
    path = nc_path(root, "rice")
    if not path.exists():
        return pd.DataFrame()
    polys = _district_polygons(root)
    if not polys:
        return pd.DataFrame()
    ds = xr.open_dataset(path)
    ha = ds["harvarea"]
    rows = []
    units = labels[["unit_id", "state_code", "district"]].drop_duplicates()
    matched = 0
    for rec in units.itertuples(index=False):
        geom, _ = _best_district_geom(polys, rec.state_code, rec.district)
        if geom is None:
            continue
        matched += 1
        minx, miny, maxx, maxy = geom.bounds
        if float(ha.lat[0]) > float(ha.lat[-1]):
            sub = ha.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))
        else:
            sub = ha.sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
        vals = np.asarray(sub.values, dtype=float)
        lats = sub["lat"].values
        lons = sub["lon"].values
        if vals.size == 0:
            continue
        vals[~np.isfinite(vals)] = 0
        from affine import Affine
        from rasterio.features import geometry_mask

        res_x = float(lons[1] - lons[0]) if len(lons) > 1 else 0.05
        res_y = float(lats[1] - lats[0]) if len(lats) > 1 else 0.05
        transform = Affine.translation(float(lons[0]) - res_x / 2, float(lats[0]) - res_y / 2) * Affine.scale(res_x, res_y)
        inside = geometry_mask([geom.__geo_interface__], out_shape=vals.shape, transform=transform, invert=True)
        vals = np.where(inside, vals, 0)
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
                    "iso3": "IND",
                    "unit_id": rec.unit_id,
                    "state_code": rec.state_code,
                    "district": rec.district,
                    "crop": "rice",
                    "lon": float(lons[c]),
                    "lat": float(lats[r]),
                    "weight": w,
                    "source": "cropgrids2020",
                }
            )
    ds.close()
    print(f"district geom matched {matched}/{len(units)}", flush=True)
    out = pd.DataFrame(rows)
    cache = root / "data_interim" / "india_rice_points.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out
