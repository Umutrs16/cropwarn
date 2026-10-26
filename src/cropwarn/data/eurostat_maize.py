"""Eurostat NUTS2 grain-maize yields (apro_cpshr).

2003 / 2018 / 2022 European heat-droughts hit a spatially stable
cross-section (unlike US county anomalies). Yield is production / area
in EU standard humidity (t/ha). FAO national maize is not used.
"""

from __future__ import annotations

from pathlib import Path
import json
import urllib.request

import numpy as np
import pandas as pd

from cropwarn.data.cropgrids import nc_path

EU_MAIZE_CALENDAR = (120, 275)

PREFIX_ISO3 = {
    "FR": "FRA",
    "DE": "DEU",
    "RO": "ROU",
    "HU": "HUN",
    "PL": "POL",
    "IT": "ITA",
    "ES": "ESP",
}

SKIP_PREFIX = ("FRY", "ES7", "PT2", "PT3", "FR9")

STAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/apro_cpshr"
    "?crops=C1500&strucpro={struc}"
)
NUTS_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326_LEVL_2.geojson"
NUTS1_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326_LEVL_1.geojson"

ISO3_META = {
    "FRA": {"region": "Europe", "latitude_band": "temperate", "exporter": True, "multi_crop": False, "name": "France"},
    "DEU": {"region": "Europe", "latitude_band": "temperate", "exporter": True, "multi_crop": False, "name": "Germany"},
    "ROU": {"region": "Europe", "latitude_band": "temperate", "exporter": True, "multi_crop": False, "name": "Romania"},
    "HUN": {"region": "Europe", "latitude_band": "temperate", "exporter": True, "multi_crop": False, "name": "Hungary"},
    "POL": {"region": "Europe", "latitude_band": "temperate", "exporter": False, "multi_crop": False, "name": "Poland"},
    "ITA": {"region": "Europe", "latitude_band": "subtropical", "exporter": False, "multi_crop": False, "name": "Italy"},
    "ESP": {"region": "Europe", "latitude_band": "subtropical", "exporter": False, "multi_crop": False, "name": "Spain"},
}


def _fetch(url: str, dest: Path, timeout: int = 180) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "cropwarn/nature-food"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    tmp.replace(dest)
    return dest


def parse_jsonstat_geo_time(path: Path, value_name: str) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    geo_i = raw["dimension"]["geo"]["category"]["index"]
    time_i = raw["dimension"]["time"]["category"]["index"]
    geo_lab = raw["dimension"]["geo"]["category"]["label"]
    geo_by = {int(v): k for k, v in geo_i.items()}
    time_by = {int(v): int(k) for k, v in time_i.items()}
    n_time = len(time_i)
    rows = []
    for key, val in (raw.get("value") or {}).items():
        i = int(key)
        gidx, tidx = divmod(i, n_time)
        geo = geo_by.get(gidx)
        year = time_by.get(tidx)
        if geo is None or year is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        rows.append({"geo": geo, "name": geo_lab.get(geo, geo), "year": year, value_name: v})
    return pd.DataFrame(rows)


def _is_region(geo: str) -> bool:
    if not geo[:2].isalpha() or geo[:2] not in PREFIX_ISO3:
        return False
    if geo.startswith(SKIP_PREFIX):
        return False
    # Germany publishes maize only at NUTS1 (DE1..DEG).
    if geo.startswith("DE"):
        return len(geo) == 3
    return len(geo) == 4


def parse_eu_maize(root: Path, n_per_country: int = 5, min_years: int = 12, min_area: float = 8.0) -> pd.DataFrame:
    d = root / "data_raw" / "eurostat"
    ar_path = _fetch(STAT_URL.format(struc="AR_THS_HA"), d / "c1500_ar.json")
    pr_path = _fetch(STAT_URL.format(struc="HPRD_HUMD_EU_THS_T"), d / "c1500_pr.json")
    ar = parse_jsonstat_geo_time(ar_path, "area_kha")
    pr = parse_jsonstat_geo_time(pr_path, "prod_kt")
    if ar.empty or pr.empty:
        return pd.DataFrame()
    raw = ar.merge(pr, on=["geo", "name", "year"], how="inner")
    raw = raw[raw["geo"].map(_is_region) & raw["area_kha"].gt(0)].copy()
    raw["yield_tha"] = raw["prod_kt"] / raw["area_kha"]
    raw = raw[raw["yield_tha"].between(1.5, 20)].copy()
    raw["iso3"] = raw["geo"].str[:2].map(PREFIX_ISO3)
    raw["unit_id"] = raw["iso3"] + "-" + raw["geo"]
    raw["crop"] = "maize"
    raw["harvested_ha"] = raw["area_kha"] * 1000.0
    raw["source"] = "eurostat_apro_cpshr"
    raw["state_code"] = raw["geo"]
    area = raw.groupby("unit_id")["harvested_ha"].mean()
    nyr = raw.groupby("unit_id")["year"].nunique()
    keep = []
    stats = pd.DataFrame({"area": area, "n": nyr}).reset_index()
    stats["iso3"] = stats["unit_id"].str.split("-").str[0]
    stats = stats[stats["n"].ge(min_years) & stats["area"].ge(min_area * 1000.0)]
    for _, g in stats.groupby("iso3", sort=False):
        keep.extend(g.sort_values("area", ascending=False)["unit_id"].head(n_per_country).tolist())
    out = raw[raw["unit_id"].isin(keep)].copy()
    cache = root / "data_interim" / "eu_maize.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def load_eu_maize(root: Path) -> pd.DataFrame:
    path = root / "data_interim" / "eu_maize.csv"
    if not path.exists() or path.stat().st_size < 200:
        return pd.DataFrame()
    return pd.read_csv(path)


def _nuts_polygons(root: Path) -> dict:
    from shapely.geometry import shape

    out = {}
    for url, name in ((NUTS_URL, "NUTS_RG_20M_2021_4326_LEVL_2.geojson"), (NUTS1_URL, "NUTS_RG_20M_2021_4326_LEVL_1.geojson")):
        path = root / "data_raw" / "eurostat" / name
        if not path.exists() or path.stat().st_size < 1000:
            _fetch(url, path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        for feat in raw.get("features", []):
            nid = (feat.get("properties") or {}).get("NUTS_ID")
            if nid:
                out[nid] = shape(feat["geometry"])
    return out


def eu_maize_points(root: Path, n_per: int = 4, min_ha: float = 20.0) -> pd.DataFrame:
    import xarray as xr
    from affine import Affine
    from rasterio.features import geometry_mask

    labels = load_eu_maize(root)
    if labels.empty:
        return pd.DataFrame()
    path = nc_path(root, "maize")
    if not path.exists():
        return pd.DataFrame()
    polys = _nuts_polygons(root)
    ds = xr.open_dataset(path)
    ha = ds["harvarea"]
    lat_desc = float(ha.lat[0]) > float(ha.lat[-1])
    rows = []
    units = labels[["unit_id", "iso3", "geo", "state_code"]].drop_duplicates()
    for rec in units.itertuples(index=False):
        geom = polys.get(rec.geo)
        if geom is None:
            continue
        minx, miny, maxx, maxy = geom.bounds
        if lat_desc:
            sub = ha.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))
        else:
            sub = ha.sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
        vals = np.asarray(sub.values, dtype=float)
        lats = sub["lat"].values
        lons = sub["lon"].values
        if vals.size == 0:
            continue
        vals[~np.isfinite(vals)] = 0
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
                    "iso3": rec.iso3,
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
    cache = root / "data_interim" / "eu_maize_points.csv"
    if len(out):
        out.to_csv(cache, index=False)
    return out
