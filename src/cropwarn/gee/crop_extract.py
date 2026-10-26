"""Crop-specific national series at SPAM harvested-area sample points.

Uses point reduceRegions instead of country polygons so GEE stays inside
memory limits. Optical is 16-day (MOD13Q1); climate/SM remain monthly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cropwarn.constants import GEE_PROJECT
from cropwarn.gee.assets import GEE_ASSETS
from cropwarn.gee.export import init_ee
from cropwarn.gee.national import country_feature_collection, study_iso3s

SCALE_OPT = 500
SCALE_CLIM = 11000


def country_bounds_table(root: Path) -> pd.DataFrame:
    cache = root / "data_interim" / "country_bounds.csv"
    if cache.exists() and cache.stat().st_size > 50:
        return pd.read_csv(cache)
    ee = init_ee(GEE_PROJECT)
    fc = country_feature_collection(ee, study_iso3s())

    def add_bounds(feat):
        ring = ee.List(feat.geometry().bounds().coordinates().get(0))
        xs = ee.List([ee.List(ring.get(0)).get(0), ee.List(ring.get(1)).get(0), ee.List(ring.get(2)).get(0)])
        ys = ee.List([ee.List(ring.get(0)).get(1), ee.List(ring.get(1)).get(1), ee.List(ring.get(2)).get(1)])
        return ee.Feature(
            None,
            {
                "iso3": feat.get("iso3"),
                "west": xs.reduce(ee.Reducer.min()),
                "east": xs.reduce(ee.Reducer.max()),
                "south": ys.reduce(ee.Reducer.min()),
                "north": ys.reduce(ee.Reducer.max()),
            },
        )

    info = fc.map(add_bounds).getInfo()
    rows = [f["properties"] for f in info.get("features", [])]
    out = pd.DataFrame(rows)
    out = out.groupby("iso3", as_index=False).agg(
        west=("west", "min"), east=("east", "max"), south=("south", "min"), north=("north", "max")
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def country_polygons(root: Path) -> dict:
    """Simplified LSIB polygons keyed by ISO3, cached as GeoJSON."""

    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union

    cache = root / "data_interim" / "country_polygons.geojson"
    if cache.exists() and cache.stat().st_size > 500:
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {k: shape(v) for k, v in raw.items()}
    ee = init_ee(GEE_PROJECT)
    fc = country_feature_collection(ee, study_iso3s())
    info = fc.getInfo()
    buckets: dict[str, list] = {}
    for feat in info.get("features", []):
        iso3 = feat.get("properties", {}).get("iso3")
        geom = feat.get("geometry")
        if iso3 and geom:
            buckets.setdefault(iso3, []).append(shape(geom))
    out = {iso: unary_union(gs) for iso, gs in buckets.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({k: v.__geo_interface__ for k, v in out.items()}), encoding="utf-8")
    return out


def _points_fc(ee, pts: pd.DataFrame):
    feats = []
    for i, r in enumerate(pts.itertuples(index=False)):
        feats.append(
            ee.Feature(
                ee.Geometry.Point([float(r.lon), float(r.lat)]),
                {"iso3": r.iso3, "crop": r.crop, "weight": float(r.weight), "pid": i},
            )
        )
    return ee.FeatureCollection(feats)


def _parse_point_table(info: dict, year: int, kind: str) -> pd.DataFrame:
    rows = []
    for feat in info.get("features", []):
        p = feat.get("properties", {})
        iso3, crop, w = p.get("iso3"), p.get("crop"), p.get("weight", 1.0)
        if not iso3:
            continue
        if kind == "optical16":
            for key, val in p.items():
                if not key.startswith("evi_") and not key.startswith("ndvi_") and not key.startswith("lst_"):
                    continue
                if key.endswith("_stdDev"):
                    continue
                name = key[:-5] if key.endswith("_mean") else key
                parts = name.split("_")
                if len(parts) < 2:
                    continue
                doy = int(parts[1])
                var = parts[0]
                date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
                rows.append({"iso3": iso3, "crop": crop, "date": date, "weight": w, var: val})
        else:
            for month in range(1, 13):
                row = {
                    "iso3": iso3,
                    "crop": crop,
                    "date": pd.Timestamp(year=year, month=month, day=15),
                    "weight": w,
                }
                prefix = f"m{month:02d}_"
                for key, val in p.items():
                    if not key.startswith(prefix):
                        continue
                    name = key[len(prefix) :]
                    if name.endswith("_mean"):
                        name = name[: -len("_mean")]
                    if name.endswith("_stdDev"):
                        continue
                    row[name] = val
                rows.append(row)
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    keys = ["iso3", "crop", "date"]
    vars_ = [c for c in raw.columns if c not in keys + ["weight"]]
    pieces = []
    for (iso3, crop, date), g in raw.groupby(keys, sort=False):
        rec = {"iso3": iso3, "crop": crop, "date": date, "year": date.year}
        ww = g["weight"].to_numpy(dtype=float)
        ww = ww if ww.sum() > 0 else None
        for v in vars_:
            rec[v] = float(np_average(g[v], ww))
        pieces.append(rec)
    return pd.DataFrame(pieces)


def np_average(series, weights):
    import numpy as np

    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if weights is None:
        return float(np.nanmean(x))
    m = np.isfinite(x) & np.isfinite(weights)
    if not m.any():
        return float("nan")
    return float(np.average(x[m], weights=weights[m]))


def extract_crop_optical16(ee, year: int, pts_fc) -> pd.DataFrame:
    raw = (
        ee.ImageCollection(GEE_ASSETS["mod13q1"])
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.notNull(["system:time_start"]))
    )
    lst_raw = (
        ee.ImageCollection(GEE_ASSETS["mod11a2"])
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.notNull(["system:time_start"]))
    )
    n = int(raw.size().getInfo() or 0)
    n_lst = int(lst_raw.size().getInfo() or 0)
    col_list = raw.toList(n or 1)
    lst_list = lst_raw.toList(n_lst or 1)
    images = []
    for i in range(n):
        img = ee.Image(col_list.get(i))
        doy = ee.Number(img.date().getRelative("day", "year")).add(1).format("%03d")
        scaled = img.select(["NDVI", "EVI"]).multiply(0.0001).updateMask(img.select("SummaryQA").lte(1))
        images.append(scaled.select("EVI").rename(ee.String("evi_").cat(doy)))
        images.append(scaled.select("NDVI").rename(ee.String("ndvi_").cat(doy)))
    for i in range(min(n_lst, 46)):
        img = ee.Image(lst_list.get(i))
        doy = ee.Number(img.date().getRelative("day", "year")).add(1).format("%03d")
        lst = img.select("LST_Day_1km").multiply(0.02).subtract(273.15).updateMask(img.select("QC_Day").bitwiseAnd(3).eq(0))
        images.append(lst.rename(ee.String("lst_").cat(doy)))
    if not images:
        return pd.DataFrame()
    stack = ee.Image.cat(images)
    table = stack.reduceRegions(collection=pts_fc, reducer=ee.Reducer.mean(), scale=SCALE_OPT, tileScale=2)
    return _parse_point_table(table.getInfo(), year, "optical16")


def extract_crop_era5(ee, year: int, pts_fc) -> pd.DataFrame:
    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(
            [
                "temperature_2m_max",
                "dewpoint_temperature_2m",
                "total_precipitation_sum",
                "potential_evaporation_sum",
                "volumetric_soil_water_layer_3",
            ]
        )
    )
    ln10 = 2.302585092994046
    clean = []
    for month in range(1, 13):
        start = ee.Date.fromYMD(year, month, 1)
        raw = col.filterDate(start, start.advance(1, "month"))
        img = ee.Image(ee.Algorithms.If(raw.size().gt(0), raw.first(), ee.Image.constant(0).updateMask(0)))
        t2m = img.select("temperature_2m_max").subtract(273.15)
        tdew = img.select("dewpoint_temperature_2m").subtract(273.15)
        es = t2m.multiply(7.5).divide(t2m.add(237.3)).multiply(ln10).exp().multiply(0.6108)
        ea = tdew.multiply(7.5).divide(tdew.add(237.3)).multiply(ln10).exp().multiply(0.6108)
        clean.extend(
            [
                t2m.rename(f"m{month:02d}_t2m"),
                img.select("total_precipitation_sum").multiply(1000).rename(f"m{month:02d}_precip"),
                img.select("potential_evaporation_sum").abs().multiply(1000).rename(f"m{month:02d}_et0"),
                img.select("volumetric_soil_water_layer_3").rename(f"m{month:02d}_swvl"),
                es.subtract(ea).max(0).rename(f"m{month:02d}_vpd"),
            ]
        )
    table = ee.Image.cat(clean).reduceRegions(collection=pts_fc, reducer=ee.Reducer.mean(), scale=SCALE_CLIM, tileScale=2)
    return _parse_point_table(table.getInfo(), year, "era5")


def extract_crop_smap(ee, year: int, pts_fc) -> pd.DataFrame:
    if year < 2015:
        return pd.DataFrame()
    l4 = ee.ImageCollection(GEE_ASSETS["smap_l4"]).select("sm_rootzone")
    l3 = ee.ImageCollection(GEE_ASSETS.get("smap_l3", "NASA/SMAP/SPL3SMP_E/005")).select("vegetation_water_content_am")
    clean = []
    for month in range(1, 13):
        if year == 2015 and month < 4:
            continue
        start = ee.Date.fromYMD(year, month, 15)
        rz = l4.filterDate(start, start.advance(2, "day")).first()
        rz = ee.Image(ee.Algorithms.If(rz, rz, l4.filterDate(start.advance(-5, "day"), start.advance(6, "day")).first()))
        clean.append(ee.Image(rz).rename(f"m{month:02d}_smap_rzsm"))
        vw = l3.filterDate(ee.Date.fromYMD(year, month, 1), ee.Date.fromYMD(year, month, 1).advance(1, "month")).mean()
        clean.append(ee.Image(ee.Algorithms.If(vw, vw.rename(f"m{month:02d}_vod"), ee.Image.constant(0).updateMask(0).rename(f"m{month:02d}_vod"))))
    table = ee.Image.cat(clean).reduceRegions(collection=pts_fc, reducer=ee.Reducer.mean(), scale=9000, tileScale=2)
    return _parse_point_table(table.getInfo(), year, "smap")


def _cache_dir(root: Path) -> Path:
    d = root / "data_raw" / "gee_crop"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_crop_year(ee, year: int, crop: str, pts: pd.DataFrame, root: Path) -> pd.DataFrame:
    cache = _cache_dir(root) / f"{crop}_{year}.csv"
    if cache.exists() and cache.stat().st_size > 200:
        return pd.read_csv(cache, parse_dates=["date"])
    sub = pts[pts["crop"] == crop]
    if sub.empty:
        return pd.DataFrame()
    fc = _points_fc(ee, sub)
    parts = []
    for name, fn in (("optical16", extract_crop_optical16), ("era5", extract_crop_era5), ("smap", extract_crop_smap)):
        try:
            part = fn(ee, year, fc)
            if len(part):
                parts.append(part)
                print(f"crop {crop} {year} {name} rows={len(part)}", flush=True)
        except Exception as exc:
            print(f"SKIP crop {crop} {year} {name}: {exc}", flush=True)
    if not parts:
        return pd.DataFrame()
    out = parts[0].copy()
    out["date"] = pd.to_datetime(out["date"])
    for extra in parts[1:]:
        extra = extra.copy()
        extra["date"] = pd.to_datetime(extra["date"])
        cols = [c for c in extra.columns if c not in {"iso3", "crop", "date", "year"}]
        merged = []
        for (iso3, crop), g in out.groupby(["iso3", "crop"], sort=False):
            r = extra[(extra["iso3"] == iso3) & (extra["crop"] == crop)][["date", *cols]].sort_values("date")
            if r.empty:
                merged.append(g)
                continue
            left = g.sort_values("date")
            merged.append(pd.merge_asof(left, r, on="date", direction="nearest", tolerance=pd.Timedelta("20D")))
        out = pd.concat(merged, ignore_index=True)
    out["year"] = pd.to_datetime(out["date"]).dt.year
    out.to_csv(cache, index=False)
    return out


def combine_crop_tables(root: Path) -> pd.DataFrame:
    d = _cache_dir(root)
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in sorted(d.glob("*_20*.csv"))]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(d / "crop_series.csv", index=False)
    return out


def extract_all_crop(root: Path, years: list[int], pts: pd.DataFrame, crops: list[str] | None = None) -> pd.DataFrame:
    ee = init_ee(GEE_PROJECT)
    crops = crops or sorted(pts["crop"].unique())
    for crop in crops:
        for year in years:
            extract_crop_year(ee, year, crop, pts, root)
    return combine_crop_tables(root)
