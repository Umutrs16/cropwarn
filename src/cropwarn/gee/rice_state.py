"""India state rice extract. Separate cache so national rice_YYYY.csv stays intact."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cropwarn.constants import GEE_PROJECT
from cropwarn.gee.export import init_ee


def _points_fc(ee, pts: pd.DataFrame):
    feats = []
    for i, r in enumerate(pts.itertuples(index=False)):
        feats.append(
            ee.Feature(
                ee.Geometry.Point([float(r.lon), float(r.lat)]),
                {
                    "iso3": r.iso3,
                    "unit_id": getattr(r, "unit_id", r.iso3),
                    "crop": r.crop,
                    "weight": float(r.weight),
                    "pid": i,
                },
            )
        )
    return ee.FeatureCollection(feats)


def _parse_unit_table(info: dict, year: int, kind: str) -> pd.DataFrame:
    from cropwarn.gee.crop_extract import np_average

    rows = []
    for feat in info.get("features", []):
        p = feat.get("properties", {})
        iso3, crop, w = p.get("iso3"), p.get("crop"), p.get("weight", 1.0)
        unit_id = p.get("unit_id") or iso3
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
                rows.append({"iso3": iso3, "unit_id": unit_id, "crop": crop, "date": date, "weight": w, var: val})
        else:
            for month in range(1, 13):
                row = {
                    "iso3": iso3,
                    "unit_id": unit_id,
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
    keys = ["iso3", "unit_id", "crop", "date"]
    vars_ = [c for c in raw.columns if c not in keys + ["weight"]]
    pieces = []
    for key, g in raw.groupby(keys, sort=False):
        rec = dict(zip(keys, key))
        rec["year"] = rec["date"].year
        ww = g["weight"].to_numpy(dtype=float)
        ww = ww if ww.sum() > 0 else None
        for v in vars_:
            rec[v] = float(np_average(g[v], ww))
        pieces.append(rec)
    return pd.DataFrame(pieces)


def _cache_dir(root: Path, folder: str = "gee_rice_state") -> Path:
    d = root / "data_raw" / folder
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_rice_state_year(
    ee, year: int, pts: pd.DataFrame, root: Path, folder: str = "gee_rice_state", prefix: str = "rice_state"
) -> pd.DataFrame:
    cache = _cache_dir(root, folder) / f"{prefix}_{year}.csv"
    if cache.exists() and cache.stat().st_size > 200:
        return pd.read_csv(cache, parse_dates=["date"])
    if pts.empty:
        return pd.DataFrame()
    fc = _points_fc(ee, pts)
    parts = []
    for name, builder in (
        ("optical16", lambda: _parse_unit_table(_optical_info(ee, year, fc), year, "optical16")),
        ("era5", lambda: _parse_unit_table(_era5_info(ee, year, fc), year, "era5")),
        ("smap", lambda: _parse_unit_table(_smap_info(ee, year, fc), year, "smap")),
    ):
        try:
            part = builder()
            if len(part):
                parts.append(part)
                print(f"rice-state {year} {name} rows={len(part)}", flush=True)
        except Exception as exc:
            print(f"SKIP rice-state {year} {name}: {exc}", flush=True)
    if not parts:
        return pd.DataFrame()
    out = parts[0].copy()
    out["date"] = pd.to_datetime(out["date"])
    for extra in parts[1:]:
        extra = extra.copy()
        extra["date"] = pd.to_datetime(extra["date"])
        cols = [c for c in extra.columns if c not in {"iso3", "unit_id", "crop", "date", "year"}]
        merged = []
        for (unit_id, crop), g in out.groupby(["unit_id", "crop"], sort=False):
            r = extra[(extra["unit_id"] == unit_id) & (extra["crop"] == crop)][["date", *cols]].sort_values("date")
            if r.empty:
                merged.append(g)
                continue
            merged.append(pd.merge_asof(g.sort_values("date"), r, on="date", direction="nearest", tolerance=pd.Timedelta("20D")))
        out = pd.concat(merged, ignore_index=True)
    out["year"] = pd.to_datetime(out["date"]).dt.year
    out.to_csv(cache, index=False)
    return out


def _optical_info(ee, year, fc):
    from cropwarn.gee.crop_extract import extract_crop_optical16
    from cropwarn.gee.assets import GEE_ASSETS

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
        return {"features": []}
    stack = ee.Image.cat(images)
    return stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=500, tileScale=2).getInfo()


def _era5_info(ee, year, fc):
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
    return ee.Image.cat(clean).reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=11000, tileScale=2).getInfo()


def _smap_info(ee, year, fc):
    from cropwarn.gee.assets import GEE_ASSETS

    if year < 2015:
        return {"features": []}
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
    return ee.Image.cat(clean).reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=9000, tileScale=2).getInfo()


def extract_all_rice_state(
    root: Path,
    years: list[int],
    pts: pd.DataFrame,
    folder: str = "gee_rice_state",
    prefix: str = "rice_state",
) -> pd.DataFrame:
    ee = init_ee(GEE_PROJECT)
    for year in years:
        extract_rice_state_year(ee, year, pts, root, folder=folder, prefix=prefix)
    d = _cache_dir(root, folder)
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in sorted(d.glob(f"{prefix}_20*.csv"))]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(d / f"{prefix.replace('_state', '')}_series.csv" if prefix != "rice_state" else d / "rice_state_series.csv", index=False)
    if prefix == "rice_district":
        out.to_csv(d / "rice_district_series.csv", index=False)
    return out


def load_china_early_series(root: Path) -> pd.DataFrame:
    path = root / "data_raw" / "gee_china_early_rice" / "china_early_series.csv"
    if path.exists() and path.stat().st_size > 200:
        return pd.read_csv(path, parse_dates=["date"])
    d = root / "data_raw" / "gee_china_early_rice"
    if not d.exists():
        return pd.DataFrame()
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in sorted(d.glob("china_early_20*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_eu_maize_series(root: Path) -> pd.DataFrame:
    path = root / "data_raw" / "gee_eu_maize" / "eu_maize_series.csv"
    if path.exists() and path.stat().st_size > 200:
        return pd.read_csv(path, parse_dates=["date"])
    d = root / "data_raw" / "gee_eu_maize"
    if not d.exists():
        return pd.DataFrame()
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in sorted(d.glob("eu_maize_20*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_usa_maize_series(root: Path) -> pd.DataFrame:
    path = root / "data_raw" / "gee_usa_maize" / "usa_maize_series.csv"
    if path.exists() and path.stat().st_size > 200:
        return pd.read_csv(path, parse_dates=["date"])
    d = root / "data_raw" / "gee_usa_maize"
    if not d.exists():
        return pd.DataFrame()
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in sorted(d.glob("usa_maize_20*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_rice_state_series(root: Path) -> pd.DataFrame:
    district = root / "data_raw" / "gee_rice_district" / "rice_district_series.csv"
    if district.exists() and district.stat().st_size > 200:
        return pd.read_csv(district, parse_dates=["date"])
    path = root / "data_raw" / "gee_rice_state" / "rice_state_series.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    d = root / "data_raw" / "gee_rice_state"
    if not d.exists():
        return pd.DataFrame()
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in sorted(d.glob("rice_state_20*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
