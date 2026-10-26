"""National cropland-weighted monthly means from Earth Engine.

Long-run mask is year-specific MCD12Q1 cropland (classes 12 and 14), not the
2021 WorldCereal map copied across 2001–2024. SMAP uses one mid-month L4
image so the 3-hourly collection does not exceed memory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cropwarn.constants import COUNTRIES, FAO_CODE_ALIASES, GEE_PROJECT, TYPICAL_AREA_MHA
from cropwarn.gee.assets import GEE_ASSETS
from cropwarn.gee.export import init_ee

GAUL = "FAO/GAUL_SIMPLIFIED_500m/2015/level0"
MCD12Q1 = "MODIS/061/MCD12Q1"
SMAP_L3 = "NASA/SMAP/SPL3SMP_E/005"
SCALE = 5000

LSIB_NAME = {
    "USA": "United States",
    "CAN": "Canada",
    "MEX": "Mexico",
    "BRA": "Brazil",
    "ARG": "Argentina",
    "CHL": "Chile",
    "FRA": "France",
    "DEU": "Germany",
    "UKR": "Ukraine",
    "RUS": "Russia",
    "ESP": "Spain",
    "ITA": "Italy",
    "POL": "Poland",
    "ROU": "Romania",
    "GBR": "United Kingdom",
    "TUR": "Turkey",
    "KAZ": "Kazakhstan",
    "CHN": "China",
    "IND": "India",
    "PAK": "Pakistan",
    "BGD": "Bangladesh",
    "IDN": "Indonesia",
    "VNM": "Vietnam",
    "THA": "Thailand",
    "PHL": "Philippines",
    "JPN": "Japan",
    "KOR": "Korea, South",
    "AUS": "Australia",
    "NZL": "New Zealand",
    "EGY": "Egypt",
    "NGA": "Nigeria",
    "ETH": "Ethiopia",
    "KEN": "Kenya",
    "ZAF": "South Africa",
    "TZA": "Tanzania",
    "GHA": "Ghana",
    "MAR": "Morocco",
    "SAU": "Saudi Arabia",
    "IRN": "Iran",
    "IRQ": "Iraq",
}


def study_iso3s() -> list[str]:
    return sorted({iso for crop in TYPICAL_AREA_MHA.values() for iso in crop})


def _cache_dir(root: Path) -> Path:
    d = root / "data_raw" / "gee_national"
    d.mkdir(parents=True, exist_ok=True)
    return d


def country_feature_collection(ee, iso3s: list[str]):
    """LSIB names are the primary geometry; GAUL codes do not always match FAO."""

    names = [LSIB_NAME[i] for i in iso3s if i in LSIB_NAME]
    name_map = ee.Dictionary({LSIB_NAME[i]: i for i in iso3s if i in LSIB_NAME})
    lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.inList("country_na", names))
    lsib = lsib.map(lambda f: f.set("iso3", name_map.get(f.get("country_na"))))
    have = set(lsib.aggregate_array("iso3").getInfo() or [])
    missing = [i for i in iso3s if i not in have]
    if missing:
        iso_of = {c.fao_code: c.iso3 for c in COUNTRIES}
        iso_of.update(FAO_CODE_ALIASES)
        codes = [c.fao_code for c in COUNTRIES if c.iso3 in missing]
        if "CHN" in missing:
            codes.append(41)
        mapping = ee.Dictionary({str(k): v for k, v in iso_of.items()})
        gaul = ee.FeatureCollection(GAUL).filter(ee.Filter.inList("ADM0_CODE", codes))
        gaul = gaul.map(lambda f: f.set("iso3", mapping.get(ee.Number(f.get("ADM0_CODE")).format("%d"))))
        lsib = lsib.merge(gaul.filter(ee.Filter.notNull(["iso3"])))
    return lsib


def cropland_mask(ee, year: int):
    y = int(min(max(year, 2001), 2023))
    lc = (
        ee.ImageCollection(MCD12Q1)
        .filter(ee.Filter.calendarRange(y, y, "year"))
        .first()
        .select("LC_Type1")
    )
    return lc.eq(12).Or(lc.eq(14)).selfMask().rename("crop")


def _safe_monthly_mean(ee, col, start, end, band: str):
    sub = col.filterDate(start, end)
    empty = ee.Image.constant(0).rename(band).updateMask(ee.Image.constant(0))
    return ee.Image(ee.Algorithms.If(sub.size().gt(0), sub.mean().select([band], [band]), empty))


def _reduce_monthly(ee, image, countries, year: int, scale: int = SCALE) -> pd.DataFrame:
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
    table = image.reduceRegions(
        collection=countries,
        reducer=reducer,
        scale=scale,
        tileScale=4,
    )
    info = table.getInfo()
    rows = []
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        iso3 = props.get("iso3")
        if not iso3:
            continue
        for month in range(1, 13):
            row = {"iso3": iso3, "year": year, "month": month, "date": f"{year}-{month:02d}-15"}
            prefix = f"m{month:02d}_"
            for key, val in props.items():
                if not key.startswith(prefix):
                    continue
                name = key[len(prefix) :]
                if name.endswith("_stdDev"):
                    row[name.replace("_stdDev", "_std")] = val
                elif name.endswith("_mean"):
                    row[name[: -len("_mean")]] = val
                else:
                    row[name] = val
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    num = [c for c in out.columns if c not in {"iso3", "year", "month", "date"}]
    return out.groupby(["iso3", "year", "month"], as_index=False).agg({c: "mean" for c in num} | {"date": "first"})


def extract_year_optical(ee, year: int, countries) -> pd.DataFrame:
    crop = cropland_mask(ee, year)
    col = (
        ee.ImageCollection(GEE_ASSETS["mod13q1"])
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .map(
            lambda img: img.select(["NDVI", "EVI"])
            .multiply(0.0001)
            .updateMask(img.select("SummaryQA").lte(1))
            .updateMask(crop)
            .copyProperties(img, ["system:time_start"])
        )
    )
    lst = (
        ee.ImageCollection(GEE_ASSETS["mod11a2"])
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .map(
            lambda img: img.select("LST_Day_1km")
            .multiply(0.02)
            .subtract(273.15)
            .updateMask(img.select("QC_Day").bitwiseAnd(3).eq(0))
            .updateMask(crop)
        )
    )
    gpp = (
        ee.ImageCollection("MODIS/061/MOD17A2H")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .map(
            lambda img: img.select("Gpp")
            .multiply(0.0001)
            .updateMask(img.select("Psn_QC").bitwiseAnd(1).eq(0))
            .updateMask(crop)
        )
    )
    images = []
    for month in range(1, 13):
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        evi = _safe_monthly_mean(ee, col, start, end, "EVI").rename(f"m{month:02d}_evi")
        ndvi = _safe_monthly_mean(ee, col, start, end, "NDVI").rename(f"m{month:02d}_ndvi")
        lst_m = _safe_monthly_mean(ee, lst, start, end, "LST_Day_1km")
        images.extend(
            [
                evi,
                ndvi,
                lst_m.rename(f"m{month:02d}_lst"),
                _safe_monthly_mean(ee, gpp, start, end, "Gpp").rename(f"m{month:02d}_gpp"),
                crop.unmask(0).rename(f"m{month:02d}_crop_frac"),
                lst_m.gt(32).rename(f"m{month:02d}_stress_area_frac"),
            ]
        )
    return _reduce_monthly(ee, ee.Image.cat(images), countries, year)


def extract_year_era5(ee, year: int, countries) -> pd.DataFrame:
    crop = cropland_mask(ee, year)
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
        img = ee.Image(
            ee.Algorithms.If(
                raw.size().gt(0),
                raw.first().updateMask(crop),
                ee.Image.constant([0, 0, 0, 0, 0]).rename(
                    [
                        "temperature_2m_max",
                        "dewpoint_temperature_2m",
                        "total_precipitation_sum",
                        "potential_evaporation_sum",
                        "volumetric_soil_water_layer_3",
                    ]
                ).updateMask(ee.Image.constant(0)),
            )
        )
        t2m = img.select("temperature_2m_max").subtract(273.15)
        tdew = img.select("dewpoint_temperature_2m").subtract(273.15)
        es = t2m.multiply(7.5).divide(t2m.add(237.3)).multiply(ln10).exp().multiply(0.6108)
        ea = tdew.multiply(7.5).divide(tdew.add(237.3)).multiply(ln10).exp().multiply(0.6108)
        vpd = es.subtract(ea).max(0)
        clean.extend(
            [
                t2m.rename(f"m{month:02d}_t2m"),
                img.select("total_precipitation_sum").multiply(1000).rename(f"m{month:02d}_precip"),
                img.select("potential_evaporation_sum").abs().multiply(1000).rename(f"m{month:02d}_et0"),
                img.select("volumetric_soil_water_layer_3").rename(f"m{month:02d}_swvl"),
                vpd.rename(f"m{month:02d}_vpd"),
            ]
        )
    return _reduce_monthly(ee, ee.Image.cat(clean), countries, year)


def extract_year_smap(ee, year: int, countries) -> pd.DataFrame:
    """One mid-month L4 RZSM image + L3 vegetation water content."""

    if year < 2015:
        return pd.DataFrame()
    crop = cropland_mask(ee, year)
    l4 = ee.ImageCollection(GEE_ASSETS["smap_l4"]).select("sm_rootzone")
    try:
        l3 = ee.ImageCollection(SMAP_L3).select("vegetation_water_content_am")
        has_l3 = True
    except Exception:
        has_l3 = False
        l3 = None
    clean = []
    for month in range(1, 13):
        if year == 2015 and month < 4:
            continue
        start = ee.Date.fromYMD(year, month, 15)
        rz = l4.filterDate(start, start.advance(2, "day")).first()
        rz = ee.Image(ee.Algorithms.If(rz, rz, l4.filterDate(start.advance(-5, "day"), start.advance(6, "day")).first()))
        rz = ee.Image(rz).updateMask(crop).rename(f"m{month:02d}_smap_rzsm")
        clean.append(rz)
        if has_l3:
            vw = l3.filterDate(ee.Date.fromYMD(year, month, 1), ee.Date.fromYMD(year, month, 1).advance(1, "month")).mean()
            vw = ee.Image(ee.Algorithms.If(vw, vw.updateMask(crop).rename(f"m{month:02d}_vod"), ee.Image.constant(0).updateMask(0).rename(f"m{month:02d}_vod")))
            clean.append(vw)
    if not clean:
        return pd.DataFrame()
    return _reduce_monthly(ee, ee.Image.cat(clean), countries, year, scale=9000)


def _merge_year_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        cols = [c for c in extra.columns if c not in {"iso3", "year", "month", "date"}]
        out = out.merge(extra[["iso3", "year", "month", *cols]], on=["iso3", "year", "month"], how="outer")
    return out


def extract_national_year(ee, year: int, countries, root: Path) -> pd.DataFrame:
    cache = _cache_dir(root) / f"year_{year}.csv"
    if cache.exists() and cache.stat().st_size > 200:
        return pd.read_csv(cache)
    parts = []
    for name, fn in (
        ("optical", extract_year_optical),
        ("era5", extract_year_era5),
        ("smap", extract_year_smap),
    ):
        try:
            part = fn(ee, year, countries)
            if len(part):
                parts.append(part)
                print(f"national {year} {name} rows={len(part)}", flush=True)
        except Exception as exc:
            print(f"SKIP national {year} {name}: {exc}", flush=True)
    out = _merge_year_frames(parts)
    if len(out):
        if "date" not in out:
            out["date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=15))
        out.to_csv(cache, index=False)
    return out


def combine_national_tables(root: Path) -> pd.DataFrame:
    d = _cache_dir(root)
    frames = [pd.read_csv(path) for path in sorted(d.glob("year_*.csv"))]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    out.to_csv(d / "national_monthly.csv", index=False)
    return out


def extract_all_national(root: Path, years: list[int] | None = None, iso3s: list[str] | None = None) -> pd.DataFrame:
    ee = init_ee(GEE_PROJECT)
    iso3s = iso3s or study_iso3s()
    years = years or list(range(2001, 2025))
    countries = country_feature_collection(ee, iso3s)
    print(f"national countries={countries.size().getInfo()}", flush=True)
    for year in years:
        extract_national_year(ee, year, countries, root)
    return combine_national_tables(root)


def sample_cropland_points(root: Path, n_per_country: int = 12) -> pd.DataFrame:
    """Cached cropland points for offline GOSIF / VODCA sampling."""

    cache = root / "data_interim" / "cropland_points.csv"
    if cache.exists() and cache.stat().st_size > 200:
        return pd.read_csv(cache)
    ee = init_ee(GEE_PROJECT)
    iso3s = study_iso3s()
    countries = country_feature_collection(ee, iso3s)
    crop = cropland_mask(ee, 2020)
    rows = []
    feats = countries.getInfo().get("features", [])
    for feat in feats:
        iso3 = feat.get("properties", {}).get("iso3")
        geom = feat.get("geometry")
        if not iso3 or not geom:
            continue
        try:
            pts = (
                crop.sample(
                    region=ee.Geometry(geom),
                    scale=8000,
                    numPixels=n_per_country,
                    seed=7,
                    geometries=True,
                    dropNulls=True,
                )
                .getInfo()
            )
            for p in pts.get("features", []):
                coords = p.get("geometry", {}).get("coordinates")
                if coords:
                    rows.append({"iso3": iso3, "lon": coords[0], "lat": coords[1]})
        except Exception as exc:
            print(f"SKIP points {iso3}: {exc}", flush=True)
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len(out):
        out.to_csv(cache, index=False)
    return out
