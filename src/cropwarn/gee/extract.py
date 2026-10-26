"""Country breadbasket time series from GEE. Cached per ISO3 / collection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cropwarn.constants import BREADBASKET_POINTS, GEE_PROJECT
from cropwarn.gee.assets import GEE_ASSETS
from cropwarn.gee.export import init_ee


def _cache_dir(root: Path) -> Path:
    d = root / "data_raw" / "gee_ts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_region(ee, collection, point, scale: int) -> pd.DataFrame:
    raw = collection.getRegion(point, scale).getInfo()
    header, *rows = raw
    df = pd.DataFrame(rows, columns=header)
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
    return df


def extract_modis_optical(ee, iso3: str, lat: float, lon: float) -> pd.DataFrame:
    point = ee.Geometry.Point([lon, lat])
    col = (
        ee.ImageCollection(GEE_ASSETS["mod13q1"])
        .filterDate("2001-01-01", "2025-01-01")
        .map(
            lambda img: img.select(["NDVI", "EVI"])
            .multiply(0.0001)
            .updateMask(img.select("SummaryQA").lte(1))
            .copyProperties(img, ["system:time_start"])
        )
    )
    df = _get_region(ee, col.select(["NDVI", "EVI"]), point, 500)
    return df.rename(columns={"NDVI": "ndvi", "EVI": "evi"})[["date", "ndvi", "evi"]]


def extract_modis_lai(ee, iso3: str, lat: float, lon: float) -> pd.DataFrame:
    point = ee.Geometry.Point([lon, lat])
    col = (
        ee.ImageCollection(GEE_ASSETS["mcd15a3h"])
        .filterDate("2002-07-01", "2025-01-01")
        .map(
            lambda img: img.select("Lai")
            .multiply(0.1)
            .updateMask(img.select("FparLai_QC").bitwiseAnd(1).eq(0))
            .copyProperties(img, ["system:time_start"])
        )
    )
    df = _get_region(ee, col.select(["Lai"]), point, 500)
    return df.rename(columns={"Lai": "lai"})[["date", "lai"]]


def extract_lst(ee, iso3: str, lat: float, lon: float) -> pd.DataFrame:
    point = ee.Geometry.Point([lon, lat])
    col = (
        ee.ImageCollection(GEE_ASSETS["mod11a2"])
        .filterDate("2001-01-01", "2025-01-01")
        .map(
            lambda img: img.select("LST_Day_1km")
            .multiply(0.02)
            .subtract(273.15)
            .updateMask(img.select("QC_Day").bitwiseAnd(3).eq(0))
            .copyProperties(img, ["system:time_start"])
        )
    )
    df = _get_region(ee, col.select(["LST_Day_1km"]), point, 1000)
    return df.rename(columns={"LST_Day_1km": "lst"})[["date", "lst"]]


def extract_era5_monthly(ee, iso3: str, lat: float, lon: float) -> pd.DataFrame:
    point = ee.Geometry.Point([lon, lat])
    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate("2001-01-01", "2025-01-01")
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
    df = _get_region(ee, col, point, 11000)
    out = pd.DataFrame(
        {
            "date": df["date"],
            "t2m": df["temperature_2m_max"] - 273.15,
            "tdew": df["dewpoint_temperature_2m"] - 273.15,
            "precip": df["total_precipitation_sum"] * 1000.0,  # m to mm
            "et0": df["potential_evaporation_sum"].abs() * 1000.0,
            "swvl": df["volumetric_soil_water_layer_3"],
        }
    )
    # Magnus VPD (kPa) from T and dewpoint.
    es = 0.6108 * (10 ** (7.5 * out["t2m"] / (out["t2m"] + 237.3)))
    ea = 0.6108 * (10 ** (7.5 * out["tdew"] / (out["tdew"] + 237.3)))
    out["vpd"] = (es - ea).clip(lower=0)
    return out[["date", "t2m", "precip", "et0", "swvl", "vpd"]]


def extract_smap(ee, iso3: str, lat: float, lon: float) -> pd.DataFrame:
    """Mid-month L4 snapshot. Do not mean the 3-hourly collection (memory limit)."""

    point = ee.Geometry.Point([lon, lat])
    raw = ee.ImageCollection(GEE_ASSETS["smap_l4"]).select("sm_rootzone")
    imgs = []
    for year in range(2015, 2025):
        for month in range(1, 13):
            if year == 2015 and month < 4:
                continue
            start = ee.Date.fromYMD(year, month, 15)
            img = raw.filterDate(start, start.advance(2, "day")).first()
            imgs.append(ee.Image(img).set("system:time_start", start.millis()))
    col = ee.ImageCollection(imgs)
    df = _get_region(ee, col.select(["sm_rootzone"]), point, 9000)
    return df.rename(columns={"sm_rootzone": "smap_rzsm"})[["date", "smap_rzsm"]]


def extract_gpp(ee, iso3: str, lat: float, lon: float) -> pd.DataFrame:
    """MOD17 GPP is a photosynthesis proxy, not GOSIF. Kept separate from true SIF."""

    point = ee.Geometry.Point([lon, lat])
    col = (
        ee.ImageCollection("MODIS/061/MOD17A2H")
        .filterDate("2001-01-01", "2025-01-01")
        .map(
            lambda img: img.select("Gpp")
            .multiply(0.0001)
            .updateMask(img.select("Psn_QC").bitwiseAnd(1).eq(0))
            .copyProperties(img, ["system:time_start"])
        )
    )
    df = _get_region(ee, col.select(["Gpp"]), point, 500)
    return df.rename(columns={"Gpp": "gpp"})[["date", "gpp"]]


def extract_lsp(ee, iso3: str, lat: float, lon: float) -> pd.DataFrame:
    point = ee.Geometry.Point([lon, lat])
    col = ee.ImageCollection(GEE_ASSETS["mcd12q2"]).filterDate("2001-01-01", "2025-01-01").select(["Greenup_1", "Senescence_1"])
    df = _get_region(ee, col, point, 500)
    # MCD12Q2 dates are days since 1970-01-01.
    epoch = pd.Timestamp("1970-01-01")
    out = pd.DataFrame({"date": df["date"]})
    out["year"] = out["date"].dt.year
    out["sos_mcd"] = pd.to_datetime(epoch + pd.to_timedelta(df["Greenup_1"], unit="D"), errors="coerce").dt.dayofyear
    out["eos_mcd"] = pd.to_datetime(epoch + pd.to_timedelta(df["Senescence_1"], unit="D"), errors="coerce").dt.dayofyear
    return out[["year", "sos_mcd", "eos_mcd"]]


EXTRACTORS = {
    "optical": extract_modis_optical,
    "lai": extract_modis_lai,
    "lst": extract_lst,
    "era5": extract_era5_monthly,
    "gpp": extract_gpp,
    "lsp": extract_lsp,
}


def extract_country(iso3: str, root: Path, ee=None, sensors: list[str] | None = None) -> dict[str, Path]:
    ee = ee or init_ee(GEE_PROJECT)
    lat, lon = BREADBASKET_POINTS[iso3]
    cache = _cache_dir(root)
    sensors = sensors or list(EXTRACTORS)
    paths = {}
    for name in sensors:
        path = cache / f"{iso3}_{name}.csv"
        if path.exists() and path.stat().st_size > 100:
            paths[name] = path
            continue
        try:
            df = EXTRACTORS[name](ee, iso3, lat, lon)
            df.insert(0, "iso3", iso3)
            df.to_csv(path, index=False)
            paths[name] = path
        except Exception as exc:
            print(f"SKIP {iso3} {name}: {exc}", flush=True)
    return paths


def extract_all(root: Path, iso3s: list[str] | None = None) -> list[str]:
    ee = init_ee(GEE_PROJECT)
    done = []
    for iso3 in iso3s or list(BREADBASKET_POINTS):
        extract_country(iso3, root, ee=ee)
        done.append(iso3)
        print(f"GEE extracted {iso3}", flush=True)
    return done
