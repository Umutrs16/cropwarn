"""Build the observational world from FAOSTAT labels + GEE breadbasket series."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from cropwarn.constants import (
    BREADBASKET_POINTS,
    COUNTRIES,
    COUNTRY_BY_ISO3,
    FAOSTAT_ELEMENTS,
    FAOSTAT_ITEMS,
    FAO_CODE_ALIASES,
    RICE_DUAL_SEASONS,
    STATIC_CALENDAR,
    TYPICAL_AREA_MHA,
)
from cropwarn.data.china_early_rice import EARLY_RICE_CALENDAR, load_china_early_rice
from cropwarn.data.india_apy import KHARIF_CALENDAR, load_india_kharif_rice
from cropwarn.data.labels import build_labels
from cropwarn.data.nass_maize import MAIZE_CALENDAR, load_usa_maize
from cropwarn.data.vodca import load_vodca_crop
from cropwarn.gee.rice_state import load_china_early_series, load_rice_state_series, load_usa_maize_series
from cropwarn.phenology.alignment import wrap_eos
from cropwarn.phenology.detect import estimate_sos_eos


def parse_qcl_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv") and "flag" not in n.lower())
        with zf.open(name) as fh:
            df = pd.read_csv(fh, encoding="latin-1", low_memory=False)
    item_codes = set(FAOSTAT_ITEMS.values())
    el_codes = set(FAOSTAT_ELEMENTS.values())
    df = df[df["Item Code"].isin(item_codes) & df["Element Code"].isin(el_codes)].copy()
    inv_item = {v: k for k, v in FAOSTAT_ITEMS.items()}
    inv_el = {v: k for k, v in FAOSTAT_ELEMENTS.items()}
    df["crop"] = df["Item Code"].map(inv_item)
    df["element"] = df["Element Code"].map(inv_el)
    fao_to_iso = {c.fao_code: c.iso3 for c in COUNTRIES}
    fao_to_iso.update(FAO_CODE_ALIASES)
    df["iso3"] = df["Area Code"].map(fao_to_iso)
    df = df[df["iso3"].notna()].copy()
    df["year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["value"] = pd.to_numeric(df["Value"], errors="coerce")
    wide = (
        df.dropna(subset=["crop", "element", "year"])
        .pivot_table(index=["iso3", "crop", "year"], columns="element", values="value", aggfunc="first")
        .reset_index()
    )
    wide.columns.name = None
    if "area_harvested" in wide:
        wide["harvested_ha"] = wide["area_harvested"]
    if "yield" in wide:
        # Current QCL normalized yield is kg/ha (element 5412).
        wide["yield_tha"] = wide["yield"] / 1000.0
    elif "production" in wide and "harvested_ha" in wide:
        wide["yield_tha"] = wide["production"] / wide["harvested_ha"].replace(0, pd.NA)
    return wide


def parse_trade_zip(zip_path: Path, year_start: int = 2000) -> pd.DataFrame:
    """Keep bilateral export quantities for the four staples and study countries."""

    item_codes = set(FAOSTAT_ITEMS.values())
    fao_to_iso = {c.fao_code: c.iso3 for c in COUNTRIES}
    fao_to_iso.update(FAO_CODE_ALIASES)
    inv_item = {v: k for k, v in FAOSTAT_ITEMS.items()}
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            chunk_iter = pd.read_csv(fh, encoding="latin-1", low_memory=False, chunksize=400_000)
            for chunk in chunk_iter:
                if "Item Code" not in chunk.columns:
                    continue
                sub = chunk[
                    chunk["Item Code"].isin(item_codes)
                    & chunk["Element Code"].eq(5910)
                    & (pd.to_numeric(chunk["Year"], errors="coerce") >= year_start)
                ]
                if sub.empty:
                    continue
                reporter = sub.get("Reporter Country Code", sub.get("Reporter Country Code (M49)"))
                partner = sub.get("Partner Country Code", sub.get("Partner Country Code (M49)"))
                sub = sub.copy()
                sub["exporter"] = reporter.map(fao_to_iso)
                sub["importer"] = partner.map(fao_to_iso)
                sub = sub[sub["exporter"].notna() & sub["importer"].notna()]
                if sub.empty:
                    continue
                sub["crop"] = sub["Item Code"].map(inv_item)
                sub["year"] = pd.to_numeric(sub["Year"], errors="coerce").astype(int)
                sub["qty"] = pd.to_numeric(sub["Value"], errors="coerce")
                rows.append(sub[["year", "crop", "exporter", "importer", "qty"]])
    if not rows:
        return pd.DataFrame(columns=["year", "crop", "exporter", "importer", "export_share", "import_dependency"])
    raw = pd.concat(rows, ignore_index=True)
    raw = raw.groupby(["year", "crop", "exporter", "importer"], as_index=False)["qty"].sum()
    totals = raw.groupby(["year", "crop", "importer"])["qty"].transform("sum")
    raw["export_share"] = raw["qty"] / totals.replace(0, np.nan)
    imp_tot = raw.groupby(["year", "crop", "importer"])["qty"].transform("sum")
    raw["import_dependency"] = np.clip(imp_tot / (imp_tot + imp_tot.median()), 0.05, 0.95)
    return raw.dropna(subset=["export_share"])


def _calendar(crop: str, iso3: str) -> tuple[int, int]:
    table = STATIC_CALENDAR.get(crop, {})
    return table.get(iso3, table.get("default", (120, 280)))


def _season_slice(base: pd.DataFrame, year: int, sos0: int, eos0: int, preseason_days: int = 16) -> pd.DataFrame:
    span = max(wrap_eos(sos0, eos0) - sos0, 60)
    start = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(sos0 - 1))
    if sos0 > 200 and eos0 < sos0:
        start = pd.Timestamp(year=year - 1, month=1, day=1) + pd.Timedelta(days=int(sos0 - 1))
    end = start + pd.Timedelta(days=int(span))
    pad = int(preseason_days)
    win = base[(base["date"] >= start - pd.Timedelta(days=pad)) & (base["date"] <= end + pd.Timedelta("10D"))].copy()
    if win.empty:
        return win
    win["doy_abs"] = (win["date"] - start).dt.days + sos0
    win["progress"] = np.clip(100.0 * (win["doy_abs"] - sos0) / max(span, 1), 0, 100)
    return win


def load_national_monthly(root: Path) -> pd.DataFrame:
    path = root / "data_raw" / "gee_national" / "national_monthly.csv"
    if not path.exists():
        frames = list((root / "data_raw" / "gee_national").glob("year_*.csv")) if (root / "data_raw" / "gee_national").exists() else []
        if not frames:
            return pd.DataFrame()
        out = pd.concat([pd.read_csv(p) for p in frames], ignore_index=True)
    else:
        out = pd.read_csv(path)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    return out


def load_gosif_national(root: Path) -> pd.DataFrame:
    crop = root / "data_interim" / "gosif_crop.csv"
    path = crop if crop.exists() and crop.stat().st_size > 50 else root / "data_interim" / "gosif_national.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_crop_series(root: Path) -> pd.DataFrame:
    path = root / "data_raw" / "gee_crop" / "crop_series.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    d = root / "data_raw" / "gee_crop"
    if not d.exists():
        return pd.DataFrame()
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in sorted(d.glob("*_20*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def observational_inventory(root: Path, world: dict | None = None) -> dict:
    national = load_national_monthly(root)
    gosif = load_gosif_national(root)
    smap_ok = False
    vod_ok = False
    if len(national):
        smap_ok = "smap_rzsm" in national.columns and national["smap_rzsm"].notna().any()
        vod_ok = "vod" in national.columns and national["vod"].notna().any()
    if world is not None and world.get("series") is not None and len(world["series"]):
        series = world["series"]
        if "sif" in series and series["sif"].notna().any() and len(gosif):
            pass
        if "vod" in series and series["vod"].notna().any():
            vod_ok = True
        if "rzsm" in series and series["rzsm"].notna().any() and smap_ok:
            smap_ok = True
    trade_n = 0
    if world is not None and world.get("trade") is not None:
        trade_n = len(world["trade"])
    return {
        "faostat_yield": (root / "data_raw" / "Production_Crops_Livestock_E_All_Data_(Normalized).zip").exists(),
        "modis_optical": bool(len(national) or (root / "data_raw" / "gee_ts").exists()),
        "era5_land": True,
        "smap_l4": bool(smap_ok),
        "mod17_gpp_proxy": True,
        "gosif": bool(len(gosif)),
        "tropomi_sif": bool(
            (root / "data_interim" / "tropomi_crop.csv").exists()
            and (root / "data_interim" / "tropomi_crop.csv").stat().st_size > 50
        ),
        "vodca": bool((root / "data_interim" / "vodca_crop.csv").exists() and (root / "data_interim" / "vodca_crop.csv").stat().st_size > 50),
        "smap_vod": bool(vod_ok),
        "bilateral_trade": bool(trade_n),
        "headline_crops": ["wheat", "soybean"],
        "seasonal_rice": bool((root / "data_interim" / "china_early_rice.csv").exists()),
        "india_rice_held_out": bool(
            (root / "data_interim" / "india_kharif_rice_district.csv").exists()
            or (root / "data_interim" / "india_kharif_rice.csv").exists()
        ),
        # China early rice harvest R² is usable, but B4 does not add lead
        # beyond optical and P_full harvest R² < 0.15, so it stays out.
        "seasonal_rice_held_out": True,
        "usa_maize": bool((root / "data_interim" / "usa_maize.csv").exists()),
        # In-sample July SIF tracks NASS anomalies; OOS harvest R² does not.
        "usa_maize_held_out": True,
        "eu_maize": bool((root / "data_interim" / "eu_maize.csv").exists()),
        # 2022 HU/RO and 2018 DE north are real; harvest R²≈0.07, no earliest_lead.
        "eu_maize_held_out": True,
        "gee_project": "hallowed-chain-458005-g9",
        "sample": (
            "crop-specific SPAM/CROPGRIDS/WorldCereal points, 16-day optical"
            if len(load_crop_series(root))
            else ("national cropland-weighted (MCD12Q1 year-specific)" if len(national) else "breadbasket point")
        ),
        "national_rows": int(len(national)),
        "gosif_rows": int(len(gosif)),
    }


def _load_gee_tables(root: Path, iso3: str) -> dict[str, pd.DataFrame]:
    d = root / "data_raw" / "gee_ts"
    out = {}
    for name in ("optical", "lai", "lst", "era5", "smap", "gpp", "lsp"):
        p = d / f"{iso3}_{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p, parse_dates=["date"] if name != "lsp" else None)
    return out


def _national_as_tables(national: pd.DataFrame, iso3: str) -> dict[str, pd.DataFrame]:
    if national is None or national.empty:
        return {}
    sub = national[national["iso3"] == iso3].copy()
    if sub.empty:
        return {}
    sub["date"] = pd.to_datetime(sub["date"])
    out: dict[str, pd.DataFrame] = {}
    opt_cols = [c for c in ("date", "ndvi", "evi", "evi_std", "stress_area_frac") if c in sub.columns]
    if "evi" in sub.columns:
        out["optical"] = sub[opt_cols].dropna(subset=["evi"], how="all")
    if "lst" in sub.columns:
        out["lst"] = sub[["date", "lst"]].dropna(subset=["lst"])
    era_cols = [c for c in ("date", "t2m", "precip", "et0", "swvl", "vpd") if c in sub.columns]
    if len(era_cols) > 1:
        out["era5"] = sub[era_cols]
    if "smap_rzsm" in sub.columns:
        out["smap"] = sub[["date", "smap_rzsm"]].dropna(subset=["smap_rzsm"])
    if "gpp" in sub.columns:
        out["gpp"] = sub[["date", "gpp"]].dropna(subset=["gpp"])
    if "vod" in sub.columns:
        out["vod"] = sub[["date", "vod"]].dropna(subset=["vod"])
    return out


def _merge_asof(left: pd.DataFrame, right: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if right.empty:
        for c in cols:
            left[c] = np.nan
        return left
    r = right.dropna(subset=["date"]).sort_values("date")
    return pd.merge_asof(left.sort_values("date"), r[["date", *cols]], on="date", direction="nearest", tolerance=pd.Timedelta("20D"))


def _assemble_base(tabs: dict[str, pd.DataFrame], gosif: pd.DataFrame | None, iso3: str) -> pd.DataFrame:
    opt = tabs["optical"].copy()
    opt["date"] = pd.to_datetime(opt["date"])
    base = opt.dropna(subset=["date"]).sort_values("date")
    if "lai" in tabs:
        base = _merge_asof(base, tabs["lai"], ["lai"])
    else:
        base["lai"] = np.nan
    if "lst" in tabs:
        base = _merge_asof(base, tabs["lst"], ["lst"])
    else:
        base["lst"] = np.nan
    if "era5" in tabs:
        e5 = tabs["era5"].copy()
        e5["date"] = pd.to_datetime(e5["date"])
        base = _merge_asof(base, e5, [c for c in ("t2m", "precip", "et0", "swvl", "vpd") if c in e5.columns])
    if "smap" in tabs:
        sm = tabs["smap"].copy()
        sm["date"] = pd.to_datetime(sm["date"])
        base = _merge_asof(base, sm, ["smap_rzsm"])
    else:
        base["smap_rzsm"] = np.nan
    if "gpp" in tabs:
        base = _merge_asof(base, tabs["gpp"], ["gpp"])
    else:
        base["gpp"] = np.nan
    if "vod" in tabs:
        base = _merge_asof(base, tabs["vod"], ["vod"])
    else:
        base["vod"] = np.nan
    if gosif is not None and len(gosif):
        g = gosif[gosif["iso3"] == iso3]
        if "crop" in g.columns and "crop" in base.columns:
            g = g[g["crop"] == base["crop"].iloc[0]] if len(base) else g
        g = g[["date", "sif"]].copy()
        if len(g):
            g["date"] = pd.to_datetime(g["date"])
            base = _merge_asof(base, g, ["sif"])
    if "swvl" not in base:
        base["swvl"] = np.nan
    if "vpd" not in base:
        base["vpd"] = np.nan
    if "precip" not in base:
        base["precip"] = np.nan
    if "evi_std" not in base:
        base["evi_std"] = np.nan
    if "stress_area_frac" not in base:
        base["stress_area_frac"] = np.nan
    base["rzsm"] = base["smap_rzsm"].where(base["smap_rzsm"].notna(), base.get("swvl"))
    if "sif" not in base:
        base["sif"] = np.nan
    # GPP is a fallback photosynthesis proxy only when true GOSIF is missing.
    base["sif"] = base["sif"].where(base["sif"].notna(), base.get("gpp"))
    if "vod" not in base:
        base["vod"] = np.nan
    return base


def _base_from_crop_series(crop_series: pd.DataFrame, iso3: str, crop: str, gosif: pd.DataFrame | None) -> pd.DataFrame:
    sub = crop_series[(crop_series["iso3"] == iso3) & (crop_series["crop"] == crop)].copy()
    if sub.empty or "evi" not in sub.columns:
        return pd.DataFrame()
    sub["date"] = pd.to_datetime(sub["date"])
    # LST 8-day rows have no EVI; keep 16-day optical dates so SOS/EOS is not noise.
    sub = sub.dropna(subset=["evi"]).sort_values("date")
    if sub.empty:
        return pd.DataFrame()
    if gosif is not None and len(gosif):
        g = gosif[gosif["iso3"] == iso3]
        if "crop" in g.columns:
            g = g[g["crop"] == crop]
        if len(g):
            g = g[["date", "sif"]].copy()
            g["date"] = pd.to_datetime(g["date"])
            sub = _merge_asof(sub, g, ["sif"])
    if "smap_rzsm" in sub.columns:
        sub["rzsm"] = sub["smap_rzsm"].where(sub["smap_rzsm"].notna(), sub.get("swvl"))
    elif "swvl" in sub.columns:
        sub["rzsm"] = sub["swvl"]
    else:
        sub["rzsm"] = np.nan
    if "sif" not in sub:
        sub["sif"] = sub["gpp"] if "gpp" in sub else np.nan
    else:
        sub["sif"] = sub["sif"].where(sub["sif"].notna(), sub["gpp"] if "gpp" in sub else np.nan)
    if "vod" not in sub:
        sub["vod"] = np.nan
    if "lai" not in sub:
        sub["lai"] = np.nan
    if "evi_std" not in sub:
        sub["evi_std"] = np.nan
    if "stress_area_frac" not in sub:
        sub["stress_area_frac"] = np.nan
    return sub


def build_series_for_country(
    root: Path,
    iso3: str,
    crops: list[str],
    years: list[int],
    lsp: pd.DataFrame | None = None,
    national: pd.DataFrame | None = None,
    gosif: pd.DataFrame | None = None,
    crop_series: pd.DataFrame | None = None,
    preseason_days: int = 16,
) -> tuple[list[dict], list[dict]]:
    shared = None
    if crop_series is None or crop_series.empty:
        tabs = _national_as_tables(national, iso3) if national is not None and len(national) else {}
        if "optical" not in tabs:
            tabs = _load_gee_tables(root, iso3)
        if "optical" not in tabs:
            return [], []
        shared = _assemble_base(tabs, gosif, iso3)
    _ = lsp  # MCD12Q2 land-surface phenology is not used as crop SOS/EOS.

    series_rows = []
    panel_extra = []
    for crop in crops:
        if iso3 not in TYPICAL_AREA_MHA.get(crop, {}):
            continue
        sos0, eos0 = _calendar(crop, iso3)
        if crop_series is not None and len(crop_series):
            base = _base_from_crop_series(crop_series, iso3, crop, gosif)
            if base.empty and shared is not None:
                base = shared
        else:
            base = shared
        if base is None or base.empty:
            continue
        for year in years:
            win = _season_slice(base, year, sos0, eos0, preseason_days=preseason_days)
            if win.empty:
                continue
            sos_dyn, eos_dyn = estimate_sos_eos(win["date"], win["evi"].to_numpy(dtype=float), sos0, eos0)
            span_used = max(wrap_eos(sos_dyn, eos_dyn) - sos_dyn, 60)
            win["progress"] = np.clip(100.0 * (win["doy_abs"] - sos0) / span_used, 0, 100)
            for r in win.itertuples(index=False):
                series_rows.append(
                    {
                        "iso3": iso3,
                        "unit_id": iso3,
                        "crop": crop,
                        "year": year,
                        "season_id": 1,
                        "date": r.date,
                        "progress": float(r.progress),
                        "doy_abs": float(r.doy_abs),
                        "evi": getattr(r, "evi", np.nan),
                        "ndvi": getattr(r, "ndvi", np.nan),
                        "lai": getattr(r, "lai", np.nan),
                        "lst": getattr(r, "lst", np.nan),
                        "vpd": getattr(r, "vpd", np.nan),
                        "precip": getattr(r, "precip", np.nan),
                        "et0": getattr(r, "et0", np.nan),
                        "rzsm": getattr(r, "rzsm", np.nan),
                        "sif": getattr(r, "sif", np.nan),
                        "vod": getattr(r, "vod", np.nan),
                        "evi_std": getattr(r, "evi_std", np.nan),
                        "stress_area_frac": getattr(r, "stress_area_frac", np.nan),
                    }
                )
            if crop == "rice" and iso3 in RICE_DUAL_SEASONS:
                sos2, eos2 = RICE_DUAL_SEASONS[iso3]
                win2 = _season_slice(base, year, sos2, eos2, preseason_days=preseason_days)
                for r in win2.itertuples(index=False):
                    series_rows.append(
                        {
                            "iso3": iso3,
                            "unit_id": iso3,
                            "crop": crop,
                            "year": year,
                            "season_id": 2,
                            "date": r.date,
                            "progress": float(r.progress),
                            "doy_abs": float(r.doy_abs),
                            "evi": getattr(r, "evi", np.nan),
                            "ndvi": getattr(r, "ndvi", np.nan),
                            "lai": getattr(r, "lai", np.nan),
                            "lst": getattr(r, "lst", np.nan),
                            "vpd": getattr(r, "vpd", np.nan),
                            "precip": getattr(r, "precip", np.nan),
                            "et0": getattr(r, "et0", np.nan),
                            "rzsm": getattr(r, "rzsm", np.nan),
                            "sif": getattr(r, "sif", np.nan),
                            "vod": getattr(r, "vod", np.nan),
                            "evi_std": getattr(r, "evi_std", np.nan),
                            "stress_area_frac": getattr(r, "stress_area_frac", np.nan),
                        }
                    )
            evi = win["evi"].to_numpy(dtype=float)
            lst = win["lst"].to_numpy(dtype=float) if "lst" in win else np.array([])
            sm = win["rzsm"].to_numpy(dtype=float) if "rzsm" in win else np.array([])
            panel_extra.append(
                {
                    "iso3": iso3,
                    "unit_id": iso3,
                    "crop": crop,
                    "year": year,
                    "sos_static": sos0,
                    "eos_static": eos0,
                    "sos_true": sos_dyn,
                    "eos_true": eos_dyn,
                    "delta_sos": sos_dyn - sos0,
                    "delta_eos": eos_dyn - eos0,
                    "heat_true": float(np.nanmean(lst > 32)) if lst.size else 0.0,
                    "dry_true": float(np.nanmean(sm < 0.18)) if sm.size and np.isfinite(sm).any() else 0.0,
                    "sif_onset": np.nan,
                    "evi_onset": np.nan,
                    "sm_onset": np.nan,
                    "vod_onset": np.nan,
                }
            )
    return series_rows, panel_extra


def _apply_vodca(series: pd.DataFrame, vodca: pd.DataFrame) -> pd.DataFrame:
    if series.empty or vodca.empty:
        return series
    out = series.copy()
    if "unit_id" not in out.columns:
        out["unit_id"] = out["iso3"]
    v = vodca.copy()
    if "unit_id" not in v.columns:
        v["unit_id"] = v["iso3"]
    v["date"] = pd.to_datetime(v["date"])
    pieces = []
    for (uid, crop), g in out.groupby(["unit_id", "crop"], sort=False):
        r = v[(v["unit_id"] == uid) & (v["crop"] == crop)][["date", "vod"]].dropna().sort_values("date")
        if r.empty:
            pieces.append(g)
            continue
        left = g.sort_values("date")
        merged = pd.merge_asof(left, r.rename(columns={"vod": "vod_vodca"}), on="date", direction="nearest", tolerance=pd.Timedelta("20D"))
        if "vod" not in merged:
            merged["vod"] = merged["vod_vodca"]
        else:
            merged["vod"] = merged["vod_vodca"].where(merged["vod_vodca"].notna(), merged["vod"])
        pieces.append(merged.drop(columns=["vod_vodca"], errors="ignore"))
    return pd.concat(pieces, ignore_index=True) if pieces else out


def _gosif_for_units(root: Path, gosif: pd.DataFrame | None) -> pd.DataFrame:
    frames = []
    for name in ("gosif_china_early_rice.csv", "gosif_rice_state.csv", "gosif_rice_district.csv", "gosif_usa_maize.csv"):
        path = root / "data_interim" / name
        if path.exists() and path.stat().st_size > 50:
            frames.append(pd.read_csv(path, parse_dates=["date"]))
    if frames:
        return pd.concat(frames, ignore_index=True)
    if gosif is None or gosif.empty:
        return pd.DataFrame()
    g = gosif.copy()
    if "unit_id" not in g.columns:
        g["unit_id"] = g["iso3"]
    return g


def build_unit_rice_block(
    labels: pd.DataFrame,
    rice_series: pd.DataFrame,
    years: list[int],
    gosif_u: pd.DataFrame,
    sos0: int,
    eos0: int,
    crop: str = "rice",
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    if labels.empty or rice_series.empty:
        return [], [], pd.DataFrame()
    series_rows = []
    extras = []
    for unit_id, base0 in rice_series.groupby("unit_id", sort=False):
        iso3 = str(base0["iso3"].iloc[0]) if "iso3" in base0.columns else "IND"
        base = base0.copy()
        base["date"] = pd.to_datetime(base["date"])
        if "evi" in base.columns:
            base = base.dropna(subset=["evi"]).sort_values("date")
        if base.empty:
            continue
        if len(gosif_u):
            g = gosif_u[gosif_u["unit_id"] == unit_id] if "unit_id" in gosif_u.columns else gosif_u[gosif_u["iso3"] == iso3]
            if g.empty and "crop" in gosif_u.columns:
                g = gosif_u[(gosif_u["iso3"] == iso3) & (gosif_u["crop"] == crop)]
            if len(g):
                g = g[["date", "sif"]].copy()
                g["date"] = pd.to_datetime(g["date"])
                base = _merge_asof(base, g, ["sif"])
        if "smap_rzsm" in base.columns:
            base["rzsm"] = base["smap_rzsm"].where(base["smap_rzsm"].notna(), base.get("swvl"))
        elif "swvl" in base.columns:
            base["rzsm"] = base["swvl"]
        else:
            base["rzsm"] = np.nan
        if "sif" not in base:
            base["sif"] = np.nan
        if "vod" not in base:
            base["vod"] = np.nan
        for year in years:
            win = _season_slice(base, year, sos0, eos0)
            if win.empty:
                continue
            sos_dyn, eos_dyn = estimate_sos_eos(win["date"], win["evi"].to_numpy(dtype=float), sos0, eos0)
            span_used = max(wrap_eos(sos_dyn, eos_dyn) - sos_dyn, 60)
            win["progress"] = np.clip(100.0 * (win["doy_abs"] - sos0) / span_used, 0, 100)
            for r in win.itertuples(index=False):
                series_rows.append(
                    {
                        "iso3": iso3,
                        "unit_id": unit_id,
                        "crop": crop,
                        "year": year,
                        "season_id": 1,
                        "date": r.date,
                        "progress": float(r.progress),
                        "doy_abs": float(r.doy_abs),
                        "evi": getattr(r, "evi", np.nan),
                        "ndvi": getattr(r, "ndvi", np.nan),
                        "lai": getattr(r, "lai", np.nan),
                        "lst": getattr(r, "lst", np.nan),
                        "vpd": getattr(r, "vpd", np.nan),
                        "precip": getattr(r, "precip", np.nan),
                        "et0": getattr(r, "et0", np.nan),
                        "rzsm": getattr(r, "rzsm", np.nan),
                        "sif": getattr(r, "sif", np.nan),
                        "vod": getattr(r, "vod", np.nan),
                        "evi_std": getattr(r, "evi_std", np.nan),
                        "stress_area_frac": getattr(r, "stress_area_frac", np.nan),
                    }
                )
            lst = win["lst"].to_numpy(dtype=float) if "lst" in win else np.array([])
            sm = win["rzsm"].to_numpy(dtype=float) if "rzsm" in win else np.array([])
            extras.append(
                {
                    "iso3": iso3,
                    "unit_id": unit_id,
                    "crop": crop,
                    "year": year,
                    "sos_static": sos0,
                    "eos_static": eos0,
                    "sos_true": sos_dyn,
                    "eos_true": eos_dyn,
                    "delta_sos": sos_dyn - sos0,
                    "delta_eos": eos_dyn - eos0,
                    "heat_true": float(np.nanmean(lst > 32)) if lst.size else 0.0,
                    "dry_true": float(np.nanmean(sm < 0.18)) if sm.size and np.isfinite(sm).any() else 0.0,
                    "sif_onset": np.nan,
                    "evi_onset": np.nan,
                    "sm_onset": np.nan,
                    "vod_onset": np.nan,
                }
            )
    extra = pd.DataFrame(extras)
    if extra.empty:
        return series_rows, extras, pd.DataFrame()
    panel = labels.merge(extra, on=["iso3", "unit_id", "crop", "year"], how="inner")
    return series_rows, extras, panel


def build_india_rice_series(root: Path, years: list[int], gosif: pd.DataFrame | None = None) -> tuple[list[dict], list[dict], pd.DataFrame]:
    return build_unit_rice_block(
        load_india_kharif_rice(root),
        load_rice_state_series(root),
        years,
        _gosif_for_units(root, gosif),
        *KHARIF_CALENDAR,
    )


def build_china_early_rice_series(root: Path, years: list[int], gosif: pd.DataFrame | None = None) -> tuple[list[dict], list[dict], pd.DataFrame]:
    return build_unit_rice_block(
        load_china_early_rice(root),
        load_china_early_series(root),
        years,
        _gosif_for_units(root, gosif),
        *EARLY_RICE_CALENDAR,
    )


def build_usa_maize_series(root: Path, years: list[int], gosif: pd.DataFrame | None = None) -> tuple[list[dict], list[dict], pd.DataFrame]:
    return build_unit_rice_block(
        load_usa_maize(root),
        load_usa_maize_series(root),
        years,
        _gosif_for_units(root, gosif),
        *MAIZE_CALENDAR,
        crop="maize",
    )


def build_observational_world(
    root: Path,
    years: list[int],
    crops: list[str],
    trade: pd.DataFrame | None = None,
    preseason_days: int = 16,
) -> dict[str, pd.DataFrame]:
    qcl_zip = root / "data_raw" / "Production_Crops_Livestock_E_All_Data_(Normalized).zip"
    labels = parse_qcl_zip(qcl_zip)
    # FAO rice is a multi-season annual total — never a headline yield target.
    # FAO national maize is dropped once county NASS maize exists.
    fao_crops = [c for c in crops if c != "rice"]
    if len(load_usa_maize(root)):
        fao_crops = [c for c in fao_crops if c != "maize"]
    labels = labels[labels["year"].between(years[0], years[-1]) & labels["crop"].isin(fao_crops)].copy()
    labels = labels.dropna(subset=["yield_tha"])
    labels["unit_id"] = labels["iso3"]

    series_rows = []
    extras = []
    national = load_national_monthly(root)
    gosif = load_gosif_national(root)
    crop_series = load_crop_series(root)
    iso3s = sorted(
        set(BREADBASKET_POINTS)
        | (set(national["iso3"]) if len(national) else set())
        | (set(crop_series["iso3"]) if len(crop_series) else set())
    )
    for iso3 in iso3s:
        srows, erows = build_series_for_country(
            root,
            iso3,
            fao_crops,
            years,
            national=national if len(national) else None,
            gosif=gosif if len(gosif) else None,
            crop_series=crop_series if len(crop_series) else None,
            preseason_days=preseason_days,
        )
        series_rows.extend(srows)
        extras.extend(erows)

    extra = pd.DataFrame(extras)
    extra_fao = extra.drop(columns=["unit_id"], errors="ignore") if len(extra) else extra
    panel = labels.merge(extra_fao, on=["iso3", "crop", "year"], how="inner")
    # India kharif/Aman was tested and held out: canopy does not track APY.
    # China NBS early rice is the seasonal label that can couple to RS.
    rice_rows, rice_extras, rice_panel = build_china_early_rice_series(root, years, gosif)
    series_rows.extend(rice_rows)
    if rice_panel is not None and len(rice_panel):
        panel = pd.concat([panel, rice_panel], ignore_index=True)
    maize_rows, maize_extras, maize_panel = build_usa_maize_series(root, years, gosif)
    series_rows.extend(maize_rows)
    if maize_panel is not None and len(maize_panel):
        panel = pd.concat([panel, maize_panel], ignore_index=True)

    meta = pd.DataFrame([asdict(c) for c in COUNTRIES])
    panel = panel.merge(meta, on="iso3", how="left")
    if "unit_id" not in panel.columns:
        panel["unit_id"] = panel["iso3"]
    else:
        panel["unit_id"] = panel["unit_id"].fillna(panel["iso3"])
    if "harvested_ha" not in panel or panel["harvested_ha"].isna().all():
        panel["harvested_ha"] = panel.apply(
            lambda r: TYPICAL_AREA_MHA.get(r["crop"], {}).get(r["iso3"], 0.2) * 1e6, axis=1
        )
    panel = build_labels(panel)
    panel["shock_true"] = panel["yield_anomaly"]
    series = pd.DataFrame(series_rows)
    series = _apply_vodca(series, load_vodca_crop(root))
    if trade is None:
        tm = root / "data_raw" / "Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip"
        trade = parse_trade_zip(tm) if tm.exists() else pd.DataFrame()
    countries = meta
    return {"panel": panel, "series": series, "trade": trade, "countries": countries}
