"""Lead-time, phenology-window and spatial-distribution feature factory."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cropwarn.constants import PHENOLOGY_WINDOWS
from cropwarn.phenology.alignment import PhenologyVersion, align_cutoff, season_progress, wrap_eos

OPTICAL = ("ndvi", "evi", "lai")
THERMAL = ("lst", "vpd")
CLIMATE = ("precip", "et0")
SM = ("rzsm",)
SIF = ("sif",)
VOD = ("vod",)

MODALITY_VARS = {
    "optical": OPTICAL,
    "thermal": THERMAL,
    "climate": CLIMATE + THERMAL,
    "sm": SM,
    "sif": SIF,
    "vod": VOD,
}


def _window_stats(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0 or not np.isfinite(arr).any():
        return {k: np.nan for k in ("last", "max", "min", "mean", "integral", "slope", "p10", "p90")}
    x = arr[np.isfinite(arr)]
    t = np.arange(len(x), dtype=float)
    slope = float(np.polyfit(t, x, 1)[0]) if len(x) >= 2 else 0.0
    return {
        "last": float(x[-1]),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "mean": float(np.mean(x)),
                    "integral": float(np.trapezoid(x) if hasattr(np, "trapezoid") else np.trapz(x)),
        "slope": slope,
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
    }


def _stress_days(values: np.ndarray, thresh: float, below: bool = True) -> float:
    if values.size == 0:
        return 0.0
    flag = values < thresh if below else values > thresh
    return float(np.sum(flag))


def _spatial_proxy(mean: float, shock: float) -> dict[str, float]:
    """Country-scale stand-in for Chen mean/std/frequency + stressed-area fraction."""

    std = abs(0.08 + 0.25 * shock)
    return {
        "std": std,
        "p10": mean - 1.28 * std,
        "p25": mean - 0.67 * std,
        "p50": mean,
        "p75": mean + 0.67 * std,
        "p90": mean + 1.28 * std,
        "stress_area_frac": float(np.clip(0.08 + 0.75 * shock, 0.0, 0.95)),
        "stress_contiguity": float(np.clip(0.05 + 0.65 * shock, 0.0, 0.95)),
    }


def build_feature_table(
    panel: pd.DataFrame,
    series: pd.DataFrame,
    leads: list[int],
    phenology: PhenologyVersion,
    modalities: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    vars_keep = []
    for m in modalities:
        vars_keep.extend(MODALITY_VARS[m])
    vars_keep = list(dict.fromkeys(vars_keep))

    series_keys = ["unit_id", "crop", "year"] if "unit_id" in series.columns else ["iso3", "crop", "year"]
    series_map: dict[tuple, pd.DataFrame] = {
        key: g.sort_values("progress")
        for key, g in series.groupby(series_keys, sort=False)
    }

    for _, row in panel.iterrows():
        if "unit_id" in series_keys:
            key = (row.get("unit_id", row["iso3"]), row["crop"], int(row["year"]))
        else:
            key = (row["iso3"], row["crop"], int(row["year"]))
        seq = series_map.get(key)
        if seq is None:
            continue
        seq2 = seq.iloc[0:0]
        if "season_id" in seq.columns:
            seq2 = seq[seq["season_id"] == 2]
            seq = seq[seq["season_id"].fillna(1) == 1]
        if seq.empty:
            continue
        for lead in leads:
            meta = align_cutoff(row, lead, panel, phenology)
            cutoff_p = meta["progress_at_cutoff"]
            cutoff_abs = float(meta["sos_used"]) + cutoff_p / 100.0 * max(
                wrap_eos(int(meta["sos_used"]), int(meta["eos_used"])) - int(meta["sos_used"]), 1
            )
            if "date" in seq.columns:
                pre = seq[pd.to_datetime(seq["date"]) <= pd.Timestamp(meta["cutoff_date"])]
            elif "doy_abs" in seq.columns:
                pre = seq[seq["doy_abs"] <= cutoff_abs]
            else:
                pre = seq[seq["progress"] <= cutoff_p]
            feat: dict = {
                **row.to_dict(),
                **meta,
            }
            # Modality availability: SMAP/VOD only reliable after 2015 in the real paper.
            year = int(row["year"])
            feat["mask_sm"] = 1.0 if "sm" in modalities else 0.0
            feat["mask_vod"] = 1.0 if year >= 2010 and "vod" in modalities else 0.0
            feat["mask_sif"] = 1.0 if "sif" in modalities else 0.0
            feat["mask_optical"] = 1.0
            feat["mask_thermal"] = 1.0

            for var in vars_keep:
                blocked = (var in SM and feat["mask_sm"] == 0) or (var in VOD and feat["mask_vod"] == 0)
                arr = np.array([]) if blocked or pre.empty else pre[var].to_numpy(dtype=float)
                stats = _window_stats(arr)
                for k, v in stats.items():
                    feat[f"{var}_{k}"] = v
                if var == "rzsm":
                    feat["rzsm_below_theta_days"] = 0.0 if blocked else _stress_days(arr, 0.22, True)
                if var == "lst":
                    feat["lst_hot_days"] = _stress_days(arr, 30.0, False)
                if var == "sif":
                    feat["sif_decline"] = 0.0
                    if arr.size >= 4:
                        feat["sif_decline"] = float(arr[: max(2, len(arr) // 3)].mean() - arr[-max(2, len(arr) // 3) :].mean())

            # Phenology-stage means (0-20 ... 80-100), truncated at cutoff.
            for lo, hi in PHENOLOGY_WINDOWS:
                if cutoff_p < lo:
                    for var in vars_keep:
                        feat[f"{var}_w{lo}_{hi}"] = np.nan
                    continue
                if "doy_abs" in seq.columns:
                    used_progress = seq["doy_abs"].map(
                        lambda d: season_progress(int(d if d <= 365 else d - 365), int(meta["sos_used"]), int(meta["eos_used"]))
                    )
                    w = seq.loc[(used_progress >= lo) & (used_progress < min(hi, cutoff_p + 1e-6)) & (seq["doy_abs"] <= cutoff_abs)]
                else:
                    w = seq[(seq["progress"] >= lo) & (seq["progress"] < min(hi, cutoff_p + 1e-6))]
                for var in vars_keep:
                    blocked = (var in SM and feat["mask_sm"] == 0) or (var in VOD and feat["mask_vod"] == 0)
                    feat[f"{var}_w{lo}_{hi}"] = np.nan if blocked or w.empty else float(w[var].mean())

            # Spatial distribution: use national cropland std / stress area when present.
            evi_mean = feat.get("evi_mean", np.nan)
            if not pre.empty and "lst" in pre:
                stress = float(np.nanmean(pre["lst"].to_numpy(dtype=float) > 32))
            else:
                stress = float(row.get("heat_true", 0.0) or 0.0)
            spatial = _spatial_proxy(0.0 if not np.isfinite(evi_mean) else evi_mean, stress)
            if not pre.empty and "evi_std" in pre.columns and pre["evi_std"].notna().any():
                spatial["std"] = float(np.nanmean(pre["evi_std"]))
            if not pre.empty and "stress_area_frac" in pre.columns and pre["stress_area_frac"].notna().any():
                spatial["stress_area_frac"] = float(np.nanmean(pre["stress_area_frac"]))
            for k, v in spatial.items():
                feat[f"spatial_{k}"] = v
            evi_arr = pre["evi"].to_numpy(dtype=float) if not pre.empty else np.array([np.nan])
            bins = np.linspace(0.0, 0.8, 6)
            hist, _ = np.histogram(evi_arr[np.isfinite(evi_arr)], bins=bins, density=False)
            hist = hist / hist.sum() if hist.sum() else np.zeros(5)
            for i, h in enumerate(hist):
                feat[f"evi_freq_b{i}"] = float(h)

            # Calendar adjustment only — never the oracle (sos_true/eos_true) gap.
            feat["sos_mismatch"] = abs(float(feat["sos_used"]) - float(row["sos_static"]))
            feat["eos_mismatch"] = abs(float(feat["eos_used"]) - float(row["eos_static"]))
            if len(seq2) and "date" in seq2.columns:
                pre2 = seq2[pd.to_datetime(seq2["date"]) <= pd.Timestamp(meta["cutoff_date"])]
                for var in vars_keep:
                    blocked = (var in SM and feat["mask_sm"] == 0) or (var in VOD and feat["mask_vod"] == 0)
                    arr2 = np.array([]) if blocked or pre2.empty else pre2[var].to_numpy(dtype=float)
                    feat[f"{var}_s2_mean"] = float(np.nanmean(arr2)) if arr2.size and np.isfinite(arr2).any() else np.nan
            rows.append(feat)

    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame, use_country_id: bool = False) -> list[str]:
    banned = {
        "yield_tha",
        "yield_trend",
        "yield_anomaly",
        "yield_z",
        "production_t",
        "production_trend_t",
        "production_anomaly_t",
        "production_shock",
        "harvested_ha",
        "shock_true",
        "heat_true",
        "dry_true",
        "sif_onset",
        "evi_onset",
        "sm_onset",
        "vod_onset",
        "country",
        "iso3",
        "unit_id",
        "state_code",
        "state_fips",
        "county_fips",
        "county",
        "geoid",
        "yield_bu_acre",
        "season",
        "phenology",
        "exporter",
        "multi_crop",
        "tech_level",
        "irrigation",
        "faostat_code",
        "sos_true",
        "eos_true",
    }
    cols = []
    for c in df.columns:
        if c in banned:
            continue
        if df[c].dtype == bool:
            cols.append(c)
            continue
        if np.issubdtype(df[c].dtype, np.number):
            cols.append(c)
    for extra in ("crop", "region", "latitude_band"):
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    if use_country_id and "iso3" in df.columns:
        cols.append("iso3")
    return cols
