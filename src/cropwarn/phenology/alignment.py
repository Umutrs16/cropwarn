"""Static, real-time dynamic, and oracle phenology. Early forecasts never see future EOS."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class PhenologyVersion(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"  # real-time: SOS from observations <= cutoff + historical EOS prior
    ORACLE = "oracle"  # post-season upper bound only


@dataclass
class SeasonWindow:
    sos: int
    eos: int
    harvest_doy: int
    version: PhenologyVersion


def wrap_eos(sos: int, eos: int) -> int:
    return int(eos if eos > sos else eos + 365)


def season_progress_signed(doy: int, sos: int, eos: int) -> float:
    """Season progress in percent. Negative = before sowing; >100 = after maturity."""

    span = max(wrap_eos(sos, eos) - sos, 1)
    pos = doy - sos
    if eos <= sos and doy < sos:
        pos = doy + 365 - sos
    return float(100.0 * pos / span)


def season_progress(doy: int, sos: int, eos: int) -> float:
    return float(np.clip(season_progress_signed(doy, sos, eos), 0.0, 100.0))


STAGE_BINS = (
    (-999, 0, "pre_sowing"),
    (0, 20, "sowing_emergence"),
    (20, 50, "vegetative"),
    (50, 70, "flowering"),
    (70, 90, "grain_filling"),
    (90, 999, "maturity"),
)


def stage_name(progress: float) -> str:
    for lo, hi, name in STAGE_BINS:
        if lo <= progress < hi:
            return name
    return "maturity"


def harvest_doy(eos: int) -> int:
    return int(eos if eos <= 365 else eos - 365)


def historical_prior(history: pd.DataFrame, iso3: str, crop: str, year: int, col: str) -> float:
    past = history[(history["iso3"] == iso3) & (history["crop"] == crop) & (history["year"] < year)]
    if past.empty or col not in past:
        return np.nan
    return float(past[col].median())


def historical_eos_prior(history: pd.DataFrame, iso3: str, crop: str, year: int) -> float:
    col = "eos_true" if history is not None and "eos_true" in history else "eos_static"
    return historical_prior(history, iso3, crop, year, col)


def _on_or_after(cutoff_doy: int, event_doy: int, sos: int) -> bool:
    cut = cutoff_doy if cutoff_doy >= sos else cutoff_doy + 365
    ev = event_doy if event_doy >= sos else event_doy + 365
    return cut >= ev


def realtime_season(
    row: pd.Series,
    cutoff_doy: int,
    history: pd.DataFrame,
    version: PhenologyVersion,
) -> SeasonWindow:
    if version is PhenologyVersion.STATIC:
        return SeasonWindow(int(row["sos_static"]), int(row["eos_static"]), harvest_doy(int(row["eos_static"])), version)
    if version is PhenologyVersion.ORACLE:
        return SeasonWindow(int(row["sos_true"]), int(row["eos_true"]), harvest_doy(int(row["eos_true"])), version)

    # Real-time: use observed SOS only after greenup has already occurred.
    sos_obs = int(row["sos_true"])
    sos_static = int(row["sos_static"])
    hist_sos = historical_prior(history, row["iso3"], row["crop"], int(row["year"]), "sos_true")
    if _on_or_after(cutoff_doy, sos_obs, sos_static):
        sos = sos_obs
    else:
        sos = int(round(hist_sos)) if np.isfinite(hist_sos) else sos_static

    true_eos = int(row["eos_true"])
    prior = historical_eos_prior(history, row["iso3"], row["crop"], int(row["year"]))
    if not np.isfinite(prior):
        prior = float(row["eos_static"])
    harvest = harvest_doy(true_eos)
    if cutoff_doy < harvest - 16:
        span = wrap_eos(sos_static, int(row["eos_static"])) - sos_static
        eos = int(round(0.65 * prior + 0.35 * (sos + span)))
        if eos > 365:
            eos = eos - 365
    else:
        eos = true_eos
    return SeasonWindow(sos, eos, harvest_doy(eos), version)


def cutoff_doy_for_lead(harvest: int, lead: int) -> int:
    x = harvest - lead
    while x <= 0:
        x += 365
    return int(x)


def cutoff_timestamp(year: int, harvest: int, lead: int) -> pd.Timestamp:
    harvest_date = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=int(harvest) - 1)
    return harvest_date - pd.Timedelta(days=int(lead))


def align_cutoff(row: pd.Series, lead: int, history: pd.DataFrame, version: PhenologyVersion) -> dict:
    # Evaluation clock is actual harvest; stage bins use the real-time window.
    harvest_ref = harvest_doy(int(row["eos_true"] if pd.notna(row.get("eos_true")) else row["eos_static"]))
    cutoff = cutoff_doy_for_lead(harvest_ref, lead)
    win = realtime_season(row, cutoff, history, version)
    progress = season_progress(cutoff, win.sos, win.eos)
    return {
        "lead": lead,
        "phenology": version.value,
        "sos_used": win.sos,
        "eos_used": win.eos,
        "harvest_doy": harvest_ref,
        "cutoff_doy": cutoff,
        "cutoff_date": cutoff_timestamp(int(row["year"]), harvest_ref, lead),
        "progress_at_cutoff": progress,
    }
