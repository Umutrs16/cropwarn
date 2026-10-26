"""Helpers for revision sub-experiments. Do not touch official results/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def filter_world_years(world: dict, year_min: int, year_max: int | None = None) -> dict:
    out = dict(world)
    panel = world["panel"]
    series = world["series"]
    m = panel["year"] >= year_min
    if year_max is not None:
        m = m & (panel["year"] <= year_max)
    out["panel"] = panel[m].copy()
    sm = series["year"] >= year_min
    if year_max is not None:
        sm = sm & (series["year"] <= year_max)
    out["series"] = series[sm].copy()
    return out


def attach_tropomi_as_sif(world: dict, tropomi: pd.DataFrame) -> dict:
    """Replace series SIF with observed TROPOMI. No GPP/GOSIF fill."""

    if tropomi is None or tropomi.empty:
        raise ValueError("TROPOMI table is empty")
    series = world["series"].copy()
    series["date"] = pd.to_datetime(series["date"])
    t = tropomi.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.rename(columns={"tropomi_sif": "tropomi_sif"})
    merged = []
    for (iso3, crop), g in series.groupby(["iso3", "crop"], sort=False):
        r = t[(t["iso3"] == iso3) & (t["crop"] == crop)][["date", "tropomi_sif"]].sort_values("date")
        left = g.sort_values("date")
        if r.empty:
            left = left.copy()
            left["tropomi_sif"] = np.nan
            merged.append(left)
            continue
        merged.append(pd.merge_asof(left, r, on="date", direction="nearest", tolerance=pd.Timedelta("20D")))
    series = pd.concat(merged, ignore_index=True)
    series["sif"] = pd.to_numeric(series["tropomi_sif"], errors="coerce")
    out = dict(world)
    out["series"] = series
    return out


def revision_config_extra(test_start: int, min_train: int) -> dict:
    return {"validation": {"rolling_test_start": test_start, "min_train_years": min_train}}
