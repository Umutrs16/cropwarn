"""Time- and event-aware validation splits. Random row splits are forbidden."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    name: str
    scheme: str
    train_idx: pd.Index
    cal_idx: pd.Index
    test_idx: pd.Index
    test_year: int | None = None
    heldout: str | None = None


def _year_cut(df: pd.DataFrame, train_end: int, test_year: int) -> tuple[pd.Index, pd.Index, pd.Index]:
    train = df.index[df["year"] < train_end]
    cal = df.index[df["year"] == train_end]
    test = df.index[df["year"] == test_year]
    return train, cal, test


def rolling_origin(df: pd.DataFrame, test_start: int, min_train_years: int) -> Iterator[Split]:
    years = sorted(df["year"].unique())
    for test_year in years:
        if test_year < test_start:
            continue
        prior = [y for y in years if y < test_year]
        if len(prior) < min_train_years + 1:
            continue
        train_end = prior[-1]
        train, cal, test = _year_cut(df, train_end, test_year)
        if len(train) == 0 or len(test) == 0:
            continue
        yield Split(f"rolling_{test_year}", "rolling_origin", train, cal, test, test_year)


def leave_one_year_out(df: pd.DataFrame, min_train_years: int) -> Iterator[Split]:
    years = sorted(df["year"].unique())
    for test_year in years:
        prior = [y for y in years if y < test_year]
        if len(prior) < min_train_years + 1:
            continue
        train_end = prior[-1]
        train, cal, test = _year_cut(df, train_end, test_year)
        if len(test) == 0:
            continue
        yield Split(f"loyo_{test_year}", "leave_one_year_out", train, cal, test, test_year)


def leave_one_region_out(df: pd.DataFrame, test_start: int) -> Iterator[Split]:
    years = sorted(df["year"].unique())
    test_years = [y for y in years if y >= test_start]
    if not test_years:
        return
    last = test_years[-1]
    prior = [y for y in years if y < last]
    if len(prior) < 4:
        return
    train_end = prior[-1]
    for region, g in df.groupby("region"):
        train = df.index[(df["year"] < train_end) & (df["region"] != region)]
        cal = df.index[(df["year"] == train_end) & (df["region"] != region)]
        test = df.index[(df["year"] == last) & (df["region"] == region)]
        if len(train) == 0 or len(test) == 0:
            continue
        yield Split(f"loro_{region}_{last}", "leave_one_region_out", train, cal, test, last, region)


def shock_holdout(df: pd.DataFrame, shock_years: list[int], min_train_years: int) -> Iterator[Split]:
    years = sorted(df["year"].unique())
    for year in shock_years:
        if year not in years:
            continue
        prior = [y for y in years if y < year]
        if len(prior) < min_train_years + 1:
            continue
        train_end = prior[-1]
        train, cal, test = _year_cut(df, train_end, year)
        if len(test) == 0:
            continue
        yield Split(f"shock_{year}", "shock_holdout", train, cal, test, year, str(year))


def iter_splits(df: pd.DataFrame, cfg, schemes: list[str] | None = None) -> Iterator[Split]:
    schemes = schemes or list(cfg.get("validation", "schemes"))
    test_start = int(cfg.get("validation", "rolling_test_start"))
    min_train = int(cfg.get("validation", "min_train_years"))
    shock_years = [int(s["year"]) for s in cfg.get("shocks")]
    if "rolling_origin" in schemes:
        yield from rolling_origin(df, test_start, min_train)
    if "leave_one_year_out" in schemes:
        yield from leave_one_year_out(df, min_train)
    if "leave_one_region_out" in schemes:
        yield from leave_one_region_out(df, test_start)
    if "shock_holdout" in schemes:
        yield from shock_holdout(df, shock_years, min_train)
