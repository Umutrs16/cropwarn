"""Runtime assertions that catch the paper's fatal leakage modes."""

from __future__ import annotations

import pandas as pd


class LeakageError(RuntimeError):
    pass


def assert_no_future_years(train: pd.DataFrame, test: pd.DataFrame, cal: pd.DataFrame | None = None) -> None:
    tmax = int(test["year"].min())
    if train["year"].max() >= tmax:
        raise LeakageError(f"Train years reach {train['year'].max()} but test starts at {tmax}")
    if cal is not None and not cal.empty and cal["year"].max() >= tmax:
        raise LeakageError("Calibration fold includes the test year")


def assert_no_future_trade(trade_year: int, forecast_year: int) -> None:
    if trade_year >= forecast_year:
        raise LeakageError("Trade weights must come from a previous complete year")


def assert_realtime_phenology(features: pd.DataFrame) -> None:
    bad = features[(features["phenology"] == "dynamic") & (features["progress_at_cutoff"] > 100.0001)]
    if len(bad):
        raise LeakageError("Real-time phenology used post-season progress > 100")


def assert_no_label_columns(X_cols: list[str]) -> None:
    leaked = {
        "yield_tha",
        "yield_anomaly",
        "yield_trend",
        "production_t",
        "production_shock",
        "shock_true",
        "sif_onset",
        "evi_onset",
    }
    hit = leaked.intersection(X_cols)
    if hit:
        raise LeakageError(f"Label or oracle columns in features: {hit}")


def audit_split(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    assert_no_future_years(train, test, cal)
    return {
        "n_train": int(len(train)),
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
        "train_year_max": int(train["year"].max()) if len(train) else -1,
        "test_year": int(test["year"].min()) if len(test) else -1,
    }
