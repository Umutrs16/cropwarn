"""Leakage-safe yield levels, rolling trends, anomalies and production."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        ok = dx != 0
        if ok.any():
            slopes.extend((dy[ok] / dx[ok]).tolist())
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def _fit_trend(years: np.ndarray, values: np.ndarray, method: str) -> np.ndarray:
    if len(years) < 3:
        return np.full_like(values, float(np.nanmean(values)), dtype=float)
    if method == "quadratic" and len(years) >= 6:
        coef = np.polyfit(years, values, deg=2)
    else:
        coef = np.polyfit(years, values, deg=1)
    return np.polyval(coef, years)


def rolling_trend_anomaly(
    panel: pd.DataFrame,
    value_col: str = "yield_tha",
    method: str = "linear",
    min_history: int = 4,
) -> pd.DataFrame:
    """Detrend year t using only years <= t-1. Never uses a full-period smoother."""

    rows: list[dict] = []
    keys = ["unit_id", "crop"] if "unit_id" in panel.columns else ["iso3", "crop"]
    for _, grp in panel.groupby(keys, sort=False):
        g = grp.sort_values("year")
        years = g["year"].to_numpy(dtype=int)
        values = g[value_col].to_numpy(dtype=float)
        trend = np.full(len(g), np.nan)
        anomaly = np.full(len(g), np.nan)
        zscore = np.full(len(g), np.nan)
        for i, year in enumerate(years):
            hist = years < year
            if hist.sum() < min_history:
                continue
            y_hist = years[hist].astype(float)
            v_hist = values[hist]
            ok = np.isfinite(v_hist)
            if ok.sum() < min_history:
                continue
            if method == "first_difference":
                prev = values[i - 1] if i > 0 else np.nan
                trend[i] = prev
                anomaly[i] = values[i] - prev if np.isfinite(prev) else np.nan
                sigma = float("nan")
                if hist.sum() >= 3:
                    diffs = np.diff(v_hist[ok])
                    sigma = float(np.std(diffs, ddof=1)) if diffs.size > 2 else np.nan
                if np.isfinite(sigma) and sigma > 1e-6 and np.isfinite(anomaly[i]):
                    zscore[i] = anomaly[i] / sigma
                continue
            if method == "theil_sen" and ok.sum() >= 4:
                slope, intercept = _theil_sen(y_hist[ok], v_hist[ok])
                fitted_hist = intercept + slope * y_hist[ok]
                resid = v_hist[ok] - fitted_hist
                sigma = float(np.std(resid, ddof=1)) if resid.size > 2 else np.nan
                trend[i] = float(intercept + slope * float(year))
            else:
                deg = 2 if method == "quadratic" and ok.sum() >= 6 else 1
                coef = np.polyfit(y_hist[ok], v_hist[ok], deg)
                fitted_hist = np.polyval(coef, y_hist[ok])
                resid = v_hist[ok] - fitted_hist
                sigma = float(np.std(resid, ddof=1)) if resid.size > 2 else np.nan
                trend[i] = float(np.polyval(coef, float(year)))
            if method == "percent" and np.isfinite(trend[i]) and abs(trend[i]) > 1e-6:
                anomaly[i] = (values[i] - trend[i]) / abs(trend[i])
            else:
                anomaly[i] = values[i] - trend[i]
            if np.isfinite(sigma) and sigma > 1e-6:
                zscore[i] = anomaly[i] / sigma
        out = g.copy()
        out["yield_trend"] = trend
        out["yield_anomaly"] = anomaly
        out["yield_z"] = zscore
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def add_production(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["production_t"] = out["yield_tha"] * out["harvested_ha"]
    if "yield_trend" in out:
        out["production_trend_t"] = out["yield_trend"] * out["harvested_ha"]
        out["production_anomaly_t"] = out["production_t"] - out["production_trend_t"]
        out["production_shock"] = out["production_anomaly_t"] / out["production_trend_t"]
    return out


def build_labels(panel: pd.DataFrame, detrend_method: str = "linear") -> pd.DataFrame:
    labeled = rolling_trend_anomaly(panel, method=detrend_method)
    return add_production(labeled)
