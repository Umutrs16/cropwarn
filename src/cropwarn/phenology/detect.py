"""Crop-calendar-aware SOS/EOS from in-season EVI. Not land-surface phenology."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cropwarn.phenology.alignment import wrap_eos


def _season_clock(doy: np.ndarray, sos_prior: int) -> np.ndarray:
    clock = doy.astype(float)
    clock[clock < sos_prior] += 365.0
    return clock


def estimate_sos_eos(
    dates: pd.Series | np.ndarray,
    evi: np.ndarray,
    sos_prior: int,
    eos_prior: int,
    search_days: int = 50,
    frac: float = 0.20,
) -> tuple[int, int]:
    """20% seasonal-amplitude crossings inside a window around the crop calendar.

    Amplitude uses only observations inside the search window, so a winter-wheat
    season is not overwritten by a summer land-surface greenup.
    """

    dates = pd.to_datetime(pd.Series(dates))
    evi = np.asarray(evi, dtype=float)
    ok = dates.notna().to_numpy() & np.isfinite(evi)
    if ok.sum() < 4:
        return int(sos_prior), int(eos_prior)

    doy = dates.dt.dayofyear.to_numpy()[ok]
    evi = evi[ok]
    clock = _season_clock(doy, int(sos_prior))
    eos_clock = float(wrap_eos(int(sos_prior), int(eos_prior)))
    mid = 0.5 * (float(sos_prior) + eos_clock)
    in_win = (clock >= sos_prior - search_days) & (clock <= eos_clock + search_days)
    if in_win.sum() < 4:
        in_win = np.ones(len(evi), dtype=bool)

    evw = evi[in_win]
    clw = clock[in_win]
    vmin, vmax = float(np.nanmin(evw)), float(np.nanmax(evw))
    amp = vmax - vmin
    if amp < 0.05:
        return int(sos_prior), int(eos_prior)
    thresh = vmin + frac * amp

    order = np.argsort(clw)
    clw, evw = clw[order], evw[order]
    if len(evw) >= 5:
        evw = np.convolve(evw, np.ones(3) / 3.0, mode="same")
    above = evw >= thresh
    sos = int(sos_prior)
    eos = int(eos_prior)
    if above.any():
        first = int(np.argmax(above))
        last = int(len(above) - 1 - np.argmax(above[::-1]))
        sos_hat = clw[first]
        eos_hat = clw[last]
        if abs(sos_hat - sos_prior) <= search_days:
            sos = int(np.clip(round(sos_hat), 1, 365))
        if abs(eos_hat - eos_clock) <= search_days:
            eos_raw = eos_hat if eos_hat <= 365 else eos_hat - 365
            eos = int(np.clip(round(eos_raw), 1, 365))
        # Guard against inverted seasons when the series is noisy.
        if wrap_eos(sos, eos) - sos < 40:
            return int(sos_prior), int(eos_prior)
    _ = mid
    return sos, eos


def realtime_sos(
    dates: pd.Series | np.ndarray,
    evi: np.ndarray,
    sos_prior: int,
    eos_prior: int,
    cutoff: pd.Timestamp,
    search_days: int = 50,
) -> int:
    """SOS using only observations on or before the forecast cutoff."""

    dates = pd.to_datetime(pd.Series(dates))
    evi = np.asarray(evi, dtype=float)
    keep = dates.notna() & (dates <= pd.Timestamp(cutoff))
    if keep.sum() < 4:
        return int(sos_prior)
    sos, _ = estimate_sos_eos(dates[keep], evi[keep], sos_prior, eos_prior, search_days=search_days)
    return sos
