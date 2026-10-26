"""Mechanistic synthetic world used for a fully reproducible demo.

The generator is not a toy random table. It encodes the four paper hypotheses:
H1 physiology leads greenness under heat-drought;
H2 static calendars fail in low-latitude / multi-cropping systems;
H3 shock years leave the training distribution;
H4 exporter shocks propagate through a lagged trade network.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from cropwarn.constants import (
    BASE_YIELD_THA,
    COUNTRIES,
    STATIC_CALENDAR,
    TYPICAL_AREA_MHA,
)
from cropwarn.data.labels import build_labels


def _calendar(crop: str, iso3: str) -> tuple[int, int]:
    table = STATIC_CALENDAR.get(crop, {})
    return table.get(iso3, table.get("default", (120, 280)))


def _wrap_eos(sos: int, eos: int) -> int:
    return eos if eos > sos else eos + 365


def _doy_to_progress(doy: int, sos: int, eos: int) -> float:
    span = max(_wrap_eos(sos, eos) - sos, 1)
    pos = doy - sos
    if eos <= sos and doy < sos:
        pos = doy + 365 - sos
    return float(np.clip(100.0 * pos / span, 0.0, 100.0))


def _shock_intensity(year: int, region: str, crop: str) -> float:
    events = {
        2003: ({"Europe"}, {"wheat", "maize"}, 0.18),
        2010: ({"Europe", "CentralAsia"}, {"wheat"}, 0.28),
        2012: ({"NorthAmerica"}, {"maize", "soybean"}, 0.26),
        2015: ({"SoutheastAsia", "SouthAmerica", "Africa"}, {"rice", "maize", "soybean"}, 0.16),
        2016: ({"SoutheastAsia", "SouthAmerica", "Africa"}, {"rice", "maize"}, 0.10),
        2019: ({"Oceania"}, {"wheat"}, 0.22),
        2020: ({"Global"}, {"wheat", "rice", "maize", "soybean"}, 0.07),
        2022: ({"Europe", "CentralAsia"}, {"wheat", "maize"}, 0.24),
    }
    if year not in events:
        return 0.0
    regions, crops, mag = events[year]
    if crop not in crops:
        return 0.0
    if "Global" in regions or region in regions:
        return mag
    return 0.0


def _area_ha(crop: str, iso3: str, rng: np.random.Generator) -> float:
    base = TYPICAL_AREA_MHA.get(crop, {}).get(iso3)
    if base is None:
        return 0.0
    return float(max(1e4, base * 1e6 * rng.uniform(0.92, 1.08)))


def generate_world(
    years: list[int],
    crops: list[str],
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    panel_rows: list[dict] = []
    series_rows: list[dict] = []
    trade_rows: list[dict] = []

    active = []
    for crop in crops:
        for c in COUNTRIES:
            if _area_ha(crop, c.iso3, rng) > 0:
                active.append((crop, c))

    # Lagged bilateral export shares from structural exporter/importer roles.
    exporters = {crop: [c for cty, c in [(cr, cc) for cr, cc in active if cr == crop] if c.exporter] for crop in crops}
    for year in years:
        for crop in crops:
            exps = exporters.get(crop, [])
            imps = [c for cr, c in active if cr == crop and not c.exporter]
            if not exps or not imps:
                continue
            weights = np.array([0.4 + e.tech_level for e in exps])
            weights = weights / weights.sum()
            for j, imp in enumerate(imps):
                pref = rng.dirichlet(weights * 8.0)
                for i, exp in enumerate(exps):
                    share = float(pref[i])
                    if share < 0.02:
                        continue
                    trade_rows.append(
                        {
                            "year": year,
                            "crop": crop,
                            "exporter": exp.iso3,
                            "importer": imp.iso3,
                            "export_share": share,
                            "import_dependency": float(0.25 + 0.55 * (1.0 - imp.tech_level)),
                        }
                    )

    progress_grid = np.linspace(0, 100, 21)
    for crop, country in active:
        sos0, eos0 = _calendar(crop, country.iso3)
        season_len = _wrap_eos(sos0, eos0) - sos0
        jitter = 28 if country.multi_crop or country.latitude_band == "tropical" else 8
        base_y = BASE_YIELD_THA[crop] * (0.55 + 0.85 * country.tech_level)
        trend_slope = 0.018 + 0.025 * country.tech_level
        area0 = _area_ha(crop, country.iso3, rng)
        for year in years:
            delta_sos = float(rng.normal(0.0, jitter))
            if country.multi_crop:
                delta_sos += float(rng.normal(0.0, 10.0))
            delta_eos = float(rng.normal(0.35 * delta_sos, jitter * 0.6))
            sos = int(np.clip(sos0 + delta_sos, 1, 360))
            eos = int(sos + max(70, season_len + delta_eos))
            shock = _shock_intensity(year, country.region, crop)
            # Rainfed / low-irrigation systems take larger hits.
            shock *= 1.0 + 0.6 * (1.0 - country.irrigation)
            heat = float(np.clip(rng.normal(shock * 2.2, 0.25), -0.4, 2.8))
            dry = float(np.clip(rng.normal(shock * 2.5, 0.22), -0.3, 2.8))
            # Physiological onset is earlier than greenness onset.
            sif_onset = 28.0 + 8.0 * (1.0 - shock)  # % of season
            sm_onset = 24.0 + 6.0 * (1.0 - shock)
            vod_onset = 32.0 + 8.0 * (1.0 - shock)
            evi_onset = 48.0 + 10.0 * (1.0 - shock)

            season_sif = []
            season_evi = []
            season_vod = []
            season_sm = []
            season_lst = []
            span = max(eos - sos, 70)
            for p in progress_grid:
                evi = 0.62 * np.sin(np.pi * p / 100.0)
                sif = 0.70 * np.sin(np.pi * p / 100.0)
                vod = 0.55 + 0.25 * np.sin(np.pi * p / 100.0)
                sm = 0.32 - 0.04 * np.sin(np.pi * p / 100.0)
                lst = 24.0 + 6.0 * np.sin(np.pi * p / 100.0)
                if shock > 0:
                    sif -= 0.70 * shock * (1.0 / (1.0 + np.exp(-(p - sif_onset) / 4.0)))
                    sm -= 0.16 * dry * (1.0 / (1.0 + np.exp(-(p - sm_onset) / 4.0)))
                    vod -= 0.22 * shock * (1.0 / (1.0 + np.exp(-(p - vod_onset) / 5.0)))
                    evi -= 0.42 * shock * (1.0 / (1.0 + np.exp(-(p - evi_onset) / 5.0)))
                    lst += 4.5 * heat * (1.0 / (1.0 + np.exp(-(p - 35.0) / 6.0)))
                season_evi.append(evi + rng.normal(0, 0.012))
                season_sif.append(sif + rng.normal(0, 0.015))
                season_vod.append(vod + rng.normal(0, 0.010))
                season_sm.append(sm + rng.normal(0, 0.008))
                season_lst.append(lst + rng.normal(0, 0.20))
                doy_abs = sos + p / 100.0 * span
                series_rows.append(
                    {
                        "iso3": country.iso3,
                        "crop": crop,
                        "year": year,
                        "progress": float(p),
                        "doy": float(doy_abs if doy_abs <= 365 else doy_abs - 365),
                        "doy_abs": float(doy_abs),
                        "evi": season_evi[-1],
                        "ndvi": season_evi[-1] * 1.12,
                        "lai": max(0.2, season_evi[-1] * 5.2),
                        "sif": season_sif[-1],
                        "vod": season_vod[-1],
                        "rzsm": season_sm[-1],
                        "lst": season_lst[-1],
                        "vpd": 1.1 + 0.35 * heat + 0.02 * p,
                    }
                )

            # Yield is driven by mid-season physiology, so long-lead SIF/SM are informative.
            sif_int = float(np.mean(season_sif[6:12]))
            evi_int = float(np.mean(season_evi[12:18]))
            sm_int = float(np.mean(season_sm[5:11]))
            vod_int = float(np.mean(season_vod[7:13]))
            weather = 1.45 * sif_int + 0.80 * sm_int + 0.40 * vod_int + 0.15 * evi_int
            weather -= 0.22 * max(heat - 0.3, 0.0) * max(dry - 0.3, 0.0)
            year_idx = year - years[0]
            yield_tha = base_y * (1.0 + trend_slope * year_idx / 10.0) * (0.58 + 0.62 * weather)
            yield_tha *= rng.normal(1.0, 0.018)
            yield_tha = float(np.clip(yield_tha, 0.35, 16.0))

            harvested = area0 * rng.uniform(0.96, 1.04)
            if shock > 0.15:
                harvested *= 1.0 - 0.12 * shock * (1.0 - country.irrigation)
            panel_rows.append(
                {
                    "iso3": country.iso3,
                    "country": country.name,
                    "faostat_code": country.fao_code,
                    "region": country.region,
                    "latitude_band": country.latitude_band,
                    "multi_crop": country.multi_crop,
                    "exporter": country.exporter,
                    "irrigation": country.irrigation,
                    "tech_level": country.tech_level,
                    "crop": crop,
                    "year": year,
                    "yield_tha": yield_tha,
                    "harvested_ha": harvested,
                    "sos_static": sos0,
                    "eos_static": eos0,
                    "sos_true": sos,
                    "eos_true": eos,
                    "delta_sos": sos - sos0,
                    "delta_eos": eos - eos0,
                    "shock_true": shock,
                    "heat_true": heat,
                    "dry_true": dry,
                    "sif_onset": sif_onset,
                    "evi_onset": evi_onset,
                    "sm_onset": sm_onset,
                    "vod_onset": vod_onset,
                }
            )

    panel = build_labels(pd.DataFrame(panel_rows))
    series = pd.DataFrame(series_rows)
    trade = pd.DataFrame(trade_rows)
    countries = pd.DataFrame([asdict(c) for c in COUNTRIES])
    return {"panel": panel, "series": series, "trade": trade, "countries": countries}
