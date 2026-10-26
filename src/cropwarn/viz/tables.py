"""Manuscript tables 1–3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cropwarn.data.io import write_table


TABLE1 = [
    ["FAOSTAT QCL", "1961–latest", "country / year", "yield, area, production", "API / bulk", "main labels", "publication lag"],
    ["MCD12Q1 cropland", "2001–2023", "500 m / year", "LC 12/14 weights", "GEE", "national mask", "not crop-specific"],
    ["WorldCereal 2021", "2021", "10 m", "crop / irrigation", "GEE ESA/WorldCereal", "mask check", "not a yearly map"],
    ["MOD13Q1", "2000–NRT", "250 m / monthly mean", "NDVI/EVI", "GEE", "optical + EVI SOS/EOS", "cloud / saturation"],
    ["MOD11A2", "2000–NRT", "1 km / monthly mean", "LST", "GEE", "heat", "cloud gaps"],
    ["ERA5-Land", "1950–NRT", "11 km / monthly", "T, P, ET, SM", "GEE", "climate / RZSM fallback", "reanalysis"],
    ["SMAP L4 mid-month", "2015–NRT", "9 km / monthly", "root-zone SM", "GEE", "physiology", "short record"],
    ["SMAP L3 VWC", "2015–NRT", "9 km / monthly", "canopy water", "GEE", "VOD proxy", "not VODCA"],
    ["GOSIF v2 monthly", "2000–2024", "0.05° / monthly", "SIF", "UNH", "physiology (tested null)", "reconstructed SIF"],
    ["VODCA v2", "1987–2021", "0.25°", "VOD", "TU Wien", "canopy water", "ingested; headline is wheat/soy"],
    ["CROPGRIDS / SPAM", "2020", "0.05° / 10 km", "crop weights", "offline", "area weights", "single year"],
    ["FAOSTAT TM / BACI", "to 2024", "country pair / year", "bilateral trade", "offline", "exposure", "use t-1 network"],
    ["China NBS early rice", "provincial / seasonal", "ADM1", "early-rice yield", "NBS", "held-out Stage 11", "harvest R²≈0.61; B4 no lead"],
    ["USDA NASS maize", "2001–2023", "county / state", "maize yield", "NASS", "held-out Stage 11", "OOS harvest R² < 0"],
    ["Eurostat apro_cpshr", "2001–2023", "NUTS", "maize yield", "Eurostat", "held-out Stage 11", "harvest R²≈0.07"],
]


def write_manuscript_tables(bundle: dict, cfg) -> list[Path]:
    out = cfg.path("results")
    t1 = pd.DataFrame(
        TABLE1,
        columns=["dataset", "time", "resolution", "variables", "access", "role", "leakage_or_risk"],
    )
    t2 = bundle["metrics"].copy() if isinstance(bundle["metrics"], pd.DataFrame) else pd.DataFrame()
    t3 = bundle["lead_skill"].copy() if isinstance(bundle["lead_skill"], pd.DataFrame) else pd.DataFrame()
    paths = [
        write_table(t1, out / "table1_datasets.csv"),
        write_table(t2, out / "table2_performance.csv"),
        write_table(t3, out / "table3_ablation.csv"),
    ]
    md = ["# Table 1. Data sources", "", t1.to_markdown(index=False)]
    try:
        (out / "table1_datasets.md").write_text("\n".join(md), encoding="utf-8")
    except Exception:
        (out / "table1_datasets.md").write_text(t1.to_string(index=False), encoding="utf-8")
    return paths
