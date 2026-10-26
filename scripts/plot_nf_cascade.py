"""Build the four Nature Food result figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from cropwarn.data.io import read_table
from cropwarn.data.labels import rolling_trend_anomaly
from cropwarn.trade.buffers import skill_score
from cropwarn.trade.exposure import as_supply_shocks, import_exposure, overlay_observed_shocks
from cropwarn.viz.nf_cascade import (
    fig1_forecast_vs_network,
    fig2_yield_information,
    fig3_trade_maps,
    fig4_attenuation,
    fig5_buffers,
)

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
P0 = ROOT / "results" / "revision_p0"
P4 = ROOT / "results" / "revision_p4"
P5 = ROOT / "results" / "revision_p5"
P6 = ROOT / "results" / "revision_p6"


def _year_skill_ci(df: pd.DataFrame, auprc: str, pi: str, n_boot: int = 2000, seed: int = 42) -> dict:
    g = df.copy()
    g["skill"] = [skill_score(a, p) for a, p in zip(g[auprc], g[pi])]
    by = g.groupby("year")["skill"].mean().dropna()
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(by.to_numpy(), size=len(by), replace=True))) for _ in range(n_boot)]
    s = pd.Series(draws)
    return {
        "skill": float(by.mean()),
        "p025": float(s.quantile(0.025)),
        "p975": float(s.quantile(0.975)),
    }


def production_rows(food: pd.DataFrame, labels: pd.DataFrame, trade: pd.DataFrame, lead: int = 192) -> pd.DataFrame:
    lab = labels.copy()
    if "production_t" not in lab.columns:
        lab["production_t"] = pd.to_numeric(lab["yield_tha"], errors="coerce") * pd.to_numeric(
            lab["harvested_ha"], errors="coerce"
        )
    anom = rolling_trend_anomaly(lab, value_col="production_t", method="percent")
    obs = anom[["iso3", "crop", "year"]].copy()
    obs["shock_obs"] = pd.to_numeric(anom["yield_anomaly"], errors="coerce")
    shocks = as_supply_shocks(food[food["lead"] == int(lead)], labels)
    shocks = overlay_observed_shocks(shocks, obs)
    pieces = []
    for (year, crop), g in shocks.groupby(["year", "crop"]):
        need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
        expo = import_exposure(g[need], trade, int(year), str(crop), 1)
        if expo.empty:
            continue
        expo = expo.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"})
        expo["year"] = int(year)
        expo["crop"] = crop
        expo["lead"] = int(lead)
        pieces.append(expo)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def forest_table() -> pd.DataFrame:
    casc = read_table(P6 / "cascade_skill_ci.csv")
    short = read_table(P6 / "shortfall_sensitivity_year.csv")
    struct = read_table(P0 / "trade_structure.csv")
    rows = []
    labels = {
        "yield_induced": ("Yield-induced exposure", "exposure"),
        "production": ("Production-based exposure", "exposure"),
        "shortfall": ("Import-quantity shortfall", "adjustment"),
    }
    for _, r in casc.iterrows():
        lab, layer = labels[str(r["stage"])]
        rows.append(
            {
                "label": lab,
                "layer": layer,
                "skill": float(r["skill"]),
                "p025": float(r["p025"]),
                "p975": float(r["p975"]),
            }
        )
    ano = short[short["definition"] == "anomaly"]
    ci = _year_skill_ci(ano, "auprc_q20", "pi_q20")
    rows.append({"label": "Import-quantity anomaly", "layer": "adjustment", **ci})
    s192 = struct[struct["lead"] == 192]
    for outcome, lab in (
        ("hhi_increase", "HHI increase"),
        ("top1_increase", "Top-supplier share increase"),
        ("supplier_loss", "Supplier-count loss"),
    ):
        sub = s192[s192["outcome"] == outcome].copy()
        sub["pi"] = 0.20
        ci = _year_skill_ci(sub, "auprc", "pi")
        rows.append({"label": lab, "layer": "adjustment", **ci})
    return pd.DataFrame(rows)


def main() -> None:
    p6s = json.loads((P6 / "summary.json").read_text(encoding="utf-8"))
    p4s = json.loads((P4 / "summary.json").read_text(encoding="utf-8"))
    ylead = read_table(P6 / "suite_yield_lag1_by_lead.csv")
    plead = read_table(P6 / "suite_production_lag1_by_lead.csv")
    y = float(p6s["yield_lag1"]["auprc_192"])
    n0 = float(p6s["yield_lag1"]["n0_192"])
    fig1_forecast_vs_network(
        ylead,
        plead,
        read_table(P6 / "perm_yield_lag1.csv"),
        read_table(P6 / "perm_production_lag1.csv"),
        read_table(P6 / "ablation_yield_lag1.csv"),
        read_table(P6 / "ablation_production_lag1.csv"),
        float(p6s["yield_lag1"]["permutation"]["observed"]),
        float(p6s["production_lag1"]["permutation"]["observed"]),
        FIGS,
        recovery=(y - n0) / (1.0 - n0),
    )

    crop = read_table(ROOT / "results" / "revision_p1" / "lead_skill_by_crop.csv")
    crop = crop[crop["experiment"] == "B2_optical_climate"]
    p5s = json.loads((P5 / "summary.json").read_text(encoding="utf-8"))
    fig2_yield_information(crop, read_table(P5 / "official_ablation_full.csv"), p5s["exporter_skill"], FIGS)

    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    food = preds[(preds["experiment"] == "P_food_system") & preds["crop"].isin(["wheat", "soybean"])].copy()
    labels = labels[labels["crop"].isin(["wheat", "soybean"])].copy()
    trade = trade[trade["crop"].isin(["wheat", "soybean"])].copy()
    expo = production_rows(food, labels, trade, lead=192)
    lab = labels.copy()
    if "production_t" not in lab.columns:
        lab["production_t"] = pd.to_numeric(lab["yield_tha"], errors="coerce") * pd.to_numeric(lab["harvested_ha"], errors="coerce")
    anom = rolling_trend_anomaly(lab, value_col="production_t", method="percent")
    shocks = anom[["iso3", "crop", "year"]].copy()
    shocks["shock_true"] = pd.to_numeric(anom["yield_anomaly"], errors="coerce")
    years = [int(y) for y in p4s["top_exposure_years"]]
    fig3_trade_maps(expo, shocks, years, FIGS, lead=192)

    fig4_attenuation(forest_table(), FIGS)
    fig5_buffers(read_table(P4 / "buffer_split_auprc.csv"), read_table(P4 / "buffer_grid.csv"), FIGS)
    print("ok", FIGS)


if __name__ == "__main__":
    main()
