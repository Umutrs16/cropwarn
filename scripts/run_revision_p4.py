"""Food-system cascade and buffer analysis. Writes only results/revision_p4/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.data.io import read_table, write_table
from cropwarn.features.factory import build_feature_table
from cropwarn.models.predictors import fit_predict
from cropwarn.phenology.alignment import PhenologyVersion
from cropwarn.trade.baselines import _auprc, importer_specific_truth, permute_forecast_auprc, year_block_auprc_ci
from cropwarn.trade.buffers import (
    attach_buffers,
    importer_accounts,
    importer_hhi_lag,
    lag_accounts,
    median_split_auprc,
    shortfall_given_exposure,
    skill_score,
)
from cropwarn.trade.exposure import as_supply_shocks, realized_import_change
from cropwarn.validation.metrics import regression_metrics
from cropwarn.validation.splits import rolling_origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_p4"
CROPS = ("wheat", "soybean")


def _mean_auprc(df: pd.DataFrame, col: str, lead: int | None = None) -> float:
    sub = df if lead is None else df[df["lead"] == int(lead)]
    return float(pd.to_numeric(sub[col], errors="coerce").mean())


def run_idr_and_buffers() -> dict:
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    expo = read_table(ROOT / "results" / "revision_p2" / "importer_exposure_rows.csv")
    labels = labels[labels["crop"].isin(CROPS)].copy()
    trade = trade[trade["crop"].isin(CROPS)].copy()
    expo = expo[expo["crop"].isin(CROPS)].copy()
    expo["lead"] = pd.to_numeric(expo["lead"], errors="coerce")
    acc = importer_accounts(trade, labels)
    acc_lag = lag_accounts(acc)
    hhi = importer_hhi_lag(trade)
    packed = attach_buffers(expo, acc_lag, hhi)
    write_table(acc, OUT / "importer_accounts.csv")
    write_table(packed, OUT / "exposure_with_buffers.csv")

    # Supply-system exposure: pooled historical Q80 of supply_obs
    rows = []
    for (year, crop, lead), g in packed.groupby(["year", "crop", "lead"]):
        past = packed[(packed["year"] < int(year)) & (packed["crop"] == crop) & (packed["lead"] == lead)]
        if past.empty or past["supply_obs"].notna().sum() < 8:
            continue
        thresh = float(past["supply_obs"].quantile(0.80))
        truth = pd.to_numeric(g["supply_obs"], errors="coerce") > thresh
        rec = {
            "year": int(year),
            "crop": crop,
            "lead": int(lead),
            "prevalence": float(truth.mean()),
            "auprc_supply": _auprc(truth, g["supply_hat"]),
            "auprc_portfolio": _auprc(truth, g["exposure_hat"]),
        }
        rows.append(rec)
    supply = pd.DataFrame(rows)
    write_table(supply, OUT / "supply_exposure_by_year.csv")

    # Realized shortfall vs portfolio / supply scores at 192 d
    chg = realized_import_change(trade)
    sub = packed[packed["lead"] == 192].copy()
    sub = sub.merge(
        chg.rename(columns={"importer": "importer"}),
        on=["year", "crop", "importer"],
        how="left",
    )
    ychg = pd.to_numeric(sub["import_chg"], errors="coerce")
    short = pd.Series(False, index=sub.index)
    for (year, crop), idx in sub.groupby(["year", "crop"]).groups.items():
        yy = ychg.loc[idx]
        if yy.notna().sum() < 8:
            continue
        short.loc[idx] = yy <= yy.quantile(0.20)
    sub["shortfall"] = short
    high_hat = pd.Series(False, index=sub.index)
    for (year, crop), idx in sub.groupby(["year", "crop"]).groups.items():
        hh = pd.to_numeric(sub.loc[idx, "exposure_hat"], errors="coerce")
        high_hat.loc[idx] = hh >= hh.quantile(0.80)
    sub["high_hat"] = high_hat

    splits = []
    # IDR piles up at 1.0 among non-producers; do not median-split that mass.
    idr = pd.to_numeric(sub["idr_lag1"], errors="coerce")
    sub["idr_bin"] = np.where(idr >= 0.5, "high", "low")
    for name, mask in (("low", sub["idr_bin"] == "low"), ("high", sub["idr_bin"] == "high"), ("all", idr.notna())):
        splits.append(
            {
                "split": "idr_lag1",
                "stratum": name,
                "threshold": 0.5,
                "n": float(mask.sum()),
                "prevalence": float(sub.loc[mask, "shortfall"].mean()) if mask.any() else np.nan,
                "auprc": _auprc(sub.loc[mask, "shortfall"], sub.loc[mask, "exposure_hat"]) if mask.any() else np.nan,
                "score": "portfolio",
            }
        )
    for col in ("hhi_lag1", "self_sufficiency_lag1", "n_suppliers_lag1"):
        if col not in sub.columns:
            continue
        part = median_split_auprc(sub, col, sub["shortfall"], sub["exposure_hat"])
        part["score"] = "portfolio"
        splits.append(part)
    split_df = pd.concat([pd.DataFrame(splits[:3]), *[s if isinstance(s, pd.DataFrame) else pd.DataFrame([s]) for s in splits[3:]]], ignore_index=True)
    write_table(split_df, OUT / "buffer_split_auprc.csv")
    write_table(sub, OUT / "shortfall_192.csv")

    grid = shortfall_given_exposure(sub.assign(idr_for_grid=idr), "idr_for_grid", "hhi_lag1", sub["shortfall"], sub["high_hat"])
    write_table(grid, OUT / "buffer_grid.csv")

    # Cascade skill at 192 d from existing official numbers + new supply
    suite = read_table(ROOT / "results" / "revision_p2" / "exposure_suite_pooled.csv")
    struct = read_table(ROOT / "results" / "revision_p0" / "trade_structure_by_lead.csv")
    imp_out = read_table(ROOT / "results" / "revision_p0" / "import_outcome.csv")
    p3 = json.loads((ROOT / "results" / "revision_p3" / "summary.json").read_text(encoding="utf-8"))
    prev_y = float(suite[suite["lead"] == 192]["prevalence"].mean())
    auprc_y = float(suite[suite["lead"] == 192]["auprc_forecast"].mean())
    auprc_prod = float(p3["production"]["production_192"])
    auprc_supply = _mean_auprc(supply, "auprc_supply", 192)
    prev_supply = _mean_auprc(supply, "prevalence", 192)
    auprc_short = float(imp_out[imp_out["lead"] == 192]["auprc"].mean()) if len(imp_out) else np.nan
    prev_short = float(imp_out[imp_out["lead"] == 192]["n_shortfall"].sum() / imp_out[imp_out["lead"] == 192]["n_importers"].sum()) if len(imp_out) else 0.20
    cascade = [
        {"stage": "Yield-induced exposure", "auprc": auprc_y, "prevalence": prev_y},
        {"stage": "Supply-system exposure", "auprc": auprc_supply, "prevalence": prev_supply},
        {"stage": "Production-anomaly exposure", "auprc": auprc_prod, "prevalence": prev_y},
        {"stage": "Import-quantity shortfall", "auprc": auprc_short, "prevalence": prev_short},
    ]
    for outcome, label in (
        ("hhi_increase", "HHI increase"),
        ("top1_increase", "Top-supplier share"),
        ("supplier_loss", "Supplier-count loss"),
    ):
        row = struct[(struct["outcome"] == outcome) & (struct["lead"] == 192)]
        if row.empty:
            continue
        cascade.append({"stage": label, "auprc": float(row["auprc"].mean()), "prevalence": 0.20})
    casc = pd.DataFrame(cascade)
    casc["skill"] = [skill_score(a, p) for a, p in zip(casc["auprc"], casc["prevalence"])]
    write_table(casc, OUT / "cascade_skill.csv")

    # Year-block CI ribbons
    ci_rows = []
    for lead in sorted(suite["lead"].unique()):
        for col in ("auprc_forecast", "auprc_n0_lag", "auprc_n3_propensity"):
            ci = year_block_auprc_ci(suite, col, int(lead))
            ci_rows.append({"lead": int(lead), "score": col, **ci})
    write_table(pd.DataFrame(ci_rows), OUT / "auprc_ci_by_lead.csv")

    # Top aggregate-exposure years
    year_tot = (
        packed[packed["lead"] == 192]
        .groupby("year", as_index=False)["exposure_obs"]
        .sum()
        .sort_values("exposure_obs", ascending=False)
    )
    write_table(year_tot, OUT / "year_aggregate_exposure.csv")

    summary = {
        "idr_median": float(acc["idr"].median()),
        "supply_auprc_192": auprc_supply,
        "supply_prevalence_192": prev_supply,
        "portfolio_auprc_192": auprc_y,
        "shortfall_auprc_192": auprc_short,
        "top_exposure_years": [int(y) for y in year_tot["year"].head(3)],
        "buffer_split": split_df.to_dict(orient="records") if len(split_df) else [],
        "buffer_grid": grid.to_dict(orient="records") if len(grid) else [],
        "cascade": casc.to_dict(orient="records"),
    }
    return summary


def run_pooled_permutation(n_perm: int = 400) -> dict:
    """One AUPRC per draw: mean of year–crop pooled-label AUPRCs after shock shuffle."""

    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    food = preds[(preds["experiment"] == "P_food_system") & preds["crop"].isin(CROPS)].copy()
    # Reuse year–crop permutation table: reconstruct a conservative pooled null
    # as the mean of year–crop perm means (already computed). Also store the
    # observed pooled mean. For the histogram, rerun a lighter pooled draw.
    from cropwarn.trade.exposure import import_exposure, observed_exposure_history

    history = observed_exposure_history(labels[labels["crop"].isin(CROPS)], trade, 1)
    shocks = as_supply_shocks(food, labels)
    shocks = shocks[shocks["lead"] == 192].copy()
    rng = np.random.default_rng(42)
    groups = list(shocks.groupby(["year", "crop"]))
    prepared = []
    for (year, crop), g in groups:
        need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
        expo = import_exposure(g[need], trade, int(year), str(crop), 1)
        if expo.empty:
            continue
        expo = expo.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"}).reset_index(drop=True)
        past = history[(history["year"] < int(year)) & (history["crop"] == crop)]
        truth = importer_specific_truth(expo, past, q=0.80, mode="pooled").reset_index(drop=True)
        prepared.append((int(year), str(crop), g, need, expo, truth))
    real_vals = [_auprc(truth, expo["exposure_hat"]) for *_, expo, truth in prepared]
    real = float(np.nanmean(real_vals))
    draws = []
    for _ in range(n_perm):
        vals = []
        for year, crop, g, need, expo, truth in prepared:
            gg = g.copy()
            gg["shock_hat"] = rng.permutation(g["shock_hat"].to_numpy(dtype=float))
            pexp = import_exposure(gg[need], trade, year, crop, 1)
            if pexp.empty:
                continue
            pexp = pexp.rename(columns={"exposure": "exposure_hat"})
            aligned = expo[["importer"]].merge(pexp[["importer", "exposure_hat"]], on="importer", how="left")
            vals.append(_auprc(truth, aligned["exposure_hat"].reset_index(drop=True)))
        if vals:
            draws.append(float(np.nanmean(vals)))
    out = pd.DataFrame({"auprc_perm": draws})
    write_table(out, OUT / "permutation_pooled_draws.csv")
    arr = np.asarray(draws, dtype=float)
    return {
        "observed": real,
        "perm_mean": float(np.nanmean(arr)) if len(arr) else np.nan,
        "perm_p95": float(np.nanquantile(arr, 0.95)) if len(arr) else np.nan,
        "p_ge_obs": float(np.mean(arr >= real)) if len(arr) else np.nan,
        "n_perm": int(len(arr)),
    }


def run_official_ablation() -> None:
    cfg = load_config(ROOT)
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    series = read_table(ROOT / "data_interim" / "obs_season_series.csv")
    panel = labels[labels["crop"].isin(CROPS)].copy()
    series = series[series["crop"].isin(CROPS)].copy()
    leads = [192, 96, 0]
    specs = {
        "climate": ["climate"],
        "optical": ["optical"],
        "optical_climate": ["optical", "climate"],
        "plus_gosif": ["optical", "climate", "sif"],
    }
    rows = []
    for name, mods in specs.items():
        feat = build_feature_table(panel, series, leads, PhenologyVersion.DYNAMIC, mods)
        for lead, block in feat.groupby("lead"):
            block = block[np.isfinite(block["yield_anomaly"])].reset_index(drop=True)
            ys = []
            for split in rolling_origin(block, 2016, 8):
                pred = fit_predict("xgb", block.loc[split.train_idx], block.loc[split.test_idx], "yield_anomaly", True, cfg)
                rec = regression_metrics(block.loc[split.test_idx, "yield_anomaly"].to_numpy(dtype=float), pred)
                ys.append(rec["r2"])
            rows.append({"model": name, "lead": int(lead), "r2": float(np.nanmean(ys)), "n_years": len(ys)})
    write_table(pd.DataFrame(rows), OUT / "official_ablation_r2.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = run_idr_and_buffers()
    print(json.dumps({k: summary[k] for k in summary if k not in ("buffer_split", "buffer_grid", "cascade")}, indent=2))
    print("cascade", summary["cascade"])
    print("buffer_split", summary["buffer_split"])
    print("buffer_grid", summary["buffer_grid"])
    perm = run_pooled_permutation(400)
    summary["permutation"] = perm
    print("permutation", perm)
    run_official_ablation()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
