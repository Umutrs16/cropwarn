"""P0 checks requested after the Agricultural Systems review.

1. Year-mean permutation (same estimand as the headline AUPRC).
2. Predict with the t−2 network, score the t−1 ex-post exposure target.
3. Climate / optical / combined skill by phenological stage and by
   whether the 192-day cutoff is still before sowing.
4. High-VPD and major-exporter splits on the existing optical–climate forecasts.
5. Share of national harvested area sitting in the 16 CROPGRIDS cells.

Does not re-extract satellite samples and does not overwrite results/predictions.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.data.io import read_table, write_table
from cropwarn.data.labels import rolling_trend_anomaly
from cropwarn.features.factory import build_feature_table
from cropwarn.models.predictors import fit_predict
from cropwarn.phenology.alignment import PhenologyVersion, harvest_doy, season_progress_signed, stage_name
from cropwarn.trade.baselines import (
    _auprc,
    _exposure_from_weights,
    importer_specific_truth,
    network_baseline_scores,
    pooled_shock_permutation,
)
from cropwarn.trade.exposure import as_supply_shocks, observed_exposure_history, overlay_observed_shocks
from cropwarn.validation.metrics import regression_metrics
from cropwarn.validation.splits import rolling_origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "as_revision"
CROPS = ("wheat", "soybean")
N_PERM = 2000


def production_obs_shocks(labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    if "production_t" not in lab.columns:
        lab["production_t"] = pd.to_numeric(lab["yield_tha"], errors="coerce") * pd.to_numeric(
            lab["harvested_ha"], errors="coerce"
        )
    anom = rolling_trend_anomaly(lab, value_col="production_t", method="percent")
    out = anom[["iso3", "crop", "year"]].copy()
    out["shock_obs"] = pd.to_numeric(anom["yield_anomaly"], errors="coerce")
    return out


def year_block(values: pd.Series, n_boot: int = 2000, seed: int = 42) -> dict:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if arr.size == 0:
        return {"mean": np.nan, "p025": np.nan, "p975": np.nan}
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(n_boot)]
    s = pd.Series(draws)
    return {"mean": float(arr.mean()), "p025": float(s.quantile(0.025)), "p975": float(s.quantile(0.975))}


def cross_network(food, trade, labels, obs_shocks, pred_lag: int, label_lag: int) -> pd.DataFrame:
    """Predicted exposure uses pred_lag; the ex-post target and its history use label_lag."""

    history = observed_exposure_history(labels, trade, label_lag, obs_shocks=obs_shocks)
    shocks = as_supply_shocks(food[food["lead"] == 192], labels)
    if obs_shocks is not None:
        shocks = overlay_observed_shocks(shocks, obs_shocks)
    rows = []
    for (year, crop), g in shocks.groupby(["year", "crop"]):
        g = g.drop_duplicates("iso3")
        hat = g.set_index("iso3")["shock_hat"]
        true = g.set_index("iso3")["shock_true"]
        w_hat = trade[(trade["year"] == int(year) - pred_lag) & (trade["crop"] == crop)]
        w_obs = trade[(trade["year"] == int(year) - label_lag) & (trade["crop"] == crop)]
        if w_hat.empty or w_obs.empty:
            continue
        expo_hat = _exposure_from_weights(w_hat, hat)
        expo_obs = _exposure_from_weights(w_obs, true)
        both = expo_hat.index.intersection(expo_obs.index)
        if len(both) < 8:
            continue
        expo = pd.DataFrame(
            {
                "importer": both,
                "exposure_hat": expo_hat.reindex(both).to_numpy(),
                "exposure_obs": expo_obs.reindex(both).to_numpy(),
            }
        )
        past = history[(history["year"] < int(year)) & (history["crop"] == crop)] if len(history) else pd.DataFrame()
        truth = importer_specific_truth(expo, past, q=0.80, mode="pooled")
        base = network_baseline_scores(expo, history, trade, int(year), str(crop), label_lag, labels=labels)
        merged = expo.merge(base, on="importer", how="left")
        rows.append(
            {
                "year": int(year),
                "crop": str(crop),
                "pred_lag": pred_lag,
                "label_lag": label_lag,
                "n": int(len(merged)),
                "prevalence": float(np.mean(truth)),
                "auprc_forecast": _auprc(truth, merged["exposure_hat"]),
                "auprc_n0": _auprc(truth, merged["n0_lag"]) if "n0_lag" in merged.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_cross(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    year = df.groupby("year", as_index=False)[["auprc_forecast", "auprc_n0", "prevalence"]].mean()
    year["delta_n0"] = year["auprc_forecast"] - year["auprc_n0"]
    ap = year_block(year["auprc_forecast"])
    d0 = year_block(year["delta_n0"])
    return {
        "auprc": ap["mean"],
        "auprc_p025": ap["p025"],
        "auprc_p975": ap["p975"],
        "n0": float(year["auprc_n0"].mean()),
        "prevalence": float(year["prevalence"].mean()),
        "delta_n0": d0["mean"],
        "delta_n0_p025": d0["p025"],
        "delta_n0_p975": d0["p975"],
        "n_years": int(len(year)),
    }


def _r2(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 12:
        return float("nan")
    rec = regression_metrics(y[m], yhat[m])
    return float(rec["r2"])


def attach_stage(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    meta = panel[["iso3", "crop", "year", "sos_true", "eos_true", "sos_static", "eos_static"]].drop_duplicates()
    p = preds.merge(meta, on=["iso3", "crop", "year"], how="left")
    sos = pd.to_numeric(p["sos_true"], errors="coerce").fillna(pd.to_numeric(p["sos_static"], errors="coerce"))
    eos = pd.to_numeric(p["eos_true"], errors="coerce").fillna(pd.to_numeric(p["eos_static"], errors="coerce"))
    progress = []
    for s, e, lead in zip(sos.to_numpy(), eos.to_numpy(), p["lead"].to_numpy()):
        if not np.isfinite(s) or not np.isfinite(e) or not np.isfinite(lead):
            progress.append(np.nan)
            continue
        cutoff = harvest_doy(int(e)) - int(lead)
        if cutoff < 1:
            cutoff += 365
        progress.append(season_progress_signed(int(cutoff), int(s), int(e)))
    p["progress"] = progress
    p["stage"] = [stage_name(v) if np.isfinite(v) else "unknown" for v in p["progress"]]
    p["timing"] = np.where(p["progress"] < 0, "pre_sowing", np.where(p["progress"].notna(), "in_season", "unknown"))
    return p


def source_ablation(panel: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config(ROOT)
    leads = [192, 160, 128, 96, 64, 32, 0]
    specs = {
        "climate": ["climate"],
        "optical": ["optical"],
        "optical_climate": ["optical", "climate"],
    }
    pieces = []
    for name, mods in specs.items():
        print("features", name, flush=True)
        feat = build_feature_table(panel, series, leads, PhenologyVersion.DYNAMIC, mods)
        for lead, block in feat.groupby("lead"):
            block = block[np.isfinite(pd.to_numeric(block["yield_anomaly"], errors="coerce"))].reset_index(drop=True)
            for split in rolling_origin(block, 2016, 8):
                pred = fit_predict("xgb", block.loc[split.train_idx], block.loc[split.test_idx], "yield_anomaly", True, cfg)
                test = block.loc[split.test_idx, ["iso3", "crop", "year", "lead", "yield_anomaly"]].copy()
                test["yhat"] = pred
                test["model"] = name
                pieces.append(test)
        print("fit", name, flush=True)
    return pd.concat(pieces, ignore_index=True)


def group_r2(preds: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in preds.groupby(cols):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(cols, key))
        rec["n"] = int(len(g))
        rec["r2"] = _r2(g["yield_anomaly"], g["yhat"])
        rows.append(rec)
    return pd.DataFrame(rows)


def vpd_and_exporter_splits(preds: pd.DataFrame, labels: pd.DataFrame, series: pd.DataFrame, trade: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = preds[(preds["experiment"] == "B2_optical_climate") & (preds["lead"] == 192) & preds["crop"].isin(CROPS)].copy()
    vpd = (
        series[series["crop"].isin(CROPS)]
        .groupby(["iso3", "crop", "year"], as_index=False)["vpd"]
        .mean()
    )
    base = vpd[vpd["year"].between(2001, 2015)].groupby(["iso3", "crop"])["vpd"].median().rename("vpd_ref")
    vpd = vpd.merge(base, on=["iso3", "crop"], how="left")
    vpd["vpd_group"] = np.where(vpd["vpd"] >= vpd["vpd_ref"], "high_vpd", "low_vpd")
    p = p.merge(vpd[["iso3", "crop", "year", "vpd_group"]], on=["iso3", "crop", "year"], how="left")
    p = p.rename(columns={"pred": "yhat"})
    stress = group_r2(p.dropna(subset=["vpd_group"]), ["crop", "vpd_group"])

    exp = trade[trade["crop"].isin(CROPS) & trade["year"].between(2001, 2015)]
    qty = exp.groupby(["exporter", "crop"], as_index=False)["qty"].mean().rename(columns={"exporter": "iso3", "qty": "export_qty"})
    qty["rank"] = qty.groupby("crop")["export_qty"].rank(pct=True)
    qty["tier"] = np.where(qty["rank"] >= 0.80, "major_exporter", "other")
    p = p.merge(qty[["iso3", "crop", "tier"]], on=["iso3", "crop"], how="left")
    p["tier"] = p["tier"].fillna("other")
    tiers = group_r2(p, ["crop", "tier"])
    return stress, tiers


def area_coverage(labels: pd.DataFrame) -> pd.DataFrame:
    pts = read_table(ROOT / "data_interim" / "crop_points.csv")
    pts = pts[pts["crop"].isin(CROPS) & (pts["source"] == "cropgrids2020")]
    cells = pts.groupby(["iso3", "crop"], as_index=False).agg(n_cells=("weight", "size"), cell_ha=("weight", "sum"))
    fao = labels[(labels["crop"].isin(CROPS)) & (labels["year"] == 2020)][["iso3", "crop", "harvested_ha"]].drop_duplicates()
    out = cells.merge(fao, on=["iso3", "crop"], how="left")
    out["share_of_2020_harvested_area"] = out["cell_ha"] / pd.to_numeric(out["harvested_ha"], errors="coerce")
    return out.sort_values(["crop", "iso3"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    series = read_table(ROOT / "data_interim" / "obs_season_series.csv")
    food = preds[(preds["experiment"] == "P_food_system") & preds["crop"].isin(CROPS)].copy()
    labels_ws = labels[labels["crop"].isin(CROPS)].copy()
    trade_ws = trade[trade["crop"].isin(CROPS)].copy()
    series_ws = series[series["crop"].isin(CROPS)].copy()
    prod = production_obs_shocks(labels_ws)

    summary = {}
    for tag, shocks in (("yield", None), ("production", prod)):
        print("permutation", tag, flush=True)
        draws, info = pooled_shock_permutation(
            food, trade_ws, labels_ws, lead=192, n_perm=N_PERM, obs_shocks=shocks, aggregate="year_mean"
        )
        write_table(draws, OUT / f"perm_{tag}_yearmean.csv")
        summary[f"perm_{tag}_yearmean"] = info
        print(tag, info, flush=True)

    cross_rows = []
    cross_sum = {}
    for tag, shocks in (("yield", None), ("production", prod)):
        for pred_lag, label_lag, name in ((2, 1, "predict_t2_label_t1"), (2, 2, "both_t2"), (1, 1, "both_t1")):
            scored = cross_network(food, trade_ws, labels_ws, shocks, pred_lag, label_lag)
            scored["endpoint"] = tag
            scored["design"] = name
            cross_rows.append(scored)
            cross_sum[f"{tag}_{name}"] = summarize_cross(scored)
            print(tag, name, cross_sum[f"{tag}_{name}"], flush=True)
    write_table(pd.concat(cross_rows, ignore_index=True), OUT / "cross_network_year.csv")
    summary["cross_network"] = cross_sum

    stress, tiers = vpd_and_exporter_splits(preds, labels_ws, series_ws, trade_ws)
    write_table(stress, OUT / "vpd_split_r2.csv")
    write_table(tiers, OUT / "exporter_tier_r2.csv")
    cover = area_coverage(labels_ws)
    write_table(cover, OUT / "sample_area_coverage.csv")
    summary["area_coverage_median_share"] = {
        crop: float(g["share_of_2020_harvested_area"].median()) for crop, g in cover.groupby("crop")
    }
    summary["vpd_split"] = stress.to_dict(orient="records")
    summary["exporter_tier"] = tiers.to_dict(orient="records")

    raw = source_ablation(labels_ws, series_ws)
    staged = attach_stage(raw, labels_ws)
    write_table(staged, OUT / "source_predictions.csv")
    by_stage = group_r2(staged, ["model", "stage"])
    by_time = group_r2(staged[staged["lead"] == 192], ["model", "timing"])
    write_table(by_stage, OUT / "source_by_stage.csv")
    write_table(by_time, OUT / "source_by_timing_192.csv")
    summary["source_by_timing_192"] = by_time.to_dict(orient="records")
    (OUT / "p0b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
