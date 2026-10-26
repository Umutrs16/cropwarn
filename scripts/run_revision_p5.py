"""P0: exporter-weighted skill, leave-one-major-exporter-out, cascade CIs.

Writes only to results/revision_p5/.
"""

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
from cropwarn.trade.baselines import _auprc, importer_specific_truth
from cropwarn.trade.buffers import skill_score
from cropwarn.trade.exposure import as_supply_shocks, import_exposure, observed_exposure_history
from cropwarn.validation.metrics import regression_metrics
from cropwarn.validation.splits import rolling_origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_p5"
CROPS = ("wheat", "soybean")


def _boot_mean(vals: np.ndarray, n: int = 400, seed: int = 42) -> dict[str, float]:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p50": np.nan, "p05": np.nan, "p95": np.nan}
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(x, size=len(x), replace=True))) for _ in range(n)]
    s = pd.Series(draws)
    return {"p50": float(s.median()), "p05": float(s.quantile(0.05)), "p95": float(s.quantile(0.95))}


def exporter_shock_skill(food: pd.DataFrame, labels: pd.DataFrame, trade: pd.DataFrame) -> dict:
    shocks = as_supply_shocks(food[food["lead"] == 192], labels)
    g = shocks[np.isfinite(shocks["shock_hat"]) & np.isfinite(shocks["shock_true"])].copy()
    exp = (
        trade[trade["crop"].isin(CROPS)]
        .groupby(["year", "crop", "exporter"], as_index=False)["qty"]
        .sum()
        .rename(columns={"exporter": "iso3", "qty": "export_qty"})
    )
    g = g.merge(exp, on=["year", "crop", "iso3"], how="left")
    g["export_qty"] = pd.to_numeric(g["export_qty"], errors="coerce").fillna(0.0)
    prod = pd.to_numeric(g.get("prod_lag1", g.get("production_t")), errors="coerce").fillna(0.0)
    y = pd.to_numeric(g["shock_true"], errors="coerce")
    hat = pd.to_numeric(g["shock_hat"], errors="coerce")
    severe = y <= y.quantile(0.20)
    auprc_neg = float(average_precision_score(severe.astype(int), (-hat).to_numpy()))
    sign = float(((hat < 0) == (y < 0)).mean())
    from scipy.stats import spearmanr

    rho = float(spearmanr(hat, y).statistic)
    w = prod.to_numpy(dtype=float)
    if w.sum() > 0:
        # weighted rank: correlate hat and y after repeating is heavy; use weighted residual
        w = w / w.sum()
        rho_w = float(np.corrcoef(hat, y)[0, 1])  # fallback
        # production-weighted Spearman via weighted ranks
        def wspear(a, b, ww):
            ra = pd.Series(a).rank().to_numpy()
            rb = pd.Series(b).rank().to_numpy()
            mu_a = np.average(ra, weights=ww)
            mu_b = np.average(rb, weights=ww)
            num = np.average((ra - mu_a) * (rb - mu_b), weights=ww)
            den = np.sqrt(np.average((ra - mu_a) ** 2, weights=ww) * np.average((rb - mu_b) ** 2, weights=ww))
            return float(num / den) if den > 0 else np.nan

        rho_prod = wspear(hat, y, np.clip(prod, 0, None) + 1e-9)
        rho_exp = wspear(hat, y, g["export_qty"].to_numpy(dtype=float) + 1e-9)
        rmse_exp = float(np.sqrt(np.average((hat - y) ** 2, weights=g["export_qty"] + 1e-9)))
        rmse = float(np.sqrt(np.mean((hat - y) ** 2)))
    else:
        rho_prod = rho_exp = rmse_exp = np.nan
        rmse = float(np.sqrt(np.mean((hat - y) ** 2)))
        rho_w = rho
    # Major exporters: top 20% of export qty
    big = g["export_qty"] >= g["export_qty"].quantile(0.80)
    auprc_big = np.nan
    if big.sum() >= 8 and severe[big].nunique() == 2:
        auprc_big = float(average_precision_score(severe[big].astype(int), (-hat[big]).to_numpy()))
    rec_severe = float(((hat < 0) & severe).sum() / max(severe.sum(), 1))
    out = {
        "n": int(len(g)),
        "r2_unweighted": float(1 - np.sum((hat - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)),
        "spearman": rho,
        "spearman_production_weighted": rho_prod,
        "spearman_export_weighted": rho_exp,
        "rmse": rmse,
        "rmse_export_weighted": rmse_exp,
        "sign_accuracy": sign,
        "severe_neg_auprc": auprc_neg,
        "severe_neg_prevalence": float(severe.mean()),
        "severe_neg_recall": rec_severe,
        "major_exporter_severe_auprc": auprc_big,
        "n_major_exporters": int(big.sum()),
    }
    write_table(g[["iso3", "crop", "year", "shock_hat", "shock_true", "export_qty"]], OUT / "exporter_shocks_192.csv")
    return out


def leave_one_exporter_out(food: pd.DataFrame, labels: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    history = observed_exposure_history(labels[labels["crop"].isin(CROPS)], trade, 1)
    shocks = as_supply_shocks(food[food["lead"] == 192], labels)
    exp_tot = (
        trade[trade["crop"].isin(CROPS)]
        .groupby("exporter")["qty"]
        .sum()
        .sort_values(ascending=False)
    )
    majors = list(exp_tot.head(8).index)
    rows = []
    # full baseline
    full_vals = []
    for (year, crop), g in shocks.groupby(["year", "crop"]):
        need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
        expo = import_exposure(g[need], trade, int(year), str(crop), 1)
        if expo.empty:
            continue
        expo = expo.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"})
        past = history[(history["year"] < int(year)) & (history["crop"] == crop)]
        truth = importer_specific_truth(expo, past, q=0.80, mode="pooled")
        full_vals.append(_auprc(truth, expo["exposure_hat"]))
    full = float(np.nanmean(full_vals))
    rows.append({"dropped": "none", "auprc": full, "delta": 0.0, "export_qty_share": 1.0})
    total_qty = float(exp_tot.sum())
    for j in majors:
        vals = []
        gg_all = shocks.copy()
        gg_all.loc[gg_all["iso3"] == j, "shock_hat"] = 0.0
        for (year, crop), g in gg_all.groupby(["year", "crop"]):
            need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
            expo = import_exposure(g[need], trade, int(year), str(crop), 1)
            if expo.empty:
                continue
            expo = expo.rename(columns={"exposure": "exposure_hat", "flow_true": "exposure_obs"})
            past = history[(history["year"] < int(year)) & (history["crop"] == crop)]
            truth = importer_specific_truth(expo, past, q=0.80, mode="pooled")
            vals.append(_auprc(truth, expo["exposure_hat"]))
        a = float(np.nanmean(vals)) if vals else np.nan
        rows.append(
            {
                "dropped": j,
                "auprc": a,
                "delta": full - a,
                "export_qty_share": float(exp_tot.loc[j] / total_qty) if j in exp_tot.index else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    write_table(out, OUT / "leave_one_exporter.csv")
    return out


def cascade_with_ci() -> pd.DataFrame:
    suite = read_table(ROOT / "results" / "revision_p2" / "exposure_suite_pooled.csv")
    prod = read_table(ROOT / "results" / "revision_p3" / "production_expost_exposure.csv")
    short = read_table(ROOT / "results" / "revision_p0" / "import_outcome.csv")
    y = suite[suite["lead"] == 192][["year", "crop", "auprc_forecast", "prevalence"]].copy()
    y["stage"] = "yield_induced"
    y["auprc"] = y["auprc_forecast"]
    p = prod[prod["lead"] == 192][["year", "crop", "auprc", "n_high", "n"]].copy()
    p["prevalence"] = p["n_high"] / p["n"].replace(0, np.nan)
    p["stage"] = "production"
    s = short[short["lead"] == 192][["year", "crop", "auprc", "n_shortfall", "n_importers"]].copy()
    s["prevalence"] = s["n_shortfall"] / s["n_importers"].replace(0, np.nan)
    s["stage"] = "shortfall"
    long = pd.concat(
        [
            y[["year", "crop", "stage", "auprc", "prevalence"]],
            p[["year", "crop", "stage", "auprc", "prevalence"]],
            s[["year", "crop", "stage", "auprc", "prevalence"]],
        ],
        ignore_index=True,
    )
    long["skill"] = [skill_score(a, pr) for a, pr in zip(long["auprc"], long["prevalence"])]
    write_table(long, OUT / "cascade_year_crop.csv")

    years = sorted(long["year"].unique())
    rng = np.random.default_rng(42)
    recs = []
    for stage in ("yield_induced", "production", "shortfall"):
        sub = long[long["stage"] == stage]
        by_year = sub.groupby("year")["skill"].mean()
        ci = _boot_mean(by_year.to_numpy())
        recs.append(
            {
                "stage": stage,
                "skill": float(by_year.mean()),
                "auprc": float(sub.groupby("year")["auprc"].mean().mean()),
                "prevalence": float(sub.groupby("year")["prevalence"].mean().mean()),
                **ci,
            }
        )
    # pairwise year-block
    ymap = {st: long[long["stage"] == st].groupby("year")["skill"].mean() for st in ("yield_induced", "production", "shortfall")}
    pairs = []
    for a, b in (("yield_induced", "production"), ("production", "shortfall"), ("yield_induced", "shortfall")):
        common = sorted(set(ymap[a].index) & set(ymap[b].index))
        diffs = []
        for _ in range(400):
            pick = rng.choice(common, size=len(common), replace=True)
            diffs.append(float(np.mean([ymap[a].loc[y] - ymap[b].loc[y] for y in pick])))
        ds = pd.Series(diffs)
        pairs.append(
            {
                "contrast": f"{a}-{b}",
                "delta": float(np.mean([ymap[a].loc[y] - ymap[b].loc[y] for y in common])),
                "p05": float(ds.quantile(0.05)),
                "p95": float(ds.quantile(0.95)),
            }
        )
    write_table(pd.DataFrame(recs), OUT / "cascade_skill_ci.csv")
    write_table(pd.DataFrame(pairs), OUT / "cascade_pairwise.csv")
    return pd.DataFrame(recs), pd.DataFrame(pairs)


def run_full_ablation() -> None:
    cfg = load_config(ROOT)
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    series = read_table(ROOT / "data_interim" / "obs_season_series.csv")
    panel = labels[labels["crop"].isin(CROPS)].copy()
    series = series[series["crop"].isin(CROPS)].copy()
    leads = [192, 160, 128, 96, 64, 32, 0]
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
                pred = fit_predict(
                    "xgb",
                    block.loc[split.train_idx],
                    block.loc[split.test_idx],
                    "yield_anomaly",
                    True,
                    cfg,
                )
                rec = regression_metrics(block.loc[split.test_idx, "yield_anomaly"].to_numpy(dtype=float), pred)
                ys.append(rec["r2"])
            rows.append({"model": name, "lead": int(lead), "r2": float(np.nanmean(ys)), "n_years": len(ys)})
    write_table(pd.DataFrame(rows), OUT / "official_ablation_full.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    food = preds[(preds["experiment"] == "P_food_system") & preds["crop"].isin(CROPS)].copy()
    labels = labels[labels["crop"].isin(CROPS)].copy()
    trade = trade[trade["crop"].isin(CROPS)].copy()

    exp_sk = exporter_shock_skill(food, labels, trade)
    print("exporter_skill", json.dumps(exp_sk, indent=2))
    loo = leave_one_exporter_out(food, labels, trade)
    print(loo.to_string(index=False))
    casc, pairs = cascade_with_ci()
    print(casc.to_string(index=False))
    print(pairs.to_string(index=False))
    run_full_ablation()
    summary = {
        "exporter_skill": exp_sk,
        "leave_one_exporter": loo.to_dict(orient="records"),
        "cascade": casc.to_dict(orient="records"),
        "pairwise": pairs.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
