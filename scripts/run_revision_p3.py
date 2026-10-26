"""P0-2 optical anomaly test and P0-3 FAOSTAT production-shock exposure.

Writes only to results/revision_p3. Does not overwrite official results/.
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
from cropwarn.phenology.alignment import PhenologyVersion
from cropwarn.trade.exposure import as_supply_shocks, evaluate_expost_exposure, import_exposure
from cropwarn.validation.metrics import regression_metrics
from cropwarn.validation.splits import rolling_origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_p3"
OPTICAL_PREFIX = ("ndvi_", "evi_", "lai_")


def _optical_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(OPTICAL_PREFIX)]


def anomalize_fold(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train.copy(), test.copy()
    keys = [c for c in ("iso3", "crop") if c in tr.columns]
    cols = _optical_cols(tr)
    if not cols:
        return tr, te
    mu = tr.groupby(keys)[cols].mean()
    mu.columns = [f"{c}__mu" for c in mu.columns]
    tr = tr.merge(mu, on=keys, how="left")
    te = te.merge(mu, on=keys, how="left")
    for col in cols:
        tr[col] = tr[col] - tr[f"{col}__mu"]
        te[col] = te[col] - te[f"{col}__mu"]
    drop = [f"{c}__mu" for c in cols]
    return tr.drop(columns=drop), te.drop(columns=drop)


def run_optical_anomaly() -> dict:
    cfg = load_config(ROOT)
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    series = read_table(ROOT / "results" / "revision_p1" / "obs_season_series_pre365.csv")
    panel = labels[labels["crop"].isin(["wheat", "soybean"])].copy()
    series = series[series["crop"].isin(["wheat", "soybean"])].copy()
    leads = [365, 192, 96, 0]
    feat = build_feature_table(panel, series, leads, PhenologyVersion.DYNAMIC, ["optical"])
    rows = []
    for lead, block in feat.groupby("lead"):
        block = block[np.isfinite(block["yield_anomaly"])].reset_index(drop=True)
        for split in rolling_origin(block, 2016, 8):
            train = block.loc[split.train_idx]
            test = block.loc[split.test_idx]
            raw_pred = fit_predict("xgb", train, test, "yield_anomaly", True, cfg)
            tr, te = anomalize_fold(train, test)
            ano_pred = fit_predict("xgb", tr, te, "yield_anomaly", True, cfg)
            y = test["yield_anomaly"].to_numpy(dtype=float)
            rec_raw = regression_metrics(y, raw_pred)
            rec_ano = regression_metrics(y, ano_pred)
            rows.append(
                {
                    "lead": int(lead),
                    "year": int(split.test_year),
                    "r2_raw": rec_raw["r2"],
                    "r2_anomaly": rec_ano["r2"],
                    "n": rec_raw["n"],
                }
            )
    out = pd.DataFrame(rows)
    by = out.groupby("lead", as_index=False)[["r2_raw", "r2_anomaly", "n"]].mean()
    write_table(out, OUT / "optical_anomaly_by_year.csv")
    write_table(by, OUT / "optical_anomaly_by_lead.csv")
    return {"by_lead": by.to_dict("records")}


def production_shock_labels(labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    if "production_t" not in lab.columns:
        lab["production_t"] = pd.to_numeric(lab["yield_tha"], errors="coerce") * pd.to_numeric(
            lab["harvested_ha"], errors="coerce"
        )
    anom = rolling_trend_anomaly(lab, value_col="production_t", method="percent")
    out = anom[["iso3", "crop", "year"]].copy()
    out["prod_shock"] = pd.to_numeric(anom["yield_anomaly"], errors="coerce")
    return out


def evaluate_production_expost(pred_panel: pd.DataFrame, trade: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Yield-based predicted exposure vs FAOSTAT production-anomaly observed exposure."""

    prod = production_shock_labels(labels)
    shocks = as_supply_shocks(pred_panel, labels)
    if "shock_hat" not in shocks.columns and "pred" in shocks.columns:
        shocks = shocks.assign(shock_hat=shocks["pred"])
    pieces = []
    for (year, crop, lead), g in shocks.groupby(["year", "crop", "lead"]):
        need = [c for c in ("iso3", "year", "crop", "lead", "shock_hat", "shock_true") if c in g.columns]
        expo = import_exposure(g[need], trade, int(year), str(crop), 1)
        if expo.empty:
            continue
        obs = prod[(prod["year"] == int(year)) & (prod["crop"] == crop)][["iso3", "prod_shock"]]
        w = trade[(trade["year"] == int(year) - 1) & (trade["crop"] == crop)]
        ww = w.merge(obs, left_on="exporter", right_on="iso3", how="left")
        ww["flow"] = pd.to_numeric(ww["export_share"], errors="coerce") * np.clip(
            -pd.to_numeric(ww["prod_shock"], errors="coerce"), 0.0, None
        )
        truth = ww.groupby("importer", as_index=False)["flow"].sum().rename(columns={"flow": "exposure_obs"})
        expo = expo.rename(columns={"exposure": "exposure_hat"}).drop(columns=["flow_true"], errors="ignore")
        expo = expo.merge(truth, on="importer", how="left")
        expo["year"] = int(year)
        pieces.append(expo)
    if not pieces:
        return pd.DataFrame()
    all_expo = pd.concat(pieces, ignore_index=True)
    from sklearn.metrics import average_precision_score

    rows = []
    for (year, crop, lead), g in all_expo.groupby(["year", "crop", "lead"]):
        past = all_expo[(all_expo["year"] < int(year)) & (all_expo["crop"] == crop)]
        thresh = float(past["exposure_obs"].quantile(0.80)) if len(past) and past["exposure_obs"].notna().any() else np.nan
        hat = pd.to_numeric(g["exposure_hat"], errors="coerce")
        obs = pd.to_numeric(g["exposure_obs"], errors="coerce")
        truth = obs > thresh if np.isfinite(thresh) else pd.Series(False, index=g.index)
        auprc = np.nan
        m = hat.notna() & obs.notna()
        if int(m.sum()) >= 8 and int(truth[m].sum()) not in (0, int(m.sum())):
            auprc = float(average_precision_score(truth[m].astype(int), hat[m]))
        from scipy.stats import spearmanr

        rho = np.nan
        if int(m.sum()) >= 8:
            try:
                rho = float(spearmanr(hat[m], obs[m]).statistic)
            except Exception:
                rho = np.nan
        rows.append(
            {
                "year": int(year),
                "crop": crop,
                "lead": int(lead),
                "auprc": auprc,
                "spearman": rho,
                "n_high": float(truth.sum()),
                "n": float(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def run_production_exposure() -> dict:
    preds = read_table(ROOT / "results" / "predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    food = preds[preds["experiment"] == "P_food_system"].copy()
    yield_ex = evaluate_expost_exposure(food, trade, labels)
    prod_ex = evaluate_production_expost(food, trade, labels)
    write_table(prod_ex, OUT / "production_expost_exposure.csv")
    yby = yield_ex.groupby("lead", as_index=False)[["auprc", "cont_spearman"]].mean()
    pby = prod_ex.groupby("lead", as_index=False)[["auprc", "spearman"]].mean()
    write_table(pby, OUT / "production_expost_by_lead.csv")
    return {
        "yield_induced_192": float(yby.loc[yby["lead"] == 192, "auprc"].mean()) if len(yby) else None,
        "production_192": float(pby.loc[pby["lead"] == 192, "auprc"].mean()) if (pby["lead"] == 192).any() else None,
        "production_0": float(pby.loc[pby["lead"] == 0, "auprc"].mean()) if (pby["lead"] == 0).any() else None,
        "production_by_lead": pby.to_dict("records"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"production": run_production_exposure()}
    print(json.dumps(summary, indent=2, default=str), flush=True)
    summary["optical_anomaly"] = run_optical_anomaly()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
