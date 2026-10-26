"""Pass climate-only, optical-only and combined yield refits through the trade network.

Scores the archived test rows in results/as_revision/source_predictions.csv.
Does not retrain models and does not overwrite results/predictions.csv.
The headline 0.71 / 0.57 figures come from the food-system specification, not from this refit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cropwarn.data.io import read_table, write_table

spec = importlib.util.spec_from_file_location("run_as_revision", ROOT / "scripts" / "run_as_revision.py")
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)

OUT = ROOT / "results" / "as_revision"
MODELS = ("climate", "optical", "optical_climate")


def year_mean_auprc(long: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    for (year, crop), g in long.groupby(["year", "crop"]):
        y = g["truth"].astype(int)
        s = pd.to_numeric(g[score], errors="coerce")
        m = y.notna() & s.notna()
        ap = float(average_precision_score(y[m], s[m])) if m.sum() >= 8 and y[m].nunique() > 1 else np.nan
        rows.append({"year": int(year), "crop": str(crop), "ap": ap})
    detail = pd.DataFrame(rows)
    year = detail.groupby("year", as_index=False)["ap"].mean()
    return detail, year


def paired(year_hat: pd.Series, year_base: pd.Series) -> dict:
    delta = (year_hat - year_base).dropna()
    ci = rev.year_block_stat(delta)
    return {"delta": ci["mean"], "p025": ci["p025"], "p975": ci["p975"], "n_years": int(delta.size)}


def r2_192(frame: pd.DataFrame) -> float:
    y = pd.to_numeric(frame["yield_anomaly"], errors="coerce")
    yhat = pd.to_numeric(frame["yhat"], errors="coerce")
    m = y.notna() & yhat.notna()
    if int(m.sum()) < 12:
        return float("nan")
    return float(r2_score(y[m], yhat[m]))


def main() -> None:
    src = read_table(OUT / "source_predictions.csv")
    labels = read_table(ROOT / "data_interim" / "obs_panel_labels.csv")
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    labels = labels[labels["crop"].isin(["wheat", "soybean"])].copy()
    trade = trade[trade["crop"].isin(["wheat", "soybean"])].copy()
    prod = rev.production_obs_shocks(labels)
    rows = []
    prec_rows = []
    for name in MODELS:
        block = src[(src["model"] == name) & (src["lead"] == 192)].copy()
        block["pred"] = pd.to_numeric(block["yhat"], errors="coerce")
        block["shock_true"] = pd.to_numeric(block["yield_anomaly"], errors="coerce")
        print("score", name, "n", len(block), flush=True)
        for endpoint, shocks in (("yield", None), ("production", prod)):
            long = rev.build_long_192(block, trade, labels, shocks, 1)
            detail, year = year_mean_auprc(long, "exposure_hat")
            _, year_n0 = year_mean_auprc(long, "n0_lag")
            both = year.merge(year_n0, on="year", suffixes=("_hat", "_n0"))
            gain = paired(both["ap_hat"], both["ap_n0"])
            ap_ci = rev.year_block_stat(both["ap_hat"])
            tk = rev.topk_table(long, "exposure_hat", ks=(5,))
            tk_n0 = rev.topk_table(long, "n0_lag", ks=(5,))
            prec = tk.groupby("year")["precision"].mean()
            prec_n0 = tk_n0.groupby("year")["precision"].mean()
            prec_ci = rev.year_block_stat(prec)
            dprec = paired(prec, prec_n0.reindex(prec.index))
            rows.append(
                {
                    "model": name,
                    "endpoint": endpoint,
                    "r2_192": r2_192(block),
                    "n_192": int(block[["iso3", "crop", "year"]].drop_duplicates().shape[0]),
                    "auprc": float(both["ap_hat"].mean()),
                    "auprc_p025": ap_ci["p025"],
                    "auprc_p975": ap_ci["p975"],
                    "n0": float(both["ap_n0"].mean()),
                    "delta_n0": gain["delta"],
                    "delta_n0_p025": gain["p025"],
                    "delta_n0_p975": gain["p975"],
                    "precision_at_5": prec_ci["mean"],
                    "precision_at_5_p025": prec_ci["p025"],
                    "precision_at_5_p975": prec_ci["p975"],
                    "precision_n0": float(prec_n0.mean()),
                    "delta_precision_at_5": dprec["delta"],
                    "delta_precision_p025": dprec["p025"],
                    "delta_precision_p975": dprec["p975"],
                }
            )
            detail.assign(model=name, endpoint=endpoint).to_csv(
                OUT / f"source_exposure_{name}_{endpoint}.csv", index=False
            )
            prec_rows.append(tk.assign(model=name, endpoint=endpoint))
            print(name, endpoint, rows[-1]["auprc"], rows[-1]["delta_n0"], rows[-1]["precision_at_5"], flush=True)
    out = pd.DataFrame(rows)
    write_table(out, OUT / "source_exposure_summary.csv")
    (OUT / "source_exposure_summary.json").write_text(out.to_json(orient="records", indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
