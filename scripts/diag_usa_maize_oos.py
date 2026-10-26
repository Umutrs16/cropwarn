"""Check shock-year R² and a one-feature JJA SIF rolling-origin ceiling."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from cropwarn.data.labels import rolling_trend_anomaly
from cropwarn.validation.metrics import earliest_reliable_lead, regression_metrics

root = Path(__file__).resolve().parents[1]
preds = pd.read_csv(root / "results" / "usa_maize" / "predictions.csv")
panel = pd.read_csv(root / "data_interim" / "usa_maize.csv")
panel = rolling_trend_anomaly(panel)

# Shock-year / |z|>=1 R² by experiment and lead
p = preds.merge(panel[["unit_id", "year", "yield_z"]], on=["unit_id", "year"], how="left")
mask = p["year"].eq(2012) | (p["yield_z"].abs() >= 1)
sub = p[mask]
target = "y" if "y" in sub.columns else "yield_anomaly"
print("shock rows", len(sub), "years", sorted(sub["year"].unique()), "target", target, "cols", list(preds.columns)[:20])
rows = []
for (exp, lead), g in sub.groupby(["experiment", "lead"]):
    rec = regression_metrics(g[target].to_numpy(), g["pred"].to_numpy())
    rec.update({"experiment": exp, "lead": int(lead)})
    rows.append(rec)
sk = pd.DataFrame(rows)
for exp in ("B2_optical_climate", "B4_plus_sif"):
    part = sk[sk["experiment"] == exp].sort_values("lead")
    print(exp)
    print(part[["lead", "r2", "n"]].to_string(index=False))
    print(" earliest", earliest_reliable_lead(part))

# All-year official-style R² already known negative. One-feature SIF ceiling.
g = pd.read_csv(root / "data_interim" / "gosif_usa_maize.csv", parse_dates=["date"])
g["month"] = g["date"].dt.month
g["year"] = g["date"].dt.year
sif = g[g["month"].isin([7, 8])].groupby(["unit_id", "year"], as_index=False)["sif"].mean()
lab = panel.merge(sif, on=["unit_id", "year"]).dropna(subset=["yield_anomaly", "sif"])
oos = []
for year in range(2016, 2025):
    tr = lab[lab["year"] < year]
    te = lab[lab["year"] == year]
    if len(tr) < 20 or len(te) < 5:
        continue
    model = LinearRegression().fit(tr[["sif"]], tr["yield_anomaly"])
    pred = model.predict(te[["sif"]])
    oos.append({"year": year, "r2": float(r2_score(te["yield_anomaly"], pred)), "n": len(te)})
oos = pd.DataFrame(oos)
print("JJA-JulAug SIF linear OOS")
print(oos.to_string(index=False))
print("mean OOS R2", float(oos["r2"].mean()) if len(oos) else None)
