"""GOSIF vs NASS county maize anomalies — does the label couple?"""

from pathlib import Path

import pandas as pd

from cropwarn.data.labels import rolling_trend_anomaly

root = Path(__file__).resolve().parents[1]
lab = rolling_trend_anomaly(pd.read_csv(root / "data_interim" / "usa_maize.csv"))
g = pd.read_csv(root / "data_interim" / "gosif_usa_maize.csv", parse_dates=["date"])
g["month"] = g["date"].dt.month
g["year"] = g["date"].dt.year
sif = g[g["month"].isin([6, 7, 8])].groupby(["unit_id", "year"], as_index=False)["sif"].mean()
ok = lab.merge(sif, on=["unit_id", "year"]).dropna(subset=["yield_anomaly", "sif"])
print("n", len(ok), "units", ok["unit_id"].nunique())
print("r SIF JJA vs anomaly", float(ok["sif"].corr(ok["yield_anomaly"])))
print("r SIF JJA vs yield", float(ok["sif"].corr(ok["yield_tha"])))
for mo in range(3, 11):
    sm = g[g["month"] == mo].groupby(["unit_id", "year"], as_index=False)["sif"].mean()
    mm = lab.merge(sm, on=["unit_id", "year"]).dropna(subset=["yield_anomaly", "sif"])
    print(
        f"  month {mo:02d} r_anom={mm['sif'].corr(mm['yield_anomaly']):+.3f} "
        f"r_yld={mm['sif'].corr(mm['yield_tha']):+.3f} n={len(mm)}"
    )
y12 = ok[ok["year"] == 2012]
print("2012 SIF vs anomaly", float(y12["sif"].corr(y12["yield_anomaly"])), "n", len(y12))
print(y12.nsmallest(8, "yield_anomaly")[["unit_id", "yield_tha", "yield_anomaly", "sif"]].to_string())
print(y12.nlargest(5, "yield_anomaly")[["unit_id", "yield_tha", "yield_anomaly", "sif"]].to_string())
