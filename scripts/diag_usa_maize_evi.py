"""In-season EVI vs NASS county maize — optical should be dead at long lead."""

from pathlib import Path

import pandas as pd

from cropwarn.data.labels import rolling_trend_anomaly
from cropwarn.gee.rice_state import load_usa_maize_series

root = Path(__file__).resolve().parents[1]
lab = rolling_trend_anomaly(pd.read_csv(root / "data_interim" / "usa_maize.csv"))
s = load_usa_maize_series(root)
s["month"] = pd.to_datetime(s["date"]).dt.month
s["year"] = pd.to_datetime(s["date"]).dt.year
print("series", len(s), "units", s["unit_id"].nunique(), "cols", [c for c in s.columns if c in ("evi", "ndvi", "sif", "vpd", "precip")])
for mo in range(3, 11):
    ev = s[s["month"] == mo].groupby(["unit_id", "year"], as_index=False)["evi"].mean()
    mm = lab.merge(ev, on=["unit_id", "year"]).dropna(subset=["yield_anomaly", "evi"])
    print(f"  EVI month {mo:02d} r_anom={mm['evi'].corr(mm['yield_anomaly']):+.3f} n={len(mm)}")
jja = s[s["month"].isin([6, 7, 8])].groupby(["unit_id", "year"], as_index=False)["evi"].mean()
ok = lab.merge(jja, on=["unit_id", "year"]).dropna(subset=["yield_anomaly", "evi"])
print("EVI JJA vs anomaly", float(ok["evi"].corr(ok["yield_anomaly"])))
