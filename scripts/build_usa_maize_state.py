"""Roll NASS county maize to CROPGRIDS-weighted state yields + state RS."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.data.labels import rolling_trend_anomaly
from cropwarn.data.nass_maize import parse_usa_maize_state, rollup_to_state

root = Path(__file__).resolve().parents[1]
labels = parse_usa_maize_state(root)
print("state maize", len(labels), "units", labels["unit_id"].nunique() if len(labels) else 0, flush=True)
if labels.empty:
    raise SystemExit("no state maize labels")
print(
    labels.groupby("state_code")
    .agg(n=("year", "nunique"), yld=("yield_tha", "mean"), src=("source", "first"))
    .to_string(),
    flush=True,
)
lab = rolling_trend_anomaly(labels)
y12 = lab[lab["year"] == 2012][["unit_id", "yield_tha", "yield_anomaly"]]
print("2012 state anomalies", flush=True)
print(y12.sort_values("yield_anomaly").to_string(), flush=True)

gosif = pd.read_csv(root / "data_interim" / "gosif_usa_maize.csv", parse_dates=["date"])
gstate = rollup_to_state(gosif)
gstate.to_csv(root / "data_interim" / "gosif_usa_maize_state.csv", index=False)
print("gosif state", len(gstate), "units", gstate["unit_id"].nunique(), flush=True)

gstate["month"] = gstate["date"].dt.month
gstate["year"] = gstate["date"].dt.year
sif = gstate[gstate["month"] == 7].groupby(["unit_id", "year"], as_index=False)["sif"].mean()
ok = lab.merge(sif, on=["unit_id", "year"]).dropna(subset=["yield_anomaly", "sif"])
print("July SIF vs state anomaly", float(ok["sif"].corr(ok["yield_anomaly"])), "n", len(ok), flush=True)
for mo in (3, 5, 7, 8):
    sm = gstate[gstate["month"] == mo].groupby(["unit_id", "year"], as_index=False)["sif"].mean()
    mm = lab.merge(sm, on=["unit_id", "year"]).dropna(subset=["yield_anomaly", "sif"])
    print(f"  month {mo:02d} r={mm['sif'].corr(mm['yield_anomaly']):+.3f}", flush=True)
