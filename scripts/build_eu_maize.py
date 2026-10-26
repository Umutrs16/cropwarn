"""Parse Eurostat NUTS2 maize and sample CROPGRIDS + GOSIF."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.data.eurostat_maize import eu_maize_points, parse_eu_maize
from cropwarn.data.gosif import sample_gosif_points
from cropwarn.data.labels import rolling_trend_anomaly

root = Path(__file__).resolve().parents[1]
labels = parse_eu_maize(root)
print("eu maize", len(labels), "units", labels["unit_id"].nunique() if len(labels) else 0, flush=True)
if labels.empty:
    raise SystemExit("no eu maize")
print(
    labels.groupby("iso3").agg(n_units=("unit_id", "nunique"), yld=("yield_tha", "mean"), area=("harvested_ha", "mean")).to_string(),
    flush=True,
)
lab = rolling_trend_anomaly(labels)
for year in (2003, 2018, 2022):
    sub = lab[lab["year"] == year][["unit_id", "yield_tha", "yield_anomaly"]].dropna()
    if sub.empty:
        continue
    print(f"{year} median anomaly {sub['yield_anomaly'].median():+.2f} n={len(sub)}", flush=True)
    print(sub.nsmallest(5, "yield_anomaly").to_string(), flush=True)

pts = eu_maize_points(root)
print("points", len(pts), pts.groupby("iso3").size().to_dict() if len(pts) else {}, flush=True)
if len(pts):
    cfg = load_config(root)
    gosif = sample_gosif_points(root, pts, cfg.years, "gosif_eu_maize.csv")
    print("gosif", len(gosif), flush=True)
    g = gosif.copy()
    g["month"] = g["date"].dt.month
    g["year"] = g["date"].dt.year
    sif = g[g["month"] == 7].groupby(["unit_id", "year"], as_index=False)["sif"].mean()
    ok = lab.merge(sif, on=["unit_id", "year"]).dropna(subset=["yield_anomaly", "sif"])
    print("July SIF vs anomaly", float(ok["sif"].corr(ok["yield_anomaly"])), "n", len(ok), flush=True)
    for mo in (3, 5, 7, 8):
        sm = g[g["month"] == mo].groupby(["unit_id", "year"], as_index=False)["sif"].mean()
        mm = lab.merge(sm, on=["unit_id", "year"]).dropna(subset=["yield_anomaly", "sif"])
        print(f"  month {mo:02d} r={mm['sif'].corr(mm['yield_anomaly']):+.3f}", flush=True)
