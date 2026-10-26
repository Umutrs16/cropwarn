"""Parse NASS county maize yields and sample CROPGRIDS + GOSIF."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.data.gosif import sample_gosif_points
from cropwarn.data.nass_maize import load_usa_maize, parse_usa_maize, usa_maize_points

root = Path(__file__).resolve().parents[1]
labels = parse_usa_maize(root)
print(
    "usa maize rows",
    len(labels),
    "units",
    labels["unit_id"].nunique() if len(labels) else 0,
    flush=True,
)
if len(labels):
    print(
        labels.groupby("state_code")
        .agg(n_units=("unit_id", "nunique"), y0=("year", "min"), y1=("year", "max"), yld=("yield_tha", "mean"))
        .to_string(),
        flush=True,
    )
    y12 = labels[labels["year"] == 2012]["yield_tha"]
    pre = labels[labels["year"].between(2008, 2011)].groupby("unit_id")["yield_tha"].mean()
    hit = labels[labels["year"] == 2012].set_index("unit_id")["yield_tha"]
    delta = (hit - pre).dropna()
    print(
        "2012 vs 2008-11 mean t/ha",
        f"median={delta.median():.2f}",
        f"p10={delta.quantile(0.1):.2f}",
        f"n={len(delta)}",
        flush=True,
    )

pts = usa_maize_points(root)
print(
    "usa maize points",
    len(pts),
    pts.groupby("state_code").size().to_dict() if len(pts) else {},
    flush=True,
)

if len(pts):
    cfg = load_config(root)
    gosif = sample_gosif_points(root, pts, cfg.years, "gosif_usa_maize.csv")
    print("gosif usa-maize rows", len(gosif), flush=True)
