"""Parse NBS provincial early-rice yields and sample CROPGRIDS + GOSIF."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.data.china_early_rice import (
    china_early_rice_points,
    parse_china_early_rice,
)
from cropwarn.data.gosif import sample_gosif_points

root = Path(__file__).resolve().parents[1]
labels = parse_china_early_rice(root)
print(
    "early rice rows",
    len(labels),
    "units",
    labels["unit_id"].nunique() if len(labels) else 0,
    flush=True,
)
if len(labels):
    print(
        labels.groupby("unit_id")
        .agg(n=("year", "nunique"), y0=("year", "min"), y1=("year", "max"), yld=("yield_tha", "mean"), sd=("yield_tha", "std"))
        .to_string(),
        flush=True,
    )

pts = china_early_rice_points(root)
print(
    "early rice points",
    len(pts),
    pts.groupby("unit_id").size().to_dict() if len(pts) else {},
    flush=True,
)

if len(pts):
    cfg = load_config(root)
    gosif = sample_gosif_points(root, pts, cfg.years, "gosif_china_early_rice.csv")
    print("gosif china-early rows", len(gosif), flush=True)
