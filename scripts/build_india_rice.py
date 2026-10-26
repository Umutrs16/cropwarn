"""Parse India kharif/Aman rice yields and sample CROPGRIDS + GOSIF by state."""

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.data.gosif import sample_gosif_points
from cropwarn.data.india_apy import india_rice_points_from_cropgrids, parse_india_kharif_rice

root = Path(__file__).resolve().parents[1]
src = root / "data_raw" / "india_apy" / "repo" / "data" / "India Agriculture Crop Production.csv"
dst = root / "data_raw" / "india_apy" / "India Agriculture Crop Production.csv"
if src.exists() and not dst.exists():
    shutil.copy2(src, dst)

labels = parse_india_kharif_rice(root)
print("kharif rice rows", len(labels), "units", labels["unit_id"].nunique() if len(labels) else 0, flush=True)
if len(labels):
    print(labels.groupby("unit_id").agg(n=("year", "nunique"), y0=("year", "min"), y1=("year", "max"), yld=("yield_tha", "mean")).to_string(), flush=True)

pts = india_rice_points_from_cropgrids(root)
print("rice state points", len(pts), pts.groupby("unit_id").size().to_dict() if len(pts) else {}, flush=True)

if len(pts):
    cfg = load_config(root)
    gosif = sample_gosif_points(root, pts, cfg.years, "gosif_rice_state.csv")
    print("gosif rice-state rows", len(gosif), flush=True)
