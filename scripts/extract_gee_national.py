"""National cropland-weighted monthly extract (MCD12Q1 mask, mid-month SMAP)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.gee.national import extract_all_national, sample_cropland_points

root = Path(__file__).resolve().parents[1]
cfg = load_config(root)
years = cfg.years
if len(sys.argv) > 1:
    years = [int(sys.argv[1])] if len(sys.argv) == 2 else list(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
print("years", years[0], years[-1], flush=True)
out = extract_all_national(root, years=years)
print("national rows", len(out), "cols", list(out.columns), flush=True)
# Per-country GEE sample() on large polygons can hang; GOSIF already
# samples breadbasket points. Opt in with --points.
if "--points" in sys.argv:
    try:
        pts = sample_cropland_points(root)
        print("cropland points", len(pts), flush=True)
    except Exception as exc:
        print("points skipped", exc, flush=True)
