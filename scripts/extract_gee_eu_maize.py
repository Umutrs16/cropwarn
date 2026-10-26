"""GEE 16-day optical + climate at EU NUTS2 maize points."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.gee.rice_state import extract_all_rice_state

root = Path(__file__).resolve().parents[1]
cfg = load_config(root)
pts = pd.read_csv(root / "data_interim" / "eu_maize_points.csv")
years = cfg.years
if len(sys.argv) >= 3:
    years = list(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
print("points", len(pts), "units", pts["unit_id"].nunique(), "years", years[0], years[-1], flush=True)
out = extract_all_rice_state(root, years, pts, folder="gee_eu_maize", prefix="eu_maize")
print("eu-maize series", len(out), flush=True)
