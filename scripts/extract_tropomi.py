"""Extract 2018–2024 TROPOMI SIF at headline crop points."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cropwarn.config import load_config
from cropwarn.gee.tropomi import extract_tropomi

root = Path(__file__).resolve().parents[1]
cfg = load_config(root)
pts_path = root / "data_interim" / "crop_points.csv"
if not pts_path.exists():
    raise SystemExit("missing data_interim/crop_points.csv")
pts = pd.read_csv(pts_path)
crops = [c for c in cfg.get("study", "headline_crops") if c in set(pts["crop"])]
years = [y for y in cfg.years if y >= 2018]
out = extract_tropomi(root, years, pts, crops=crops)
print("tropomi rows", len(out), "years", sorted(out["year"].unique()) if len(out) else [], flush=True)
