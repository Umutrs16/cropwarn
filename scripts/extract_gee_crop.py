"""Crop-specific 16-day optical + monthly climate at SPAM/CROPGRIDS/WorldCereal points."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.gee.crop_extract import extract_all_crop
from cropwarn.gee.crop_points import crop_points_worldcereal

root = Path(__file__).resolve().parents[1]
cfg = load_config(root)
pts_path = root / "data_interim" / "crop_points.csv"
if pts_path.exists() and pts_path.stat().st_size > 200:
    import pandas as pd

    pts = pd.read_csv(pts_path)
    print("loaded points", len(pts), pts.groupby("crop").size().to_dict(), flush=True)
else:
    pts = crop_points_worldcereal(root)
    print("worldcereal points", len(pts), flush=True)
years = cfg.years
if len(sys.argv) >= 3:
    years = list(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
crops = None
if "--crop" in sys.argv:
    crops = [sys.argv[sys.argv.index("--crop") + 1]]
out = extract_all_crop(root, years, pts, crops=crops)
print("crop series", len(out), flush=True)
