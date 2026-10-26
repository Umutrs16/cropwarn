"""Download GOSIF v2 monthly and sample cropland / breadbasket points."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import load_config
from cropwarn.data.gosif import download_gosif_monthly, sample_gosif_national

root = Path(__file__).resolve().parents[1]
cfg = load_config(root)
paths = download_gosif_monthly(root, cfg.years)
print("downloaded", len(paths), flush=True)
tbl = sample_gosif_national(root, cfg.years)
print("gosif rows", len(tbl), "countries", tbl["iso3"].nunique() if len(tbl) else 0, flush=True)
