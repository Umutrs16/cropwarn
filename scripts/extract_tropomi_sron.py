"""Stream mid-month TROPOSIF L2B (2018–2021) and sample headline crop points."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cropwarn.data.tropomi_sron import sample_tropomi_sron

root = Path(__file__).resolve().parents[1]
pts = pd.read_csv(root / "data_interim" / "crop_points.csv")
pts = pts[pts["crop"].isin(["wheat", "soybean"])]
out = sample_tropomi_sron(root, list(range(2018, 2022)), pts)
print("tropomi rows", len(out), "sif_ok", float(out["tropomi_sif"].notna().mean()) if len(out) else 0, flush=True)
