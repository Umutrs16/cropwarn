"""Unzip VODCA L and sample monthly means at crop + India-rice points."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.data.vodca import sample_vodca_crop

root = Path(__file__).resolve().parents[1]
tbl = sample_vodca_crop(root)
print("vodca rows", len(tbl), flush=True)
