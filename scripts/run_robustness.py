"""Stage 12 robustness on the official wheat+soy world. Writes results/robustness/."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import ensure_dirs, load_config
from cropwarn.data.io import read_table
from cropwarn.data.observational import build_observational_world
from cropwarn.experiments.robustness import run_robustness_suite

root = Path(__file__).resolve().parents[1]
cfg = load_config(root)
ensure_dirs(cfg)
trade = read_table(root / "data_interim" / "obs_trade_network.csv")
world = build_observational_world(root, cfg.years, cfg.crops, trade=trade)
bundle = run_robustness_suite(world, cfg, root / "results" / "robustness")
summary = bundle["summary"]
print(summary.to_string(index=False) if len(summary) else "no summary")
print(json.dumps({"n_rows": int(len(summary))}, indent=2))
