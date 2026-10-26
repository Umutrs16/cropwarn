"""Retrain revision sub-experiments. Writes results/revision_p0/*, never official results/."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.config import ensure_dirs, load_config
from cropwarn.data.io import read_table
from cropwarn.data.observational import build_observational_world, observational_inventory
from cropwarn.experiments.robustness import run_robustness_suite
from cropwarn.experiments.runner import run_experiment_suite
from cropwarn.gee.tropomi import load_tropomi
from cropwarn.revision.world_ops import attach_tropomi_as_sif, filter_world_years, revision_config_extra
from cropwarn.validation.metrics import earliest_reliable_lead, stable_reliable_lead

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_p0"


def _lead_table(lead_skill, label: str):
    rows = []
    for exp, g in lead_skill.groupby("experiment"):
        harvest = g.loc[g["lead"] == 0, "r2"]
        rows.append(
            {
                "variant": label,
                "experiment": exp,
                "harvest_r2": float(harvest.mean()) if len(harvest) else float("nan"),
                "r2_192": float(g.loc[g["lead"] == 192, "r2"].mean()) if (g["lead"] == 192).any() else float("nan"),
                "isolated_lead": earliest_reliable_lead(g),
                "stable_lead": stable_reliable_lead(g),
            }
        )
    return rows


def main() -> None:
    jobs = sys.argv[1:] or ["labels", "common2015", "tropomi"]
    cfg = load_config(ROOT)
    ensure_dirs(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    trade = read_table(ROOT / "data_interim" / "obs_trade_network.csv")
    world = build_observational_world(ROOT, cfg.years, cfg.crops, trade=trade)
    summary_rows = []

    if "labels" in jobs:
        print("=== label robustness theil_sen / first_difference / percent ===", flush=True)
        bundle = run_robustness_suite(
            world,
            cfg,
            OUT / "labels",
            methods=["theil_sen", "first_difference", "percent"],
        )
        if len(bundle.get("summary", [])):
            summary_rows.extend(bundle["summary"].assign(block="labels").to_dict("records"))

    if "common2015" in jobs:
        print("=== common-period 2015–2024 ablation ===", flush=True)
        extra = revision_config_extra(test_start=2019, min_train=3)
        sub_cfg = load_config(ROOT, extra=extra)
        sub = filter_world_years(world, 2015)
        bundle = run_experiment_suite(
            sub,
            sub_cfg,
            demo=True,
            schemes=["rolling_origin"],
            results_dir=OUT / "common_2015",
            experiments=["B2_optical_climate", "B3_plus_sm", "B4_plus_sif", "B5_plus_vod", "P_full_physiology"],
        )
        summary_rows.extend(_lead_table(bundle["lead_skill"], "common_2015_2024"))

    if "tropomi" in jobs:
        tropomi = load_tropomi(ROOT)
        inv = observational_inventory(ROOT, world)
        print("TROPOMI rows", len(tropomi), "inventory.tropomi_sif would be", bool(len(tropomi)), flush=True)
        if tropomi.empty:
            (OUT / "tropomi_status.json").write_text(
                json.dumps({"ok": False, "reason": "tropomi_crop.csv missing; run scripts/extract_tropomi.py"}, indent=2),
                encoding="utf-8",
            )
        else:
            extra = revision_config_extra(test_start=2020, min_train=2)
            extra["ablations"] = {
                "B2_optical_climate": {
                    "phenology": "dynamic",
                    "modalities": ["optical", "thermal", "climate"],
                    "model": "xgb",
                    "conformal": False,
                    "trade": False,
                },
                "B4_tropomi_sif": {
                    "phenology": "dynamic",
                    "modalities": ["optical", "thermal", "climate", "sif"],
                    "model": "xgb",
                    "conformal": False,
                    "trade": False,
                },
            }
            sub_cfg = load_config(ROOT, extra=extra)
            tw = attach_tropomi_as_sif(filter_world_years(world, 2018, 2021), tropomi)
            bundle = run_experiment_suite(
                tw,
                sub_cfg,
                demo=True,
                schemes=["rolling_origin"],
                results_dir=OUT / "tropomi_2018",
                experiments=["B2_optical_climate", "B4_tropomi_sif"],
            )
            summary_rows.extend(_lead_table(bundle["lead_skill"], "tropomi_2018_2024"))
            (OUT / "tropomi_status.json").write_text(
                json.dumps({"ok": True, "n": int(len(tropomi)), "inventory_was": inv}, indent=2, default=str),
                encoding="utf-8",
            )

    if summary_rows:
        import pandas as pd

        pd.DataFrame(summary_rows).to_csv(OUT / "retrain_summary.csv", index=False)
        print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
