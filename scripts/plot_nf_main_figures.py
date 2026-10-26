"""Nature Food main figures: exposure skill + robustness panels."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.data.io import read_table
from cropwarn.viz.figures import fig1_framework, fig3_lead_skill, fig_nf_exposure_skill, fig_nf_robustness

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"


def main() -> None:
    fig1_framework(FIGS)
    lead = read_table(ROOT / "results" / "lead_skill.csv")
    crop = read_table(ROOT / "results" / "revision_p1" / "lead_skill_by_crop.csv")
    b2_crop = crop[crop["experiment"] == "B2_optical_climate"] if "experiment" in crop.columns else crop
    fig3_lead_skill(lead, FIGS, crop_skill=b2_crop)
    suite = read_table(ROOT / "results" / "revision_p2" / "exposure_suite_pooled_by_lead.csv")
    p1 = fig_nf_exposure_skill(suite, FIGS)
    coverage = {"Pooled": 0.892, "Wheat": 0.88, "Europe": 0.78, "Oceania": 0.79}
    detrend = {
        "Linear": 0.19,
        "Theil–Sen": 0.17,
        "Percent": 0.23,
        "Δy": 0.13,
        "Quadratic": 0.02,
    }
    p2 = fig_nf_robustness(b2_crop, coverage, 0.62, FIGS, detrend=detrend)
    print(p1, p2)


if __name__ == "__main__":
    main()
