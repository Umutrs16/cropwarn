"""Write Fig. 7–9 from revision_p0 CSVs without touching official results/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropwarn.data.io import read_table
from cropwarn.viz.figures import write_revision_main_figures

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "revision_p0"
FIGS = ROOT / "figures"


def main() -> None:
    paths = write_revision_main_figures(
        read_table(REV / "season_progress_skill.csv"),
        read_table(REV / "trade_metrics_by_lead.csv"),
        read_table(REV / "import_outcome_by_lead.csv"),
        read_table(REV / "trade_structure_by_lead.csv"),
        read_table(REV / "robustness_matrix.csv"),
        FIGS,
    )
    season = FIGS / "fig7_season_progress.png"
    if season.exists():
        shutil.copy2(season, FIGS / "fig_season_progress.png")
        shutil.copy2(season, REV / "fig_season_progress.png")
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
