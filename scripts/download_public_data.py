"""Best-effort public downloads. Large raster products stay manual (see data_raw/EXTERNAL_DATA.md)."""

from __future__ import annotations

from pathlib import Path

from cropwarn.config import ensure_dirs, load_config
from cropwarn.data.external import write_external_readme
from cropwarn.data.faostat import faostat_api_panel, tidy_production
from cropwarn.data.io import write_table
from cropwarn.data.trade import write_trade_readme


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    write_external_readme(cfg)
    write_trade_readme(cfg.path("data_raw") / "trade" / "README.txt")
    try:
        raw = faostat_api_panel(cfg, int(cfg.get("study", "year_start")), int(cfg.get("study", "year_end")))
        tidy = tidy_production(raw)
        write_table(tidy, cfg.path("data_raw") / "faostat_qcl_study_countries.csv")
        print("Wrote FAOSTAT study-country extract")
    except Exception as exc:
        print("FAOSTAT API extract skipped:", exc)
    print("Raster products (GOSIF, VODCA, CROPGRIDS, MIRCA-OS) must be placed under data_raw/")


if __name__ == "__main__":
    main()
