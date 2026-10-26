"""Command-line interface for the crop early-warning project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cropwarn.config import ensure_dirs, load_config
from cropwarn.data.external import write_external_readme
from cropwarn.pipeline import STAGES, run_demo, run_full, run_observational, stage_notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropwarn",
        description="Trustworthy early warning of global crop production shocks.",
    )
    parser.add_argument("--root", default=None, help="Project root (default: package parent).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run-demo", help="Synthetic but mechanistic end-to-end paper run.")
    sub.add_parser("run-full", help="Same suite with the full lead-time and split matrix.")
    sub.add_parser("run-observational", help="FAOSTAT wheat+soy observational suite; NC/RSE verdict.")
    sub.add_parser("stages", help="Print Stage 0-12 contract.")
    sub.add_parser("init", help="Create folders and data READMEs.")
    p_download = sub.add_parser("download-notes", help="Write external data instructions.")
    p_download.add_argument("--try-faostat", action="store_true")

    args = parser.parse_args(argv)
    root = args.root

    if args.cmd == "stages":
        for name, note in stage_notes().items():
            print(f"{name}: {note}")
        return 0
    if args.cmd == "init":
        cfg = load_config(root)
        ensure_dirs(cfg)
        write_external_readme(cfg)
        print(f"Initialized under {cfg.root}")
        return 0
    if args.cmd == "download-notes":
        cfg = load_config(root)
        path = write_external_readme(cfg)
        print(path)
        if args.try_faostat:
            from cropwarn.data.faostat import faostat_api_panel, tidy_production

            raw = faostat_api_panel(cfg, 2018, 2022)
            tidy = tidy_production(raw)
            out = cfg.path("data_raw") / "faostat_sample.csv"
            tidy.to_csv(out, index=False)
            print(out)
        return 0
    if args.cmd == "run-demo":
        bundle = run_demo(root)
        print(json.dumps(bundle["gates"], indent=2))
        print("Figures:")
        for p in bundle["figures"]:
            print(" ", p)
        return 0
    if args.cmd == "run-observational":
        bundle = run_observational(root)
        print(json.dumps(bundle["verdict"], indent=2, default=str))
        return 0
    if args.cmd == "run-full":
        bundle = run_full(root)
        print(json.dumps(bundle["gates"], indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
