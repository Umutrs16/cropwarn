import numpy as np
import pandas as pd

from cropwarn.data.observational import _season_slice
from cropwarn.trade.exposure import evaluate_expost_exposure
from cropwarn.validation.bootstrap import cluster_bootstrap_lead
from cropwarn.validation.metrics import robust_stable_lead, stable_reliable_lead


def test_robust_stable_lead_uses_lower_bound():
    skill = pd.DataFrame(
        {
            "lead": [0, 32, 64, 96, 128],
            "r2": [0.20, 0.18, 0.17, 0.16, 0.151],
            "r2_p05": [0.16, 0.15, 0.15, 0.08, 0.04],
        }
    )
    assert stable_reliable_lead(skill) == 128
    assert robust_stable_lead(skill) == 64


def test_robust_stable_lead_none_when_harvest_ci_fails():
    skill = pd.DataFrame({"lead": [0, 192], "r2_p05": [0.11, 0.16]})
    assert np.isnan(robust_stable_lead(skill))


def test_cluster_bootstrap_reports_stable_lead_delta():
    rows = []
    for year in range(2016, 2024):
        for crop in ("wheat", "soybean"):
            for lead in (0, 96, 192):
                y = 0.2 * (year - 2020) + (0.1 if crop == "wheat" else -0.1)
                rows.append(
                    {
                        "experiment": "B2_optical_climate",
                        "iso3": "USA",
                        "crop": crop,
                        "year": year,
                        "lead": lead,
                        "yield_anomaly": y,
                        "pred": 0.15 * (year - 2020),
                    }
                )
                rows.append(
                    {
                        "experiment": "B4_plus_sif",
                        "iso3": "USA",
                        "crop": crop,
                        "year": year,
                        "lead": lead,
                        "yield_anomaly": y,
                        "pred": 0.16 * (year - 2020),
                    }
                )
    out = cluster_bootstrap_lead(pd.DataFrame(rows), n_boot=20, seed=0)
    delta = out[out["experiment"] == "B4_minus_B2"]
    assert "dstable_ge96" in delta.columns
    assert "dstable_p50" in delta.columns


def test_expost_exposure_uses_history_threshold_not_same_year_top20():
    labels = []
    trade = []
    preds = []
    exporters = ["AAA", "BBB"]
    importers = [f"I{i}" for i in range(10)]
    for year in range(2010, 2019):
        shock = -0.8 if year == 2012 else (-0.05 if year < 2016 else -0.10)
        for iso3 in exporters:
            labels.append(
                {
                    "iso3": iso3,
                    "crop": "wheat",
                    "year": year,
                    "yield_anomaly": shock,
                    "yield_tha": 3.0 + shock,
                    "harvested_ha": 1_000_000,
                    "production_t": 3_000_000,
                }
            )
        for imp in importers:
            trade.append(
                {
                    "year": year,
                    "crop": "wheat",
                    "exporter": "AAA",
                    "importer": imp,
                    "export_share": 0.7,
                    "import_dependency": 0.4,
                    "qty": 100.0,
                }
            )
            trade.append(
                {
                    "year": year,
                    "crop": "wheat",
                    "exporter": "BBB",
                    "importer": imp,
                    "export_share": 0.3,
                    "import_dependency": 0.4,
                    "qty": 40.0,
                }
            )
        if year >= 2016:
            for lead in (0, 192):
                for iso3 in exporters:
                    preds.append(
                        {
                            "iso3": iso3,
                            "crop": "wheat",
                            "year": year,
                            "lead": lead,
                            "pred": shock * 0.8,
                            "shock_true": shock,
                        }
                    )
    out = evaluate_expost_exposure(pd.DataFrame(preds), pd.DataFrame(trade), pd.DataFrame(labels))
    assert not out.empty
    # Quiet-ish forecast years should not be forced to 20% positives.
    later = out[out["year"] >= 2016]
    assert later["n_high"].min() < later["n_importers"].min() * 0.2 + 1e-6 or later["n_high"].nunique() > 1
    assert "cont_spearman" in out.columns


def test_season_slice_preseason_pad():
    dates = pd.date_range("2011-06-01", "2012-10-01", freq="16D")
    base = pd.DataFrame({"date": dates, "evi": 0.4})
    short = _season_slice(base, 2012, sos0=120, eos0=280, preseason_days=16)
    long = _season_slice(base, 2012, sos0=120, eos0=280, preseason_days=200)
    assert len(long) >= len(short)
