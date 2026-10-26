import numpy as np
import pandas as pd

from cropwarn.phenology.alignment import cutoff_timestamp
from cropwarn.trade.baselines import evaluate_exposure_suite, importer_specific_truth, yield_persistence
from cropwarn.trade.exposure import evaluate_expost_exposure, importer_exposure_rows


def test_importer_specific_threshold_is_not_global_top_importers():
    g = pd.DataFrame(
        {
            "importer": ["A", "B", "C"],
            "exposure_obs": [0.90, 0.20, 0.30],
        }
    )
    past = pd.DataFrame(
        {
            "importer": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "exposure_obs": [0.85, 0.88, 0.92, 0.05, 0.06, 0.07, 0.04, 0.05, 0.06],
        }
    )
    pooled = importer_specific_truth(g, past, mode="pooled")
    own = importer_specific_truth(g, past, mode="importer_q")
    # Global Q80 flags the structurally dependent importer A.
    assert bool(pooled.iloc[0])
    # Importer-specific Q80 flags C's unusual year, not A's usual high exposure.
    assert not bool(own.iloc[0])
    assert bool(own.iloc[2])


def test_yield_persistence_uses_lagged_anomaly():
    rows = []
    for year in range(2010, 2020):
        rows.append({"iso3": "USA", "crop": "wheat", "year": year, "yield_anomaly": 0.4 if year % 2 else -0.4})
    out = yield_persistence(pd.DataFrame(rows), crops=("wheat",))
    assert out["pooled"]["n"] >= 3
    assert np.isfinite(out["pooled"]["r2"])


def test_expost_pooled_mode_still_runs():
    labels = []
    trade = []
    preds = []
    for year in range(2012, 2019):
        shock = -0.4 if year == 2016 else -0.05
        for iso3 in ("AAA", "BBB"):
            labels.append(
                {
                    "iso3": iso3,
                    "crop": "wheat",
                    "year": year,
                    "yield_anomaly": shock,
                    "yield_tha": 3.0,
                    "harvested_ha": 1_000_000,
                    "production_t": 3_000_000,
                }
            )
        for imp in [f"I{i}" for i in range(10)]:
            trade.append(
                {
                    "year": year,
                    "crop": "wheat",
                    "exporter": "AAA",
                    "importer": imp,
                    "export_share": 0.8,
                    "import_dependency": 0.4,
                    "qty": 80.0,
                }
            )
            trade.append(
                {
                    "year": year,
                    "crop": "wheat",
                    "exporter": "BBB",
                    "importer": imp,
                    "export_share": 0.2,
                    "import_dependency": 0.4,
                    "qty": 20.0,
                }
            )
        if year >= 2016:
            for iso3 in ("AAA", "BBB"):
                preds.append(
                    {
                        "iso3": iso3,
                        "crop": "wheat",
                        "year": year,
                        "lead": 192,
                        "pred": shock * 0.9,
                        "shock_true": shock,
                    }
                )
    out = evaluate_expost_exposure(pd.DataFrame(preds), pd.DataFrame(trade), pd.DataFrame(labels))
    assert not out.empty
    rows = importer_exposure_rows(pd.DataFrame(preds), pd.DataFrame(trade), pd.DataFrame(labels))
    assert {"importer", "exposure_hat", "exposure_obs"} <= set(rows.columns)
    assert not rows.empty


def _toy_trade_world():
    labels, trade, preds = [], [], []
    importers = [f"I{i}" for i in range(10)]
    for year in range(2012, 2019):
        shock_a = -0.55 if year == 2015 else (-0.50 if year >= 2016 else -0.02)
        shock_b = -0.02
        for iso3, shock in (("AAA", shock_a), ("BBB", shock_b)):
            labels.append(
                {
                    "iso3": iso3,
                    "crop": "wheat",
                    "year": year,
                    "yield_anomaly": shock,
                    "yield_tha": 3.0,
                    "harvested_ha": 1_000_000,
                    "production_t": 3_000_000,
                }
            )
        for i, imp in enumerate(importers):
            share_a = 0.90 - 0.08 * i
            trade.append(
                {
                    "year": year,
                    "crop": "wheat",
                    "exporter": "AAA",
                    "importer": imp,
                    "export_share": share_a,
                    "import_dependency": 0.4,
                    "qty": 100.0 * share_a,
                }
            )
            trade.append(
                {
                    "year": year,
                    "crop": "wheat",
                    "exporter": "BBB",
                    "importer": imp,
                    "export_share": 1.0 - share_a,
                    "import_dependency": 0.4,
                    "qty": 100.0 * (1.0 - share_a),
                }
            )
        if year >= 2016:
            for lead in (0, 192):
                for iso3, shock in (("AAA", shock_a), ("BBB", shock_b)):
                    preds.append(
                        {
                            "iso3": iso3,
                            "crop": "wheat",
                            "year": year,
                            "lead": lead,
                            "pred": shock * (0.9 if lead == 192 else 1.0),
                            "shock_true": shock,
                        }
                    )
    return pd.DataFrame(preds), pd.DataFrame(trade), pd.DataFrame(labels)


def test_exposure_suite_auprc_finite_at_two_leads():
    preds, trade, labels = _toy_trade_world()
    suite = evaluate_exposure_suite(preds, trade, labels, label_mode="pooled")
    assert set(suite["lead"]) >= {0, 192}
    for lead in (0, 192):
        sub = suite[suite["lead"] == lead]
        assert sub["auprc_forecast"].notna().any()
        assert sub["auprc_n0_lag"].notna().any() or sub["auprc_n1_mean"].notna().any()


def test_cutoff_timestamp_365_is_one_year_before_harvest():
    harvest = cutoff_timestamp(2020, 280, 0)
    year_ahead = cutoff_timestamp(2020, 280, 365)
    assert (harvest - year_ahead).days == 365
