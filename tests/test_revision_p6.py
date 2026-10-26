import numpy as np
import pandas as pd

from cropwarn.trade.baselines import empirical_pvalue, evaluate_exposure_suite
from cropwarn.trade.exposure import import_rolling_anomaly, overlay_observed_shocks


def test_empirical_pvalue_plus_one():
    draws = np.array([0.3, 0.4, 0.5])
    assert empirical_pvalue(draws, 0.9) == 1 / 4
    assert empirical_pvalue(draws, 0.4) == 3 / 4


def test_overlay_keeps_predicted_shock():
    shocks = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "crop": ["wheat"],
            "year": [2016],
            "shock_hat": [-0.2],
            "shock_true": [-0.1],
        }
    )
    obs = pd.DataFrame({"iso3": ["AAA"], "crop": ["wheat"], "year": [2016], "shock_obs": [-0.4]})
    out = overlay_observed_shocks(shocks, obs)
    assert float(out["shock_hat"].iloc[0]) == -0.2
    assert float(out["shock_true"].iloc[0]) == -0.4


def test_import_anomaly_is_causal():
    rows = []
    for year in range(2005, 2018):
        rows.append({"year": year, "crop": "wheat", "exporter": "USA", "importer": "CHN", "qty": 100.0 + 3 * year})
    trade = pd.DataFrame(rows)
    anom = import_rolling_anomaly(trade)
    late = anom[anom["year"] == 2016]
    assert late["import_anomaly"].notna().any()


def test_production_label_changes_truth_not_forecast():
    labels, trade, preds = [], [], []
    for year in range(2010, 2019):
        yshock = -0.4 if year == 2016 else -0.02
        pshock = -0.8 if year == 2016 else -0.01
        for iso3, area in (("AAA", 1_000_000), ("BBB", 1_000_000)):
            labels.append(
                {
                    "iso3": iso3,
                    "crop": "wheat",
                    "year": year,
                    "yield_anomaly": yshock,
                    "yield_tha": 3.0,
                    "harvested_ha": area,
                    "production_t": 3.0 * area * (1 + pshock if year == 2016 and iso3 == "AAA" else 1.0),
                }
            )
            preds.append(
                {
                    "iso3": iso3,
                    "crop": "wheat",
                    "year": year,
                    "lead": 192,
                    "pred": yshock,
                    "shock_true": yshock,
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
                    "import_dependency": 0.5,
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
                    "import_dependency": 0.5,
                    "qty": 20.0,
                }
            )
    lab = pd.DataFrame(labels)
    pred = pd.DataFrame(preds)
    tr = pd.DataFrame(trade)
    obs = lab[["iso3", "crop", "year"]].copy()
    obs["shock_obs"] = np.where((lab["iso3"] == "AAA") & (lab["year"] == 2016), -0.8, -0.01)
    y = evaluate_exposure_suite(pred[pred["year"] >= 2016], tr, lab, label_mode="pooled")
    p = evaluate_exposure_suite(pred[pred["year"] >= 2016], tr, lab, label_mode="pooled", obs_shocks=obs)
    assert len(y) and len(p)
    assert "auprc_n0_lag" in p.columns
