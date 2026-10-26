import numpy as np
import pandas as pd

from cropwarn.config import load_config
from cropwarn.validation.go_nogo import evaluate_gates


def test_phenology_gate_uses_lowlat_column():
    cfg = load_config()
    summary = pd.DataFrame(
        {
            "experiment": ["B0_chen_rf", "B0_chen_rf", "B1_dynamic_rf", "B1_dynamic_rf"],
            "lead": [0, 64, 0, 64],
            "r2": [0.20, 0.10, 0.18, 0.08],
            "r2_lowlat_multicrop": [0.10, 0.05, 0.18, 0.12],
            "earliest_lead": [0, 0, 0, 0],
            "coverage90": [0.90, 0.90, 0.90, 0.90],
        }
    )
    report = evaluate_gates(summary, cfg, trade_gain_days=32)
    assert report["dynamic_phenology"]["go"] is True
    assert report["dynamic_phenology"]["value"] > 0.03


def test_phenology_gate_rejects_less_negative_r2():
    cfg = load_config()
    summary = pd.DataFrame(
        {
            "experiment": ["B0_chen_rf", "B1_dynamic_rf"],
            "lead": [0, 0],
            "r2": [-0.40, -0.32],
            "r2_lowlat_multicrop": [-0.40, -0.32],
            "earliest_lead": [np.nan, np.nan],
            "coverage90": [0.90, 0.90],
        }
    )
    report = evaluate_gates(summary, cfg, trade_gain_days=128)
    assert report["dynamic_phenology"]["go"] is False


def test_physiology_gate_does_not_invent_lead_from_auroc():
    cfg = load_config()
    summary = pd.DataFrame(
        {
            "experiment": ["B2_optical_climate", "B2_optical_climate", "B4_plus_sif", "B4_plus_sif"],
            "lead": [64, 128, 64, 128],
            "r2": [-0.05, -0.10, -0.04, -0.08],
            "auroc": [0.65, 0.62, 0.70, 0.68],
            "earliest_lead": [np.nan, np.nan, np.nan, np.nan],
            "coverage90": [0.90, 0.90, 0.90, 0.90],
        }
    )
    report = evaluate_gates(summary, cfg, trade_gain_days=None)
    assert report["physiology_lead"]["go"] is False
    assert report["physiology_lead"]["value"] != report["physiology_lead"]["value"]
