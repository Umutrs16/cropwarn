import numpy as np

from cropwarn.validation.metrics import earliest_reliable_lead, regression_metrics
import pandas as pd


def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["r2"] == 1.0
    assert m["rmse"] == 0.0


def test_earliest_reliable_lead_picks_longest_ok_lead():
    df = pd.DataFrame({"lead": [128, 64, 0], "r2": [0.05, 0.2, 0.4]})
    assert earliest_reliable_lead(df, r2_floor=0.15) == 64


def test_stable_lead_ignores_isolated_long_lead():
    from cropwarn.validation.metrics import stable_reliable_lead

    isolated = pd.DataFrame({"lead": [0, 32, 64, 96, 128, 160, 192], "r2": [0.20, 0.18, 0.16, 0.10, 0.12, 0.10, 0.18]})
    assert stable_reliable_lead(isolated) == 64
    assert earliest_reliable_lead(isolated) == 192


def test_stable_lead_contiguous_from_harvest():
    from cropwarn.validation.metrics import stable_reliable_lead

    all_ok = pd.DataFrame({"lead": [0, 32, 64, 96, 128, 160, 192], "r2": [0.20] * 7})
    assert stable_reliable_lead(all_ok) == 192
