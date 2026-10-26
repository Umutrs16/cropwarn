import pandas as pd

from cropwarn.config import load_config
from cropwarn.models.predictors import fit_predict


def test_fit_predict_does_not_pool_crops():
    cfg = load_config()
    train = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "BBB", "BBB"],
            "crop": ["wheat", "rice", "wheat", "rice"],
            "year": [2010, 2010, 2011, 2011],
            "yield_anomaly": [0.4, -0.5, 0.5, -0.4],
            "evi_mean": [0.3, 0.5, 0.3, 0.5],
        }
    )
    test = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "crop": ["wheat", "rice"],
            "year": [2012, 2012],
            "yield_anomaly": [0.0, 0.0],
            "evi_mean": [0.3, 0.5],
        }
    )
    pred = fit_predict("climatology", train, test, "yield_anomaly", True, cfg)
    assert pred[0] > 0.3
    assert pred[1] < -0.3
