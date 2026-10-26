import pandas as pd
import pytest

from cropwarn.features.factory import feature_columns
from cropwarn.validation.leakage import (
    LeakageError,
    assert_no_future_trade,
    assert_no_future_years,
    assert_no_label_columns,
)


def test_future_year_split_is_rejected():
    train = pd.DataFrame({"year": [2018, 2019]})
    test = pd.DataFrame({"year": [2019]})
    with pytest.raises(LeakageError):
        assert_no_future_years(train, test)


def test_trade_network_must_lag():
    with pytest.raises(LeakageError):
        assert_no_future_trade(2022, 2022)


def test_feature_contract_drops_labels():
    df = pd.DataFrame(
        {
            "evi_mean": [0.4],
            "yield_anomaly": [0.1],
            "shock_true": [0.2],
            "iso3": ["USA"],
        }
    )
    cols = feature_columns(df)
    assert "yield_anomaly" not in cols
    assert "shock_true" not in cols
    assert_no_label_columns(cols)
