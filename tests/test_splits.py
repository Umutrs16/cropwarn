import pandas as pd

from cropwarn.validation.splits import rolling_origin, shock_holdout


def test_rolling_origin_never_trains_on_test_year():
    df = pd.DataFrame(
        {
            "year": [2015, 2016, 2017, 2018] * 3,
            "region": ["Europe"] * 12,
            "iso3": ["FRA"] * 12,
        }
    )
    splits = list(rolling_origin(df, test_start=2018, min_train_years=2))
    assert splits
    for s in splits:
        assert df.loc[s.train_idx, "year"].max() < df.loc[s.test_idx, "year"].min()


def test_shock_holdout_holds_entire_year():
    df = pd.DataFrame({"year": list(range(2005, 2013)), "region": ["NorthAmerica"] * 8})
    splits = list(shock_holdout(df, [2012], min_train_years=4))
    assert splits[0].test_year == 2012
    assert set(df.loc[splits[0].test_idx, "year"]) == {2012}
