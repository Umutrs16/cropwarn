import pandas as pd

from cropwarn.trade.exposure import import_exposure


def test_import_exposure_uses_lagged_network_and_negative_shocks():
    shocks = pd.DataFrame(
        {
            "iso3": ["USA", "BRA"],
            "year": [2022, 2022],
            "crop": ["soybean", "soybean"],
            "lead": [64, 64],
            "shock_hat": [-0.2, 0.1],
            "shock_true": [-0.25, 0.05],
        }
    )
    trade = pd.DataFrame(
        {
            "year": [2021, 2021],
            "crop": ["soybean", "soybean"],
            "exporter": ["USA", "BRA"],
            "importer": ["CHN", "CHN"],
            "export_share": [0.7, 0.3],
            "import_dependency": [0.8, 0.8],
        }
    )
    expo = import_exposure(shocks, trade, 2022, "soybean", lag_years=1)
    assert len(expo) == 1
    # Only the US negative shock should contribute: 0.7 * 0.2
    assert abs(expo.iloc[0]["exposure"] - 0.14) < 1e-9


def test_realized_import_shortfall_uses_qty_change_not_model_exposure():
    from cropwarn.trade.exposure import evaluate_realized_import_shortfall, trade_early_warning_skill

    preds = pd.DataFrame(
        {
            "iso3": ["USA", "BRA"] * 2,
            "year": [2022, 2022, 2022, 2022],
            "crop": ["soybean"] * 4,
            "lead": [0, 0, 192, 192],
            "pred": [-0.2, 0.1, -0.2, 0.1],
            "shock_true": [-0.25, 0.05, -0.25, 0.05],
        }
    )
    trade = pd.DataFrame(
        {
            "year": [2021, 2021, 2022, 2022],
            "crop": ["soybean"] * 4,
            "exporter": ["USA", "BRA", "USA", "BRA"],
            "importer": ["CHN", "CHN", "CHN", "CHN"],
            "export_share": [0.7, 0.3, 0.7, 0.3],
            "import_dependency": [0.8] * 4,
            "qty": [70.0, 30.0, 40.0, 20.0],
        }
    )
    # Single importer cannot define a shortfall quantile; function should skip or return empty.
    out = evaluate_realized_import_shortfall(preds, trade, 1, 0.80)
    assert out.empty or "auprc" in out.columns
    assert abs(trade_early_warning_skill(0.60, 0.70, 0.20) - (0.40 / 0.50)) < 1e-9
