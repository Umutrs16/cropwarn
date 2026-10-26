from cropwarn.constants import PHENOLOGY_WINDOWS
from cropwarn.data.synthetic import generate_world
from cropwarn.features.factory import build_feature_table, feature_columns
from cropwarn.phenology.alignment import PhenologyVersion
from cropwarn.validation.leakage import assert_no_label_columns, assert_realtime_phenology


def test_synthetic_world_and_features_respect_contract():
    world = generate_world(list(range(2014, 2023)), ["maize", "wheat"], seed=0)
    assert world["panel"]["yield_anomaly"].notna().any()
    feat = build_feature_table(
        world["panel"],
        world["series"],
        [64, 0],
        PhenologyVersion.DYNAMIC,
        ["optical", "sif", "sm"],
    )
    assert feat["lead"].nunique() == 2
    assert any(c.endswith("_w0_20") for c in feat.columns)
    assert_realtime_phenology(feat)
    assert_no_label_columns(feature_columns(feat))
    # SMAP-style soil moisture must be masked before 2015; here all years >= 2018.
    assert feat["mask_sm"].max() == 1.0
    assert set(range(len(PHENOLOGY_WINDOWS)))
