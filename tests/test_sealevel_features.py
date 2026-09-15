import pandas as pd
import pytest

from migracoast_ml.features.sealevel import SeaLevelIndex


@pytest.fixture
def index():
    return SeaLevelIndex(
        pd.DataFrame(
            {
                "lat": [42.4, 25.8],
                "lon": [-71.1, -80.2],
                "slr_2050_ft": [1.08, 1.15],
                "slr_2100_ft": [2.59, 2.99],
            }
        )
    )


def test_returns_both_horizon_features(index):
    feats = index.features(42.4, -71.1)
    assert set(feats) == {"slr_2050_ft", "slr_2100_ft"}


def test_picks_the_nearest_projection_site(index):
    near_boston = index.features(42.0, -71.0)
    assert near_boston["slr_2050_ft"] == pytest.approx(1.08)

    near_miami = index.features(26.0, -80.0)
    assert near_miami["slr_2050_ft"] == pytest.approx(1.15)


def test_a_distant_point_still_returns_a_value(index):
    feats = index.features(0.0, 0.0)
    assert feats["slr_2050_ft"] == pytest.approx(1.08)


def test_empty_table_raises():
    with pytest.raises(ValueError, match="empty"):
        SeaLevelIndex(pd.DataFrame(columns=["lat", "lon", "slr_2050_ft", "slr_2100_ft"]))


def test_picks_the_true_great_circle_nearest_site():
    # Query at (43.0, -70.0). Site A sits due north at (52.8, -70.0),
    # site B sits due west at (43.0, -83.4). The equirectangular
    # approximation (which scales longitude by cos(query latitude)) puts
    # site A closer, but true haversine distance is A=1089.7118 km vs
    # B=1088.5691 km, so B is actually nearer by about 1.14 km.
    site_a = SeaLevelIndex(
        pd.DataFrame(
            {
                "lat": [52.8, 43.0],
                "lon": [-70.0, -83.4],
                "slr_2050_ft": [1.08, 1.15],
                "slr_2100_ft": [2.59, 2.99],
            }
        )
    )
    feats = site_a.features(43.0, -70.0)
    assert feats["slr_2050_ft"] == pytest.approx(1.15)
    assert feats["slr_2100_ft"] == pytest.approx(2.99)
