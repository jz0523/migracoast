import pandas as pd
import pytest

from migracoast_ml.features.hazards import HazardIndex


@pytest.fixture
def index():
    events = pd.DataFrame(
        {
            "lat": [42.36, 42.38, 42.41, 25.77],
            "lon": [-71.06, -71.09, -71.12, -80.19],
            "kind": ["flood", "flood", "storm", "flood"],
        }
    )
    return HazardIndex(events, n_years=25)


def test_returns_both_rate_features(index):
    assert set(index.features(42.4, -71.1)) == {
        "flood_events_per_year",
        "storm_events_per_year",
    }


def test_counts_are_divided_by_the_year_span(index):
    feats = index.features(42.4, -71.1)
    # Two floods and one storm fall in the 42.4, -71.1 cell.
    assert feats["flood_events_per_year"] == pytest.approx(2 / 25)
    assert feats["storm_events_per_year"] == pytest.approx(1 / 25)


def test_a_cell_with_no_events_reports_zero(index):
    feats = index.features(44.0, -69.0)
    assert feats["flood_events_per_year"] == 0.0
    assert feats["storm_events_per_year"] == 0.0


def test_lookup_is_by_cell_not_by_radius(index):
    # Any point inside the cell yields the cell's counts.
    assert index.features(42.38, -71.07) == index.features(42.4, -71.1)


def test_zero_years_raises():
    with pytest.raises(ValueError, match="n_years"):
        HazardIndex(pd.DataFrame(columns=["lat", "lon", "kind"]), n_years=0)
