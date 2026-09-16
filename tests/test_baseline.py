import numpy as np
import pandas as pd
import pytest

from migracoast_ml.baseline import (
    REGION_LISTS,
    alignment_confidence,
    extract_point_lists,
    reconstruct_coordinates,
)

NOTEBOOK = "Check_inundation.ipynb"


def test_region_lists_cover_every_region_in_the_csv():
    csv = pd.read_csv("score_time_series.csv")
    assert set(REGION_LISTS) == set(csv["State"])


def test_extracted_lists_have_the_documented_lengths():
    lists = extract_point_lists(NOTEBOOK)
    expected = {
        "MA_points": 28, "FL_points": 28, "CA_points": 28, "NY_points": 40,
        "PA_points": 24, "BOS_points": 28, "PA_points_new": 16,
        "MA_points_new": 12, "BOS_points_new": 12, "FL_points_new": 12,
        "CA_points_new": 12,
    }
    for name, count in expected.items():
        assert len(lists[name]) == count, name


def test_reconstructed_row_count_matches_the_csv():
    coords = reconstruct_coordinates(NOTEBOOK, "score_time_series.csv")
    assert len(coords) == 480


def test_every_region_gets_the_right_number_of_rows():
    coords = reconstruct_coordinates(NOTEBOOK, "score_time_series.csv")
    per_scenario = coords[coords["Scenario"] == "SSP585"]
    counts = per_scenario["State"].value_counts().to_dict()
    assert counts == {"BOS": 52, "NY": 40, "PA": 40, "FL": 40, "CA": 40, "MA": 28}


def test_reconstruction_raises_if_counts_disagree(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"2020": [1], "State": ["MA"], "Scenario": ["SSP585"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="count"):
        reconstruct_coordinates(NOTEBOOK, bad)


def test_alignment_confidence_is_positive_when_score_tracks_elevation():
    frame = pd.DataFrame(
        {
            "rule_score": list(range(10, 110, 10)),
            "elev_mean": [0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60],
        }
    )
    assert alignment_confidence(frame) > 0.5


def test_alignment_confidence_is_negative_when_the_mapping_is_shuffled():
    frame = pd.DataFrame(
        {
            "rule_score": list(range(100, 0, -10)),
            "elev_mean": [0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60],
        }
    )
    assert alignment_confidence(frame) < 0


def test_alignment_confidence_is_nan_below_ten_rows():
    frame = pd.DataFrame({"rule_score": [1, 2, 3], "elev_mean": [1, 2, 3]})
    assert np.isnan(alignment_confidence(frame))
