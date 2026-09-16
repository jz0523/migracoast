import numpy as np
import pandas as pd
import pytest

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.dataset import assemble, coastal_mask, target_from_counts


def test_target_is_log1p_of_claim_count():
    counts = pd.Series([0, 1, 9])
    assert list(target_from_counts(counts)) == pytest.approx(
        [0.0, np.log(2.0), np.log(10.0)]
    )


def test_cells_with_no_claims_are_kept_as_zero_targets():
    cells = pd.DataFrame({"cell_lat": [42.4, 44.0], "cell_lon": [-71.1, -69.0], "state": ["MA", "MA"]})
    counts = pd.DataFrame({"cell_lat": [42.4], "cell_lon": [-71.1], "claim_count": [7]})
    features = pd.DataFrame(
        {name: [1.0, 2.0] for name in FEATURE_NAMES}
    )
    out = assemble(cells, counts, features)
    assert len(out) == 2
    assert int(out.loc[out.cell_lat == 44.0, "claim_count"].iloc[0]) == 0


def test_assemble_adds_block_and_target_columns():
    cells = pd.DataFrame({"cell_lat": [42.4], "cell_lon": [-71.1], "state": ["MA"]})
    counts = pd.DataFrame({"cell_lat": [42.4], "cell_lon": [-71.1], "claim_count": [3]})
    features = pd.DataFrame({name: [1.0] for name in FEATURE_NAMES})
    out = assemble(cells, counts, features)
    assert "block" in out.columns
    assert "target" in out.columns
    assert out.target.iloc[0] == pytest.approx(np.log(4.0))


def test_assemble_preserves_every_feature_column():
    cells = pd.DataFrame({"cell_lat": [42.4], "cell_lon": [-71.1], "state": ["MA"]})
    counts = pd.DataFrame(columns=["cell_lat", "cell_lon", "claim_count"])
    features = pd.DataFrame({name: [1.0] for name in FEATURE_NAMES})
    out = assemble(cells, counts, features)
    for name in FEATURE_NAMES:
        assert name in out.columns


def test_coastal_mask_uses_the_configured_threshold():
    frame = pd.DataFrame({"coast_km": [5.0, 29.9, 30.1, 500.0]})
    assert list(coastal_mask(frame)) == [True, True, False, False]


def test_each_cell_keeps_its_own_features():
    """Features are attached positionally. If the order slipped, every cell
    would carry another cell's data and nothing else would notice."""
    cells = pd.DataFrame(
        {
            "cell_lat": [42.4, 25.8, 40.8],
            "cell_lon": [-71.1, -80.1, -77.8],
            "state": ["MA", "FL", "PA"],
        }
    )
    counts = pd.DataFrame(columns=["cell_lat", "cell_lon", "claim_count"])
    # Each row's features are derived from its own latitude, so a shifted or
    # reversed join shows up immediately.
    features = pd.DataFrame(
        {name: [lat * 100 + i for lat in cells["cell_lat"]] for i, name in enumerate(FEATURE_NAMES)}
    )

    out = assemble(cells, counts, features)

    for i, name in enumerate(FEATURE_NAMES):
        assert list(out[name]) == [42.4 * 100 + i, 25.8 * 100 + i, 40.8 * 100 + i]
