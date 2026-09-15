import pytest

from migracoast_ml.features.grid import block_id, snap_to_cell


def test_snap_rounds_to_the_nearest_tenth():
    assert snap_to_cell(42.3612, -71.0589) == (42.4, -71.1)


def test_snap_is_idempotent():
    once = snap_to_cell(25.7743, -80.1937)
    assert snap_to_cell(*once) == once


def test_snap_handles_negative_longitude_consistently():
    assert snap_to_cell(25.0, -80.04) == (25.0, -80.0)
    assert snap_to_cell(25.0, -80.06) == (25.0, -80.1)


def test_every_point_in_a_cell_snaps_to_the_same_centre():
    centre = snap_to_cell(41.68, -70.75)
    for dlat in (-0.04, 0.0, 0.04):
        for dlon in (-0.04, 0.0, 0.04):
            assert snap_to_cell(centre[0] + dlat, centre[1] + dlon) == centre


def test_block_id_groups_neighbouring_cells_together():
    assert block_id(42.3, -71.1) == block_id(42.9, -71.9)
    assert block_id(42.3, -71.1) != block_id(43.1, -71.1)


def test_block_id_is_a_string():
    assert isinstance(block_id(42.3, -71.1), str)


@pytest.mark.parametrize("lat,lon", [(90.0, 180.0), (-90.0, -180.0)])
def test_snap_tolerates_extreme_coordinates(lat, lon):
    out = snap_to_cell(lat, lon)
    assert len(out) == 2
