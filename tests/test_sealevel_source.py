from pathlib import Path

import pytest

from migracoast_ml.sources import sealevel

FIXTURE = Path(__file__).parent / "fixtures" / "slr_sample.csv"


@pytest.fixture
def projections():
    return sealevel.parse_projections(FIXTURE, scenario="1.0 - MED")


def test_skips_the_seventeen_junk_header_lines(projections):
    assert list(projections.columns) == ["lat", "lon", "slr_2050_ft", "slr_2100_ft"]


def test_keeps_only_the_requested_scenario(projections):
    # Boston has three scenario rows in the fixture; only one survives.
    boston = projections[(projections.lat == 42.4) & (projections.lon == -71.1)]
    assert len(boston) == 1


def test_drops_rows_without_coordinates(projections):
    # The GMSL row has empty Lat and Long and must not appear.
    assert len(projections) == 2


def test_converts_centimetres_to_feet_and_adds_the_offset(projections):
    boston = projections[projections.lat == 42.4].iloc[0]
    # RSL2050 is 30 cm, the 2000-to-2005 offset is 3 cm, so 33 cm.
    assert boston.slr_2050_ft == pytest.approx(33 * 0.0328084, rel=1e-6)
    # RSL2100 is 76 cm plus the same 3 cm offset.
    assert boston.slr_2100_ft == pytest.approx(79 * 0.0328084, rel=1e-6)


def test_unknown_scenario_raises(projections):
    with pytest.raises(ValueError, match="scenario"):
        sealevel.parse_projections(FIXTURE, scenario="9.9 - MED")


def test_drops_sites_with_no_projection(projections):
    # A site whose every RSL year is zero has no projection. Adding the
    # baseline offset to it would fabricate a small positive rise.
    placeholder = projections[(projections.lat == 40.0) & (projections.lon == -79.0)]
    assert placeholder.empty
