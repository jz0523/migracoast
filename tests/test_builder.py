import numpy as np
import pandas as pd
import pytest

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.features.builder import FeatureBuilder


class StubElevation:
    def features(self, lat, lon, half=0.05):
        return {
            "elev_mean": 4.0,
            "elev_min": 1.0,
            "elev_p10": 2.0,
            "frac_below_2m": 0.25,
            "frac_below_5m": 0.75,
            "land_fraction": 0.9,
        }


class StubCoast:
    def distance_km(self, lat, lon):
        return 12.5


class StubSeaLevel:
    def features(self, lat, lon):
        return {"slr_2050_ft": 1.08, "slr_2100_ft": 2.59}


class StubHazards:
    def features(self, lat, lon):
        return {"flood_events_per_year": 0.08, "storm_events_per_year": 0.16}


@pytest.fixture
def builder():
    return FeatureBuilder(StubElevation(), StubCoast(), StubSeaLevel(), StubHazards())


def test_build_returns_exactly_the_configured_features(builder):
    feats = builder.build(42.36, -71.06)
    assert set(feats) == set(FEATURE_NAMES)


def test_build_excludes_geography(builder):
    feats = builder.build(42.36, -71.06)
    for banned in ("lat", "lon", "state", "latitude", "longitude"):
        assert banned not in feats


def test_every_point_inside_a_cell_yields_an_identical_vector(builder):
    """The invariant that eliminates training and serving skew."""
    centre = builder.build(42.4, -71.1)
    for dlat in (-0.04, 0.0, 0.04):
        for dlon in (-0.04, 0.0, 0.04):
            assert builder.build(42.4 + dlat, -71.1 + dlon) == centre


def test_build_frame_preserves_row_order_and_columns(builder):
    frame = builder.build_frame([(42.4, -71.1), (25.8, -80.2)])
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert len(frame) == 2


def test_build_frame_on_empty_input_returns_empty_frame_with_columns(builder):
    frame = builder.build_frame([])
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert frame.empty


def test_values_flow_through_from_the_providers(builder):
    feats = builder.build(42.4, -71.1)
    assert feats["coast_km"] == pytest.approx(12.5)
    assert feats["slr_2100_ft"] == pytest.approx(2.59)
    assert feats["storm_events_per_year"] == pytest.approx(0.16)
    assert feats["elev_mean"] == pytest.approx(4.0)


class VaryingElevation:
    """Returns values derived from the coordinates it is given."""

    def features(self, lat, lon, half=0.05):
        return {
            "elev_mean": lat + lon,
            "elev_min": lat,
            "elev_p10": lon,
            "frac_below_2m": 0.25,
            "frac_below_5m": 0.75,
            "land_fraction": 0.9,
        }


class VaryingCoast:
    def distance_km(self, lat, lon):
        return lat * 1000 + lon


class VaryingSeaLevel:
    def features(self, lat, lon):
        return {"slr_2050_ft": lat, "slr_2100_ft": lon}


class VaryingHazards:
    def features(self, lat, lon):
        return {"flood_events_per_year": lat - lon, "storm_events_per_year": 0.16}


@pytest.fixture
def varying_builder():
    return FeatureBuilder(VaryingElevation(), VaryingCoast(), VaryingSeaLevel(), VaryingHazards())


def test_parity_holds_when_providers_depend_on_their_input(varying_builder):
    """The real parity guarantee.

    The other parity test uses constant-returning stubs, so it cannot tell
    whether build() snapped its input. These providers vary with the
    coordinates they receive, so this test fails if the snap is removed.
    """
    centre = varying_builder.build(42.4, -71.1)
    for dlat in (-0.04, 0.0, 0.04):
        for dlon in (-0.04, 0.0, 0.04):
            assert varying_builder.build(42.4 + dlat, -71.1 + dlon) == centre
