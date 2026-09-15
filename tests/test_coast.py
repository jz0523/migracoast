import geopandas as gpd
import pytest
from shapely.geometry import LineString

from migracoast_ml.features.coast import CoastlineIndex, haversine_km


def test_haversine_matches_a_known_distance():
    # Boston to New York, roughly 306 km.
    assert haversine_km(42.36, -71.06, 40.71, -74.01) == pytest.approx(306, abs=5)


def test_haversine_is_zero_for_identical_points():
    assert haversine_km(25.0, -80.0, 25.0, -80.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_is_symmetric():
    forward = haversine_km(42.0, -71.0, 25.0, -80.0)
    backward = haversine_km(25.0, -80.0, 42.0, -71.0)
    assert forward == pytest.approx(backward, rel=1e-9)


@pytest.fixture
def index():
    # A coastline running due north along the 70 W meridian.
    coast = gpd.GeoDataFrame(
        {"geometry": [LineString([(-70.0, 40.0), (-70.0, 45.0)])]},
        crs="EPSG:4326",
    )
    return CoastlineIndex(coast)


def test_distance_is_zero_on_the_coastline(index):
    assert index.distance_km(42.0, -70.0) == pytest.approx(0.0, abs=0.5)


def test_distance_grows_with_separation(index):
    near = index.distance_km(42.0, -70.5)
    far = index.distance_km(42.0, -71.5)
    assert near < far


def test_one_degree_of_longitude_at_forty_two_north(index):
    # cos(42 deg) * 111.32 km is about 82.7 km.
    assert index.distance_km(42.0, -71.0) == pytest.approx(82.7, abs=2.0)
