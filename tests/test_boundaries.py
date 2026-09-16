import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from migracoast_ml.sources import boundaries


def test_urls_point_at_the_natural_earth_cdn():
    assert boundaries.STATES_URL.endswith("ne_10m_admin_1_states_provinces.zip")
    assert boundaries.COAST_URL.endswith("ne_10m_coastline.zip")


def test_filter_states_keeps_only_the_configured_five():
    gdf = gpd.GeoDataFrame(
        {
            "admin": ["United States of America"] * 6 + ["Canada"],
            "postal": ["MA", "NY", "PA", "FL", "CA", "TX", "ON"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1)])] * 7,
        },
        crs="EPSG:4326",
    )
    out = boundaries.filter_states(gdf)
    assert sorted(out.postal) == ["CA", "FL", "MA", "NY", "PA"]


def test_filter_states_raises_when_a_state_is_missing():
    gdf = gpd.GeoDataFrame(
        {
            "admin": ["United States of America"],
            "postal": ["MA"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1)])],
        },
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="missing"):
        boundaries.filter_states(gdf, required=("MA", "FL"))


def test_clip_coastline_drops_geometries_outside_the_region():
    coast = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)]), LineString([(50, 50), (51, 51)])]},
        crs="EPSG:4326",
    )
    out = boundaries.clip_coastline(coast, bounds=(-1, -1, 2, 2))
    assert len(out) == 1


def test_downloads_identify_the_client(monkeypatch, tmp_path):
    """NOAA rejects the default python-requests user agent with HTTP 403."""
    seen = {}

    class Response:
        content = b"payload"

        def raise_for_status(self):
            return None

    def fake_get(url, timeout=None, allow_redirects=None, headers=None):
        seen["headers"] = headers
        return Response()

    monkeypatch.setattr(boundaries.requests, "get", fake_get)
    boundaries._download("https://example.invalid/x.zip", tmp_path / "x.zip", force=True)
    assert "User-Agent" in seen["headers"]
    assert "python-requests" not in seen["headers"]["User-Agent"]
