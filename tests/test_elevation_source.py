from urllib.parse import parse_qs, urlparse

from migracoast_ml.sources import elevation


def test_uses_the_thirty_arc_second_dataset():
    # The 60 arc-second dataset returns HTTP 500 from this service.
    assert "30s" in elevation.ETOPO_URL
    assert "60s" not in elevation.ETOPO_URL


def test_request_url_carries_the_bounding_box():
    url = elevation.build_url(south=24.3, north=31.1, west=-87.7, east=-79.8)
    query = parse_qs(urlparse(url).query)
    assert query["var"] == ["z"]
    assert float(query["south"][0]) == 24.3
    assert float(query["north"][0]) == 31.1
    assert float(query["west"][0]) == -87.7
    assert float(query["east"][0]) == -79.8


def test_accept_format_is_netcdf_not_netcdf4():
    # accept=netcdf4 returns HTTP 400 "Format netcdf4 is not supported".
    query = parse_qs(urlparse(elevation.build_url(0, 1, 0, 1)).query)
    assert query["accept"] == ["netcdf"]


def test_bounds_are_padded_so_edge_cells_have_full_windows():
    padded = elevation.pad_bounds((-71.0, 42.0, -70.0, 43.0), pad=0.2)
    assert padded == (41.8, 43.2, -71.2, -69.8)
