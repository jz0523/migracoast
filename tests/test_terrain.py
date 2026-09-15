import numpy as np
import pytest

from migracoast_ml.features.terrain import ElevationGrid, terrain_features


def test_all_ocean_window_reports_no_land():
    feats = terrain_features(np.full((12, 12), -5.0))
    assert feats["land_fraction"] == pytest.approx(0.0)
    assert np.isnan(feats["elev_mean"])
    assert np.isnan(feats["frac_below_2m"])


def test_empty_window_reports_nan_everywhere():
    feats = terrain_features(np.array([]))
    assert np.isnan(feats["land_fraction"])
    assert np.isnan(feats["elev_mean"])


def test_flat_land_window_has_known_statistics():
    feats = terrain_features(np.full((10, 10), 3.0))
    assert feats["land_fraction"] == pytest.approx(1.0)
    assert feats["elev_mean"] == pytest.approx(3.0)
    assert feats["elev_min"] == pytest.approx(3.0)
    assert feats["elev_p10"] == pytest.approx(3.0)
    assert feats["frac_below_2m"] == pytest.approx(0.0)
    assert feats["frac_below_5m"] == pytest.approx(1.0)


def test_half_ocean_half_land_window():
    window = np.concatenate([np.full(50, -2.0), np.full(50, 10.0)])
    feats = terrain_features(window)
    assert feats["land_fraction"] == pytest.approx(0.5)
    # Statistics ignore the ocean half.
    assert feats["elev_mean"] == pytest.approx(10.0)


def test_low_lying_land_is_detected():
    window = np.array([0.5, 1.0, 1.5, 3.0, 4.0, 10.0])
    feats = terrain_features(window)
    assert feats["frac_below_2m"] == pytest.approx(3 / 6)
    assert feats["frac_below_5m"] == pytest.approx(5 / 6)


def test_masked_values_are_excluded():
    window = np.ma.masked_array([1.0, 2.0, -99999.0], mask=[False, False, True])
    feats = terrain_features(window)
    assert feats["land_fraction"] == pytest.approx(1.0)
    assert feats["elev_mean"] == pytest.approx(1.5)


def test_grid_returns_a_window_around_a_point(tmp_path):
    grid = _synthetic_grid(tmp_path)
    window = grid.window(42.0, -71.0, half=0.05)
    assert window.size > 0


def test_grid_returns_empty_outside_coverage(tmp_path):
    grid = _synthetic_grid(tmp_path)
    assert grid.window(10.0, 10.0, half=0.05).size == 0


def _synthetic_grid(tmp_path):
    """Write a small NetCDF covering 41.5-42.5 N, 71.5-70.5 W and load it."""
    import netCDF4

    path = tmp_path / "etopo_TEST.nc"
    lats = np.arange(41.5, 42.5, 1 / 120)
    lons = np.arange(-71.5, -70.5, 1 / 120)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", len(lats))
        ds.createDimension("lon", len(lons))
        lat_var = ds.createVariable("lat", "f8", ("lat",))
        lat_var[:] = lats
        lon_var = ds.createVariable("lon", "f8", ("lon",))
        lon_var[:] = lons
        z_var = ds.createVariable("z", "f4", ("lat", "lon"))
        z_var[:, :] = np.full((len(lats), len(lons)), 4.0)
    return ElevationGrid.from_dir(tmp_path)
