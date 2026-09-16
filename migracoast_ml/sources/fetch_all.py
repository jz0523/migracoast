"""Download every external data source in dependency order."""

from migracoast_ml.sources import boundaries, elevation, fema, sealevel, storms


def fetch_all(force: bool = False) -> dict:
    """Run every fetcher. Elevation depends on the state polygons."""
    states_path, coast_path = boundaries.fetch_boundaries(force=force)
    states = boundaries.load_states(states_path)

    slr_zip = sealevel.fetch_sealevel(force=force)
    slr_csv = sealevel.extract_projections_csv(slr_zip)

    elevation_paths = elevation.fetch_elevation(states, force=force)
    claims_path = fema.fetch_claims(force=force)
    storms_path = storms.fetch_storms(force=force)

    return {
        "states": states_path,
        "coastline": coast_path,
        "sealevel": slr_csv,
        "elevation": elevation_paths,
        "claims": claims_path,
        "storms": storms_path,
    }
