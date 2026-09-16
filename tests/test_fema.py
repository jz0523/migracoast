from urllib.parse import parse_qs, urlparse

import pandas as pd

from migracoast_ml.sources import fema


def test_targets_the_v3_endpoint():
    # v2 FimaNfipClaims is deprecated and removed on 2026-10-15.
    assert fema.FEMA_URL.endswith("/api/open/v3/NfipClaims")
    assert "FimaNfipClaims" not in fema.FEMA_URL


def test_page_url_requests_csv_with_the_maximum_page_size():
    query = parse_qs(urlparse(fema.build_page_url("MA", skip=0)).query)
    assert query["$format"] == ["csv"]
    assert query["$top"] == ["10000"]
    assert query["$filter"] == ["state eq 'MA'"]


def test_page_url_advances_the_skip_offset():
    query = parse_qs(urlparse(fema.build_page_url("FL", skip=30000)).query)
    assert query["$skip"] == ["30000"]


def test_selected_fields_use_v3_names():
    selected = set(fema.FEMA_FIELDS)
    # v3 renamed these two columns.
    assert "censusGeoid" in selected
    assert "ratedFloodZone" in selected
    assert "censusTract" not in selected
    assert "floodZoneCurrent" not in selected


def test_claims_are_counted_per_cell():
    claims = pd.DataFrame(
        {
            "latitude": [42.36, 42.38, 25.77],
            "longitude": [-71.06, -71.09, -80.19],
            "state": ["MA", "MA", "FL"],
        }
    )
    counts = fema.count_claims_per_cell(claims)
    boston = counts[(counts.cell_lat == 42.4) & (counts.cell_lon == -71.1)]
    assert int(boston.claim_count.iloc[0]) == 2
    assert int(counts.claim_count.sum()) == 3


def test_rows_with_missing_coordinates_are_dropped():
    claims = pd.DataFrame(
        {
            "latitude": [42.36, None],
            "longitude": [-71.06, -71.0],
            "state": ["MA", "MA"],
        }
    )
    assert int(fema.count_claims_per_cell(claims).claim_count.sum()) == 1
