from pathlib import Path

import pandas as pd
import pytest

from migracoast_ml.sources import storms

FIXTURE = Path(__file__).parent / "fixtures" / "storm_index.html"


@pytest.fixture
def listing():
    return storms.parse_index(FIXTURE.read_text())


def test_finds_a_details_file_for_each_year(listing):
    assert listing[2000] == "StormEvents_details-ftp_v1.0_d2000_c20260323.csv.gz"
    assert listing[2024] == "StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz"


def test_matches_regardless_of_the_creation_date_suffix(listing):
    # 2000 and 2024 carry different c-dates; both must resolve.
    assert len({listing[2000], listing[2024]}) == 2


def test_ignores_fatalities_and_locations_files(listing):
    assert all("details" in name for name in listing.values())
    assert len(listing) == 3


def test_classify_splits_flood_and_storm_events():
    events = pd.DataFrame(
        {
            "EVENT_TYPE": ["Coastal Flood", "Hurricane", "Winter Storm", "Flash Flood"],
            "BEGIN_LAT": [42.0, 25.0, 40.0, 41.0],
            "BEGIN_LON": [-71.0, -80.0, -75.0, -70.0],
        }
    )
    out = storms.classify_events(events)
    assert sorted(out["kind"]) == ["flood", "flood", "storm"]


def test_events_without_coordinates_are_dropped():
    events = pd.DataFrame(
        {
            "EVENT_TYPE": ["Flood", "Flood"],
            "BEGIN_LAT": [42.0, None],
            "BEGIN_LON": [-71.0, -71.0],
        }
    )
    assert len(storms.classify_events(events)) == 1


def test_a_missing_year_fails_loudly(monkeypatch):
    """A skipped year would shrink the numerator while HazardIndex still
    divides by a fixed 25, silently understating hazard rates."""
    monkeypatch.setattr(
        storms,
        "parse_index",
        lambda html: {2000: "StormEvents_details-ftp_v1.0_d2000_c20260323.csv.gz"},
    )

    class Response:
        text = "<html></html>"

        def raise_for_status(self):
            return None

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, timeout=None):
            raise AssertionError("fetch_storms must not download when a year is missing")

    monkeypatch.setattr(storms.requests, "get", lambda url, timeout=None: Response())
    monkeypatch.setattr(storms.requests, "Session", Session)

    with pytest.raises(ValueError, match="2001"):
        storms.fetch_storms(years=(2000, 2001), force=True)
