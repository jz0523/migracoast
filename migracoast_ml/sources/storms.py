"""NOAA Storm Events, the source of hazard history features."""

import gzip
import io
import re
from pathlib import Path

import pandas as pd
import requests

from migracoast_ml.config import (
    FLOOD_EVENT_TYPES,
    HTTP_HEADERS,
    RAW_DIR,
    STORM_EVENT_TYPES,
    STORM_YEARS,
    ensure_dirs,
)
from migracoast_ml.sources._io import atomic_write_csv

STORM_INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

# The trailing creation date changes whenever NOAA republishes a year, so the
# year is matched by pattern and the suffix is captured rather than assumed.
DETAILS_PATTERN = re.compile(
    r"StormEvents_details-ftp_v1\.0_d(\d{4})_c\d{8}\.csv\.gz"
)

USE_COLUMNS = (
    "BEGIN_LAT",
    "BEGIN_LON",
    "EVENT_TYPE",
    "EVENT_ID",
    "BEGIN_DATE_TIME",
    "CZ_NAME",
    "CZ_TYPE",
    "STATE",
    "YEAR",
)


def parse_index(html: str) -> dict[int, str]:
    """Map each year to its details filename from the directory listing."""
    found = {}
    for match in DETAILS_PATTERN.finditer(html):
        found[int(match.group(1))] = match.group(0)
    return found


def fetch_storms(years=STORM_YEARS, force: bool = False) -> Path:
    """Download the requested years and write one combined CSV."""
    ensure_dirs()
    destination = RAW_DIR / "storm_events.csv"
    if destination.exists() and not force:
        return destination

    index = parse_index(
        requests.get(STORM_INDEX_URL, timeout=180, headers=HTTP_HEADERS).text
    )

    # A year silently missing from the listing would shrink the event count
    # while HazardIndex still divides by the full span, understating rates.
    missing = [year for year in years if year not in index]
    if missing:
        raise ValueError(f"Storm Events listing is missing years: {missing}")

    frames = []
    with requests.Session() as session:
        for year in years:
            filename = index[year]
            response = session.get(
                STORM_INDEX_URL + filename, timeout=300, headers=HTTP_HEADERS
            )
            response.raise_for_status()
            text = gzip.decompress(response.content).decode("utf-8", "replace")
            frame = pd.read_csv(io.StringIO(text), low_memory=False)
            keep = [c for c in USE_COLUMNS if c in frame.columns]
            frames.append(frame[keep])

    atomic_write_csv(pd.concat(frames, ignore_index=True), destination)
    return destination


def classify_events(events: pd.DataFrame) -> pd.DataFrame:
    """Keep georeferenced flood and storm events, labelled by kind.

    Events without coordinates are dropped. The rule engine also matched
    events by county name, but that introduces a population-size bias the
    model would exploit, so it is deliberately not reproduced here.
    """
    rows = events.dropna(subset=["BEGIN_LAT", "BEGIN_LON"]).copy()
    kind = pd.Series(pd.NA, index=rows.index, dtype="object")
    kind[rows["EVENT_TYPE"].isin(FLOOD_EVENT_TYPES)] = "flood"
    kind[rows["EVENT_TYPE"].isin(STORM_EVENT_TYPES)] = "storm"
    rows["kind"] = kind
    rows = rows.dropna(subset=["kind"])
    return pd.DataFrame(
        {
            "lat": rows["BEGIN_LAT"].astype(float),
            "lon": rows["BEGIN_LON"].astype(float),
            "kind": rows["kind"].astype(str),
        }
    ).reset_index(drop=True)
