"""FEMA NFIP redacted claims, the supervised target for the model."""

import io
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from migracoast_ml.config import HTTP_HEADERS, RAW_DIR, STATES, ensure_dirs
from migracoast_ml.features.grid import snap_to_cell
from migracoast_ml.sources._io import atomic_write_csv

# v2 (FimaNfipClaims) is deprecated and disappears on 2026-10-15.
FEMA_URL = "https://www.fema.gov/api/open/v3/NfipClaims"

PAGE_SIZE = 10000  # the documented maximum for $top

FEMA_FIELDS = (
    "latitude",
    "longitude",
    "state",
    "yearOfLoss",
    "dateOfLoss",
    "censusGeoid",
    "countyCode",
    "reportedZipCode",
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "ratedFloodZone",
)


def build_page_url(state: str, skip: int) -> str:
    """Compose one page request for a single state."""
    # $orderby pins a stable sort. Without it, paging with $skip relies on an
    # undocumented default order, and a shift between requests would duplicate
    # or drop claims with nothing downstream able to notice.
    query = urlencode(
        {
            "$filter": f"state eq '{state}'",
            "$orderby": "id",
            "$select": ",".join(FEMA_FIELDS),
            "$top": PAGE_SIZE,
            "$skip": skip,
            "$format": "csv",
        }
    )
    return f"{FEMA_URL}?{query}"


def fetch_claims(states=STATES, force: bool = False) -> Path:
    """Page through the API for each state and write one combined CSV."""
    ensure_dirs()
    destination = RAW_DIR / "nfip_claims.csv"
    if destination.exists() and not force:
        return destination

    frames = []
    with requests.Session() as session:
        for state in states:
            skip = 0
            while True:
                response = session.get(
                    build_page_url(state, skip), timeout=300, headers=HTTP_HEADERS
                )
                response.raise_for_status()
                page = pd.read_csv(io.StringIO(response.text), low_memory=False)
                if page.empty:
                    break
                frames.append(page)
                if len(page) < PAGE_SIZE:
                    break
                skip += PAGE_SIZE

    if not frames:
        raise ValueError(f"No claims returned for states: {tuple(states)}")

    atomic_write_csv(pd.concat(frames, ignore_index=True), destination)
    return destination


def count_claims_per_cell(claims: pd.DataFrame) -> pd.DataFrame:
    """Aggregate claims into 0.1 degree cells."""
    rows = claims.dropna(subset=["latitude", "longitude"]).copy()
    snapped = [snap_to_cell(a, o) for a, o in zip(rows["latitude"], rows["longitude"])]
    rows["cell_lat"] = [s[0] for s in snapped]
    rows["cell_lon"] = [s[1] for s in snapped]
    return (
        rows.groupby(["cell_lat", "cell_lon"])
        .size()
        .reset_index(name="claim_count")
    )
