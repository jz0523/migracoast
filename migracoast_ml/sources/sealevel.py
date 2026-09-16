"""NOAA 2022 Sea Level Rise Technical Report projections."""

import zipfile
from pathlib import Path

import pandas as pd
import requests

from migracoast_ml.config import HTTP_HEADERS, RAW_DIR, SLR_SCENARIO, ensure_dirs
from migracoast_ml.sources._io import atomic_write_bytes

SLR_URL = "https://oceanservice.noaa.gov/hazards/sealevelrise/Sea_Level_Rise_Datasets_2022.zip"

# The published file carries 17 lines of citation text before the real header.
HEADER_ROW = 17

CM_TO_FEET = 0.0328084


def fetch_sealevel(force: bool = False) -> Path:
    """Download the NOAA sea level archive. The URL redirects to a CDN."""
    destination = RAW_DIR / "Sea_Level_Rise_Datasets_2022.zip"
    if destination.exists() and not force:
        return destination
    ensure_dirs()
    response = requests.get(SLR_URL, timeout=180, allow_redirects=True, headers=HTTP_HEADERS)
    response.raise_for_status()
    return atomic_write_bytes(destination, response.content)


def extract_projections_csv(zip_path: Path, target_dir: Path | None = None) -> Path:
    """Pull the sea level projections CSV out of the archive."""
    target_dir = target_dir or RAW_DIR
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if "Sea Level Projections" in n]
        if not names:
            raise ValueError("No sea level projections CSV inside the archive")
        out = target_dir / "slr_projections.csv"
        return atomic_write_bytes(out, archive.read(names[0]))


def parse_projections(csv_path: Path, scenario: str = SLR_SCENARIO) -> pd.DataFrame:
    """Return lat, lon and sea level rise in feet for 2050 and 2100."""
    raw = pd.read_csv(csv_path, skiprows=HEADER_ROW, low_memory=False)
    raw.columns = [str(c).strip() for c in raw.columns]

    if scenario not in set(raw["Scenario"].dropna()):
        raise ValueError(f"Unknown scenario {scenario!r}")

    rows = raw[raw["Scenario"] == scenario].copy()
    rows = rows.dropna(subset=["Lat", "Long"])

    # Some grid points carry zero in every projection year, meaning no
    # projection exists. Adding the baseline offset would turn that into a
    # fabricated rise, so those sites are excluded entirely.
    year_columns = [
        c for c in rows.columns
        if c.startswith("RSL") and c.endswith("(cm)") and "VLM" not in c
    ]
    projected = rows[year_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rows = rows[(projected != 0).any(axis=1)]

    offset = pd.to_numeric(rows["Offset 2000 to 2005 (cm)"], errors="coerce").fillna(0.0)
    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(rows["Lat"], errors="coerce"),
            "lon": pd.to_numeric(rows["Long"], errors="coerce"),
            "slr_2050_ft": (pd.to_numeric(rows["RSL2050 (cm)"], errors="coerce") + offset) * CM_TO_FEET,
            "slr_2100_ft": (pd.to_numeric(rows["RSL2100 (cm)"], errors="coerce") + offset) * CM_TO_FEET,
        }
    )
    out = out.dropna().reset_index(drop=True)
    # Longitudes arrive in either convention depending on the site.
    out["lon"] = out["lon"].where(out["lon"] <= 180, out["lon"] - 360)
    return out
