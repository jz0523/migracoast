# Learned Coastal Flood Risk Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a machine learning system to Migracoast that learns coastal flood risk from observed FEMA flood insurance claims and serves point and heatmap predictions through a Flask API.

**Architecture:** Five scripted data fetchers write to `data/raw/`. A feature layer computes eleven physical features for any 0.1 degree grid cell through a single code path shared by training and serving. Three scikit-learn models compete under spatial cross-validation; the winner is serialized and served by a Flask blueprint that runs standalone.

**Tech Stack:** Python 3.13, scikit-learn, pandas, numpy, geopandas, shapely, netCDF4, Flask, pytest.

**Design spec:** `docs/superpowers/specs/2026-09-15-ml-flood-risk-design.md`

---

## Conventions for every task

**Commit messages.** Every commit in this plan ends with this trailer line, after a blank line:

```
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

The commit commands below show only the subject line. Add the trailer to each one.

**Running commands.** All commands assume the repository root as the working directory and the project virtual environment active. On Windows the interpreter is `.venv/Scripts/python.exe`; a POSIX shell uses `.venv/bin/python`. The plan writes `python` and means the virtual environment's interpreter.

**Test isolation.** No test may reach the network. Fetchers are tested against recorded fixtures.

---

## Verified environment facts

These were confirmed on 2026-09-15 against the live services. Do not re-derive them.

| Fact | Value |
|---|---|
| System Python | 3.13.7, with a broken install where pandas is present but numpy is missing |
| Fix | A clean virtual environment; all packages install without compilers |
| FEMA endpoint | `https://www.fema.gov/api/open/v3/NfipClaims`, max `$top` 10000 |
| FEMA claim counts | MA 35,375, NY 175,430, PA 77,060, FL 448,618, CA 53,563 |
| ETOPO subset | 30 arc-second only; `accept=netcdf` works, `accept=netcdf4` returns HTTP 400; the 60 arc-second dataset returns HTTP 500 |
| ETOPO window | A 0.1 degree cell covers roughly 12 by 12 pixels |
| Sea level CSV | Real header at line index 17; values in centimeters |
| Natural Earth states | The `postal` column holds two-letter codes; `admin` equals "United States of America" |
| Coastline rows | 4,133 global geometries in EPSG:4326 |

---

## File structure

```
migracoast_ml/
  __init__.py            package marker
  __main__.py            CLI with fetch/build/train/evaluate/baseline/serve
  config.py              paths, states, feature names, constants
  sources/
    __init__.py
    boundaries.py        Natural Earth states and coastline
    sealevel.py          NOAA 2022 sea level projections
    elevation.py         ETOPO 2022 via THREDDS
    fema.py              NFIP claims
    storms.py            NOAA Storm Events
    fetch_all.py         orchestrator
  features/
    __init__.py
    grid.py              cell snapping, block ids, cell universe
    terrain.py           ElevationGrid
    coast.py             CoastlineIndex, haversine_km
    sealevel.py          SeaLevelIndex
    hazards.py           HazardIndex
    builder.py           FeatureBuilder, the single shared code path
  dataset.py             joins claims to cells and features
  models/
    __init__.py
    ladder.py            the three competing models
    evaluate.py          spatial cross-validation
    train.py             fit, select, serialize
    scoring.py           RiskScorer, percentile scoring
    explain.py           permutation importance and SHAP
  baseline.py            rule-engine score versus learned score
  api.py                 Flask blueprint
  serve.py               standalone server
tests/
  fixtures/              committed offline test data
  test_*.py
notebooks/results.ipynb
data/raw/, data/processed/   gitignored
models/                      committed artifacts
```

**Interfaces locked here.** Later tasks depend on these exact names.

```python
# features/grid.py
snap_to_cell(lat, lon) -> tuple[float, float]
block_id(lat, lon, deg=1.0) -> str
build_cell_universe(states_gdf) -> pd.DataFrame   # cell_lat, cell_lon, state

# features/coast.py
haversine_km(lat1, lon1, lat2, lon2) -> float
CoastlineIndex(coast_gdf).distance_km(lat, lon) -> float

# features/terrain.py
ElevationGrid.from_dir(directory) -> ElevationGrid
ElevationGrid.window(lat, lon, half) -> np.ndarray
ElevationGrid.features(lat, lon, half) -> dict

# features/sealevel.py
SeaLevelIndex(df).features(lat, lon) -> dict

# features/hazards.py
HazardIndex(events_df, n_years).features(cell_lat, cell_lon) -> dict

# features/builder.py
FeatureBuilder(elevation, coast, sealevel, hazards)
FeatureBuilder.load() -> FeatureBuilder
FeatureBuilder.build(lat, lon) -> dict        # snaps to cell first
FeatureBuilder.build_frame(points) -> pd.DataFrame

# models/scoring.py
RiskScorer(model, reference, feature_names, metadata)
RiskScorer.score_features(feats: dict) -> dict
RiskScorer.save(path) / RiskScorer.load(path)
```

**Critical invariant.** `FeatureBuilder.build` snaps its input to the containing 0.1 degree cell centre before computing anything. Training and serving therefore run identical arithmetic, and a query anywhere inside a cell returns that cell's score. Task 12 tests this invariant directly.

---

## Task 1: Repository scaffolding and configuration

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `migracoast_ml/__init__.py`
- Create: `migracoast_ml/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create the virtual environment and install dependencies**

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
```

Create `requirements.txt` with the versions verified on 2026-09-15:

```
numpy==2.5.3
pandas==3.0.5
scipy==1.18.1
scikit-learn==1.9.1
joblib==1.6.0
geopandas==1.1.4
shapely==2.1.2
pyogrio==0.13.0
pyproj==3.8.0
netCDF4==1.7.4
requests==2.34.2
Flask==3.1.3
flask-cors==6.0.5
matplotlib==3.11.2
pytest==9.1.1
```

Then install:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

- [ ] **Step 2: Create `.gitignore`**

The repository has none today. Create it before any data is downloaded.

```
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.env
data/raw/
data/processed/
*.nc
*.zip
.ipynb_checkpoints/
```

- [ ] **Step 3: Write the failing test**

Create `tests/__init__.py` as an empty file. Create `tests/test_config.py`:

```python
from migracoast_ml import config


def test_states_are_the_five_covered_regions():
    assert config.STATES == ("MA", "NY", "PA", "FL", "CA")


def test_cell_geometry_matches_fema_rounding():
    assert config.CELL_SIZE == 0.1
    assert config.CELL_HALF == 0.05


def test_feature_names_has_eleven_entries_and_no_geography():
    assert len(config.FEATURE_NAMES) == 11
    assert len(set(config.FEATURE_NAMES)) == 11
    for banned in ("lat", "lon", "state"):
        assert not any(name == banned for name in config.FEATURE_NAMES)


def test_directories_are_under_the_repository_root():
    assert config.RAW_DIR.parts[-2:] == ("data", "raw")
    assert config.PROCESSED_DIR.parts[-2:] == ("data", "processed")
    assert config.MODELS_DIR.name == "models"
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml'`

- [ ] **Step 5: Write the implementation**

Create `migracoast_ml/__init__.py` as an empty file. Create `migracoast_ml/config.py`:

```python
"""Central configuration for the Migracoast ML pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

STATES = ("MA", "NY", "PA", "FL", "CA")

# FEMA rounds claim coordinates to one decimal place, which fixes the
# spatial resolution of the whole project at 0.1 degrees.
CELL_SIZE = 0.1
CELL_HALF = 0.05

# Cross-validation groups cells into blocks this many degrees wide so that
# a cell's neighbour can never land in the opposite fold.
BLOCK_DEG = 1.0

# A cell is "coastal" when its centre is within this distance of the coast.
COASTAL_KM = 30.0

# NOAA 2022 Intermediate scenario at the median (50th percentile).
SLR_SCENARIO = "1.0 - MED"

STORM_YEARS = tuple(range(2000, 2025))

FLOOD_EVENT_TYPES = ("Flood", "Flash Flood", "Coastal Flood")
STORM_EVENT_TYPES = ("Thunderstorm Wind", "Tornado", "Hail", "Hurricane")

# Latitude, longitude and state are deliberately absent. Including them
# would let a tree model memorise geography instead of learning physics.
FEATURE_NAMES = (
    "elev_mean",
    "elev_min",
    "elev_p10",
    "frac_below_2m",
    "frac_below_5m",
    "land_fraction",
    "coast_km",
    "slr_2050_ft",
    "slr_2100_ft",
    "flood_events_per_year",
    "storm_events_per_year",
)


def ensure_dirs() -> None:
    """Create the data and model directories if they do not exist."""
    for path in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: PASS, 4 passed

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt migracoast_ml/ tests/
git commit -m "feat: add ML package scaffolding and configuration"
```

---

## Task 2: Grid helpers

**Files:**
- Create: `migracoast_ml/features/__init__.py`
- Create: `migracoast_ml/features/grid.py`
- Create: `tests/test_grid.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_grid.py`:

```python
import pytest

from migracoast_ml.features.grid import block_id, snap_to_cell


def test_snap_rounds_to_the_nearest_tenth():
    assert snap_to_cell(42.3612, -71.0589) == (42.4, -71.1)


def test_snap_is_idempotent():
    once = snap_to_cell(25.7743, -80.1937)
    assert snap_to_cell(*once) == once


def test_snap_handles_negative_longitude_consistently():
    assert snap_to_cell(25.0, -80.04) == (25.0, -80.0)
    assert snap_to_cell(25.0, -80.06) == (25.0, -80.1)


def test_every_point_in_a_cell_snaps_to_the_same_centre():
    centre = snap_to_cell(41.68, -70.75)
    for dlat in (-0.04, 0.0, 0.04):
        for dlon in (-0.04, 0.0, 0.04):
            assert snap_to_cell(centre[0] + dlat, centre[1] + dlon) == centre


def test_block_id_groups_neighbouring_cells_together():
    assert block_id(42.3, -71.1) == block_id(42.9, -71.9)
    assert block_id(42.3, -71.1) != block_id(43.1, -71.1)


def test_block_id_is_a_string():
    assert isinstance(block_id(42.3, -71.1), str)


@pytest.mark.parametrize("lat,lon", [(90.0, 180.0), (-90.0, -180.0)])
def test_snap_tolerates_extreme_coordinates(lat, lon):
    out = snap_to_cell(lat, lon)
    assert len(out) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.features'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/features/__init__.py` as an empty file. Create `migracoast_ml/features/grid.py`:

```python
"""Grid cell arithmetic shared by training and serving."""

import math

import pandas as pd

from migracoast_ml.config import BLOCK_DEG, CELL_SIZE


def snap_to_cell(lat: float, lon: float) -> tuple[float, float]:
    """Return the centre of the 0.1 degree cell containing this point.

    Rounding to one decimal place mirrors how FEMA redacts claim
    coordinates, so a claim and the cell it belongs to always agree.
    """
    return (round(round(lat / CELL_SIZE) * CELL_SIZE, 1),
            round(round(lon / CELL_SIZE) * CELL_SIZE, 1))


def block_id(lat: float, lon: float, deg: float = BLOCK_DEG) -> str:
    """Return the identifier of the cross-validation block holding this point."""
    return f"{math.floor(lat / deg)}_{math.floor(lon / deg)}"


def build_cell_universe(states_gdf) -> pd.DataFrame:
    """Enumerate every 0.1 degree cell whose centre lies inside a state.

    Returns a frame with cell_lat, cell_lon and state columns.
    """
    from shapely.geometry import Point

    rows = []
    for _, state_row in states_gdf.iterrows():
        geom = state_row.geometry
        min_lon, min_lat, max_lon, max_lat = geom.bounds
        lat = round(math.floor(min_lat / CELL_SIZE) * CELL_SIZE, 1)
        while lat <= max_lat + CELL_SIZE:
            lon = round(math.floor(min_lon / CELL_SIZE) * CELL_SIZE, 1)
            while lon <= max_lon + CELL_SIZE:
                if geom.contains(Point(lon, lat)):
                    rows.append((lat, lon, state_row.postal))
                lon = round(lon + CELL_SIZE, 1)
            lat = round(lat + CELL_SIZE, 1)

    frame = pd.DataFrame(rows, columns=["cell_lat", "cell_lon", "state"])
    return frame.drop_duplicates(subset=["cell_lat", "cell_lon"]).reset_index(drop=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_grid.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/features/ tests/test_grid.py
git commit -m "feat: add grid cell snapping and block grouping"
```

---

## Task 3: Natural Earth boundaries source

**Files:**
- Create: `migracoast_ml/sources/__init__.py`
- Create: `migracoast_ml/sources/boundaries.py`
- Create: `tests/test_boundaries.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_boundaries.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_boundaries.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.sources'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/sources/__init__.py` as an empty file. Create `migracoast_ml/sources/boundaries.py`:

```python
"""Natural Earth state polygons and coastline geometry."""

from pathlib import Path

import geopandas as gpd
import requests

from migracoast_ml.config import RAW_DIR, STATES, ensure_dirs

STATES_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
COAST_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"


def _download(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        return destination
    ensure_dirs()
    response = requests.get(url, timeout=180, allow_redirects=True)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def fetch_boundaries(force: bool = False) -> tuple[Path, Path]:
    """Download the state and coastline archives. Returns their paths."""
    states = _download(STATES_URL, RAW_DIR / "ne_10m_admin_1_states_provinces.zip", force)
    coast = _download(COAST_URL, RAW_DIR / "ne_10m_coastline.zip", force)
    return states, coast


def filter_states(gdf, required=STATES):
    """Keep only the US states named in `required`, and verify all are present."""
    us = gdf[gdf["admin"] == "United States of America"]
    out = us[us["postal"].isin(required)].copy()
    missing = set(required) - set(out["postal"])
    if missing:
        raise ValueError(f"State boundaries missing for: {sorted(missing)}")
    return out.reset_index(drop=True)


def clip_coastline(gdf, bounds):
    """Keep coastline geometries intersecting (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    return gdf.cx[min_lon:max_lon, min_lat:max_lat].reset_index(drop=True)


def load_states(path: Path | None = None):
    path = path or (RAW_DIR / "ne_10m_admin_1_states_provinces.zip")
    return filter_states(gpd.read_file(f"zip://{path}"))


def load_coastline(path: Path | None = None, bounds=None):
    path = path or (RAW_DIR / "ne_10m_coastline.zip")
    gdf = gpd.read_file(f"zip://{path}")
    return clip_coastline(gdf, bounds) if bounds else gdf
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_boundaries.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/sources/ tests/test_boundaries.py
git commit -m "feat: add Natural Earth boundary fetcher"
```

---

## Task 3b: Crash-safe writes for fetchers

> **Added after review of Task 3. Already implemented** (see the corresponding
> commit in the repository history) — this section exists to keep the plan's
> record of the codebase accurate, not to be executed again.

A review of Task 3 found a real correctness risk: `boundaries._download`
writes downloaded bytes directly to their final path, and skips the download
whenever that path already exists. If the process is killed mid-write, a
truncated file is left at the final path, and every later run silently
reuses it forever. The fix is one shared helper that writes to a sibling
`.part` file and renames it into place, so the destination only ever holds a
complete file. `boundaries.py` is retrofitted to use it here; the four
fetchers written later in this plan (Tasks 4, 7, 9 and 10) use it from the
start.

**Files:**
- Create: `migracoast_ml/sources/_io.py`
- Create: `tests/test_io.py`
- Modify: `migracoast_ml/sources/boundaries.py`
- Modify: `.gitignore`

- [x] **Step 1: Write the failing test**

Create `tests/test_io.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from migracoast_ml.sources._io import atomic_write_bytes, atomic_write_csv


def test_writes_bytes_to_the_destination(tmp_path):
    target = tmp_path / "archive.zip"
    atomic_write_bytes(target, b"payload")
    assert target.read_bytes() == b"payload"


def test_leaves_no_partial_file_behind(tmp_path):
    target = tmp_path / "archive.zip"
    atomic_write_bytes(target, b"payload")
    assert list(tmp_path.iterdir()) == [target]


def test_overwrites_an_existing_file(tmp_path):
    target = tmp_path / "archive.zip"
    target.write_bytes(b"old")
    atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"


def test_a_failed_write_never_corrupts_the_destination(tmp_path, monkeypatch):
    """The property that makes skip-if-exists safe."""
    target = tmp_path / "archive.zip"
    target.write_bytes(b"complete old file")

    def interrupted(self, other):
        raise KeyboardInterrupt("killed mid-rename")

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"complete old file"


def test_writes_a_frame_as_csv_without_index(tmp_path):
    target = tmp_path / "claims.csv"
    atomic_write_csv(pd.DataFrame({"a": [1, 2]}), target)
    assert target.read_text().splitlines() == ["a", "1", "2"]
    assert list(tmp_path.iterdir()) == [target]
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_io.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'migracoast_ml.sources._io'`

- [x] **Step 3: Write the implementation**

Create `migracoast_ml/sources/_io.py`:

```python
"""Crash-safe file writes for the data fetchers.

Fetchers skip a download when the destination already exists. Writing
straight to that path would let a killed process leave a truncated file
that every later run silently reuses. Writing to a sibling .part file and
renaming it into place means the destination only ever holds a complete
file. Path.replace is used rather than rename because it overwrites an
existing target on Windows too.
"""

from pathlib import Path


def _part_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part")


def atomic_write_bytes(destination: Path, data: bytes) -> Path:
    part = _part_path(destination)
    part.write_bytes(data)
    part.replace(destination)
    return destination


def atomic_write_csv(frame, destination: Path) -> Path:
    part = _part_path(destination)
    frame.to_csv(part, index=False)
    part.replace(destination)
    return destination
```

- [x] **Step 4: Retrofit `boundaries.py`**

In `migracoast_ml/sources/boundaries.py`, add the import:

```python
from migracoast_ml.sources._io import atomic_write_bytes
```

and in `_download` replace:

```python
    destination.write_bytes(response.content)
    return destination
```

with:

```python
    return atomic_write_bytes(destination, response.content)
```

Nothing else in that file changes.

- [x] **Step 5: Ignore leftover part files**

Append one line to `.gitignore`:

```
*.part
```

- [x] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: PASS, 22 passed (the existing 17 plus 5 new).

- [x] **Step 7: Commit**

```bash
git add migracoast_ml/sources/_io.py migracoast_ml/sources/boundaries.py tests/test_io.py .gitignore
git commit -m "fix: make fetcher downloads crash-safe with atomic writes"
```

---

## Task 4: Sea level projections source

Revised after real-data review: see commit message.

The NOAA CSV has 17 lines of citation text before the real header, values in
centimetres, and 15 scenario rows per site. All three are handled here.

**Files:**
- Create: `migracoast_ml/sources/sealevel.py`
- Create: `tests/fixtures/slr_sample.csv`
- Create: `tests/test_sealevel_source.py`

- [ ] **Step 1: Create the test fixture**

Create `tests/fixtures/slr_sample.csv`. The first 17 lines are filler standing in
for NOAA's citation block, line 18 is the real header, and the rest are data rows.
The column order matches the live file.

```
SLR_TF U.S. Sea Level Projections
filler line 2
filler line 3
filler line 4
filler line 5
filler line 6
filler line 7
filler line 8
filler line 9
filler line 10
filler line 11
filler line 12
filler line 13
filler line 14
filler line 15
filler line 16
filler line 17
PSMSL Site,PSMSL ID,NOAA ID,NOAA Name,RSL GridNum,Lat,Long,Regional Classification,US Coastline Intersect (1 = yes),Scenario,RSL contribution from VLM (trend: cm/year),Offset 1992 to 2000 (cm),Offset 2000 to 2005 (cm),RSL2005 (cm),RSL2020 (cm),RSL2030 (cm),RSL2040 (cm),RSL2050 (cm),RSL2060 (cm),RSL2070 (cm),RSL2080 (cm),RSL2090 (cm),RSL2100 (cm),RSL2110 (cm),RSL2120 (cm),RSL2130 (cm),RSL2140 (cm),RSL2150 (cm)
GMSL,0,0,GMSL,,,,,1,1.0 - MED,0,2,2,0,6,10,14,19,24,30,36,42,48,55,61,67,74,80
Boston,235,8443970,"Boston, MA",,42.4,-71.1,Northeast,1,1.0 - MED,0.1,2,3,0,10,16,22,30,38,47,56,66,76,86,96,106,116,126
Boston,235,8443970,"Boston, MA",,42.4,-71.1,Northeast,1,1.0 - LOW,0.1,2,3,0,8,13,18,25,32,40,48,56,64,72,80,88,96,104
Boston,235,8443970,"Boston, MA",,42.4,-71.1,Northeast,1,0.5 - MED,0.1,2,3,0,7,11,15,20,26,32,38,44,50,56,62,68,74,80
Miami,520,8723170,"Miami, FL",,25.8,-80.2,Southeast,1,1.0 - MED,0.2,2,4,0,12,19,26,35,45,56,67,79,91,103,115,127,139,151
grid_40.0_281.0,1000000000,0,grid_40.0_281.0,,40.0,-79.0,Northeast,1,1.0 - MED,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_sealevel_source.py`:

```python
from pathlib import Path

import pytest

from migracoast_ml.sources import sealevel

FIXTURE = Path(__file__).parent / "fixtures" / "slr_sample.csv"


@pytest.fixture
def projections():
    return sealevel.parse_projections(FIXTURE, scenario="1.0 - MED")


def test_skips_the_seventeen_junk_header_lines(projections):
    assert list(projections.columns) == ["lat", "lon", "slr_2050_ft", "slr_2100_ft"]


def test_keeps_only_the_requested_scenario(projections):
    # Boston has three scenario rows in the fixture; only one survives.
    boston = projections[(projections.lat == 42.4) & (projections.lon == -71.1)]
    assert len(boston) == 1


def test_drops_rows_without_coordinates(projections):
    # The GMSL row has empty Lat and Long and must not appear.
    assert len(projections) == 2


def test_converts_centimetres_to_feet_and_adds_the_offset(projections):
    boston = projections[projections.lat == 42.4].iloc[0]
    # RSL2050 is 30 cm, the 2000-to-2005 offset is 3 cm, so 33 cm.
    assert boston.slr_2050_ft == pytest.approx(33 * 0.0328084, rel=1e-6)
    # RSL2100 is 76 cm plus the same 3 cm offset.
    assert boston.slr_2100_ft == pytest.approx(79 * 0.0328084, rel=1e-6)


def test_unknown_scenario_raises(projections):
    with pytest.raises(ValueError, match="scenario"):
        sealevel.parse_projections(FIXTURE, scenario="9.9 - MED")


def test_drops_sites_with_no_projection(projections):
    # A site whose every RSL year is zero has no projection. Adding the
    # baseline offset to it would fabricate a small positive rise.
    placeholder = projections[(projections.lat == 40.0) & (projections.lon == -79.0)]
    assert placeholder.empty
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sealevel_source.py -v`
Expected: FAIL with `ImportError: cannot import name 'sealevel'`

- [ ] **Step 4: Write the implementation**

Create `migracoast_ml/sources/sealevel.py`:

```python
"""NOAA 2022 Sea Level Rise Technical Report projections."""

import zipfile
from pathlib import Path

import pandas as pd
import requests

from migracoast_ml.config import RAW_DIR, SLR_SCENARIO, ensure_dirs
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
    response = requests.get(SLR_URL, timeout=180, allow_redirects=True)
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sealevel_source.py -v`
Expected: PASS, 6 passed

- [ ] **Step 6: Commit**

```bash
git add migracoast_ml/sources/sealevel.py tests/fixtures/slr_sample.csv tests/test_sealevel_source.py
git commit -m "feat: add NOAA sea level projection parser"
```

---

## Task 5: Coastline distance index

**Files:**
- Create: `migracoast_ml/features/coast.py`
- Create: `tests/test_coast.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_coast.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_coast.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.features.coast'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/features/coast.py`:

```python
"""Distance from a point to the nearest coastline."""

import math

import numpy as np
from shapely import STRtree
from shapely.geometry import Point
from shapely.ops import nearest_points

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


class CoastlineIndex:
    """Spatial index over coastline geometries for nearest-distance queries."""

    def __init__(self, coast_gdf):
        self._geoms = list(coast_gdf.geometry)
        if not self._geoms:
            raise ValueError("Coastline index requires at least one geometry")
        self._tree = STRtree(self._geoms)

    def distance_km(self, lat: float, lon: float) -> float:
        """Kilometres from this point to the closest point on any coastline."""
        point = Point(lon, lat)
        nearest_index = int(self._tree.nearest(point))
        _, closest = nearest_points(point, self._geoms[nearest_index])
        return haversine_km(lat, lon, closest.y, closest.x)

    def distances_km(self, lats, lons) -> np.ndarray:
        """Vectorised convenience wrapper over `distance_km`."""
        return np.array([self.distance_km(a, o) for a, o in zip(lats, lons)])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_coast.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/features/coast.py tests/test_coast.py
git commit -m "feat: add coastline distance index"
```

---

## Task 6: Sea level feature index

Revised after real-data review: see commit message.

**Files:**
- Create: `migracoast_ml/features/sealevel.py`
- Create: `tests/test_sealevel_features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sealevel_features.py`:

```python
import pandas as pd
import pytest

from migracoast_ml.features.sealevel import SeaLevelIndex


@pytest.fixture
def index():
    return SeaLevelIndex(
        pd.DataFrame(
            {
                "lat": [42.4, 25.8],
                "lon": [-71.1, -80.2],
                "slr_2050_ft": [1.08, 1.15],
                "slr_2100_ft": [2.59, 2.99],
            }
        )
    )


def test_returns_both_horizon_features(index):
    feats = index.features(42.4, -71.1)
    assert set(feats) == {"slr_2050_ft", "slr_2100_ft"}


def test_picks_the_nearest_projection_site(index):
    near_boston = index.features(42.0, -71.0)
    assert near_boston["slr_2050_ft"] == pytest.approx(1.08)

    near_miami = index.features(26.0, -80.0)
    assert near_miami["slr_2050_ft"] == pytest.approx(1.15)


def test_a_distant_point_still_returns_a_value(index):
    feats = index.features(0.0, 0.0)
    assert feats["slr_2050_ft"] == pytest.approx(1.08)


def test_empty_table_raises():
    with pytest.raises(ValueError, match="empty"):
        SeaLevelIndex(pd.DataFrame(columns=["lat", "lon", "slr_2050_ft", "slr_2100_ft"]))


def test_picks_the_true_great_circle_nearest_site():
    # Query at (43.0, -70.0). Site A sits due north at (52.8, -70.0),
    # site B sits due west at (43.0, -83.4). The equirectangular
    # approximation (which scales longitude by cos(query latitude)) puts
    # site A closer, but true haversine distance is A=1089.7118 km vs
    # B=1088.5691 km, so B is actually nearer by about 1.14 km.
    site_a = SeaLevelIndex(
        pd.DataFrame(
            {
                "lat": [52.8, 43.0],
                "lon": [-70.0, -83.4],
                "slr_2050_ft": [1.08, 1.15],
                "slr_2100_ft": [2.59, 2.99],
            }
        )
    )
    feats = site_a.features(43.0, -70.0)
    assert feats["slr_2050_ft"] == pytest.approx(1.15)
    assert feats["slr_2100_ft"] == pytest.approx(2.99)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sealevel_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.features.sealevel'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/features/sealevel.py`:

```python
"""Nearest-site lookup of projected sea level rise."""

import numpy as np
import pandas as pd


class SeaLevelIndex:
    """Nearest-neighbour lookup over NOAA sea level projection sites."""

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Sea level projection table is empty")
        self._lat = df["lat"].to_numpy(dtype=float)
        self._lon = df["lon"].to_numpy(dtype=float)
        self._slr_2050 = df["slr_2050_ft"].to_numpy(dtype=float)
        self._slr_2100 = df["slr_2100_ft"].to_numpy(dtype=float)

    def _nearest(self, lat: float, lon: float) -> int:
        # True great-circle distance. An equirectangular shortcut picked the
        # wrong site for about 2% of real grid cells.
        phi1 = np.radians(lat)
        phi2 = np.radians(self._lat)
        dphi = phi2 - phi1
        dlambda = np.radians(self._lon - lon)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return int(np.argmin(a))

    def features(self, lat: float, lon: float) -> dict:
        i = self._nearest(lat, lon)
        return {
            "slr_2050_ft": float(self._slr_2050[i]),
            "slr_2100_ft": float(self._slr_2100[i]),
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sealevel_features.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/features/sealevel.py tests/test_sealevel_features.py
git commit -m "feat: add sea level feature index"
```

---

## Task 7: Elevation source

**Files:**
- Create: `migracoast_ml/sources/elevation.py`
- Create: `tests/test_elevation_source.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_elevation_source.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_elevation_source.py -v`
Expected: FAIL with `ImportError: cannot import name 'elevation'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/sources/elevation.py`:

```python
"""ETOPO 2022 elevation, fetched as per-state subsets.

The global file is 1.6 GB. NOAA's THREDDS NetCDF Subset Service returns a
clipped box instead, a few megabytes per state.
"""

from pathlib import Path
from urllib.parse import urlencode

import requests

from migracoast_ml.config import RAW_DIR, ensure_dirs
from migracoast_ml.sources._io import atomic_write_bytes

# Only the 30 arc-second dataset works; the 60 arc-second one returns HTTP 500.
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/ncss/grid/global/ETOPO2022/30s/"
    "30s_surface_elev_netcdf/ETOPO_2022_v1_30s_N90W180_surface.nc"
)

# Padding so that cells on a state edge still get a complete window.
BOUNDS_PAD = 0.2


def pad_bounds(bounds, pad: float = BOUNDS_PAD) -> tuple[float, float, float, float]:
    """Expand (min_lon, min_lat, max_lon, max_lat) into (south, north, west, east)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    return (
        round(min_lat - pad, 6),
        round(max_lat + pad, 6),
        round(min_lon - pad, 6),
        round(max_lon + pad, 6),
    )


def build_url(south: float, north: float, west: float, east: float) -> str:
    """Compose a NetCDF Subset Service request for one bounding box."""
    query = urlencode(
        {
            "var": "z",
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            # accept=netcdf4 is rejected with HTTP 400 by this service.
            "accept": "netcdf",
        }
    )
    return f"{ETOPO_URL}?{query}"


def fetch_elevation(states_gdf, force: bool = False) -> list[Path]:
    """Download one elevation subset per state. Returns the written paths."""
    ensure_dirs()
    out_dir = RAW_DIR / "elevation"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for _, row in states_gdf.iterrows():
        destination = out_dir / f"etopo_{row.postal}.nc"
        if destination.exists() and not force:
            written.append(destination)
            continue
        south, north, west, east = pad_bounds(row.geometry.bounds)
        response = requests.get(build_url(south, north, west, east), timeout=600)
        response.raise_for_status()
        atomic_write_bytes(destination, response.content)
        written.append(destination)
    return written
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_elevation_source.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/sources/elevation.py tests/test_elevation_source.py
git commit -m "feat: add ETOPO elevation subset fetcher"
```

---

## Task 8: Terrain features

Revised after real-data review: see commit message.

Elevation arrays are masked, and ocean pixels are negative. Land is `z > 0`.
Elevation statistics are computed over land pixels only; `land_fraction` is
computed over all valid pixels.

**Files:**
- Create: `migracoast_ml/features/terrain.py`
- Create: `tests/test_terrain.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_terrain.py`:

```python
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


def test_overlapping_boxes_do_not_double_count_pixels():
    """Padded state boxes overlap. Pixels shared by two boxes must count once."""
    step = 1 / 120
    lats = np.arange(41.0, 42.0, step) + step / 2
    lons = np.arange(-72.0, -70.0, step) + step / 2
    # Elevation rises with longitude, so over-weighting any strip biases the mean.
    z_full = np.tile(np.linspace(1.0, 50.0, lons.size), (lats.size, 1))

    def cut(lon_lo, lon_hi):
        cols = np.where((lons >= lon_lo) & (lons <= lon_hi))[0]
        return (lats.copy(), lons[cols].copy(), np.ma.asarray(z_full[:, cols]))

    box_a = cut(-72.0, -70.95)   # eastern edge at about -70.95
    box_b = cut(-71.3, -70.0)    # fully covers the query window

    merged = ElevationGrid([cut(-72.0, -70.0)])
    overlapping = ElevationGrid([box_a, box_b])

    query = (41.5, -71.0)        # window spans -71.05 to -70.95, straddling box A's edge
    assert overlapping.window(*query, half=0.05).size == merged.window(*query, half=0.05).size
    assert overlapping.features(*query, half=0.05) == pytest.approx(
        merged.features(*query, half=0.05), nan_ok=True
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_terrain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.features.terrain'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/features/terrain.py`:

```python
"""Terrain statistics from ETOPO elevation subsets."""

from pathlib import Path

import numpy as np

from migracoast_ml.config import CELL_HALF

NAN = float("nan")


def terrain_features(window) -> dict:
    """Summarise an elevation window. Land is any pixel above zero metres."""
    values = np.ma.masked_invalid(np.ma.asarray(window)).compressed()
    values = values[values > -30000]  # drop ETOPO fill values

    if values.size == 0:
        return {
            "elev_mean": NAN,
            "elev_min": NAN,
            "elev_p10": NAN,
            "frac_below_2m": NAN,
            "frac_below_5m": NAN,
            "land_fraction": NAN,
        }

    land = values[values > 0]
    land_fraction = float(land.size) / float(values.size)

    if land.size == 0:
        return {
            "elev_mean": NAN,
            "elev_min": NAN,
            "elev_p10": NAN,
            "frac_below_2m": NAN,
            "frac_below_5m": NAN,
            "land_fraction": land_fraction,
        }

    return {
        "elev_mean": float(np.mean(land)),
        "elev_min": float(np.min(land)),
        "elev_p10": float(np.percentile(land, 10)),
        "frac_below_2m": float(np.mean(land < 2.0)),
        "frac_below_5m": float(np.mean(land < 5.0)),
        "land_fraction": land_fraction,
    }


class ElevationGrid:
    """One or more ETOPO boxes, queried by geographic window."""

    def __init__(self, boxes):
        # Each box is (lat_array, lon_array, z_array).
        self._boxes = boxes

    @classmethod
    def from_dir(cls, directory) -> "ElevationGrid":
        import netCDF4

        boxes = []
        for path in sorted(Path(directory).glob("*.nc")):
            with netCDF4.Dataset(path) as ds:
                boxes.append(
                    (
                        np.array(ds.variables["lat"][:], dtype=float),
                        np.array(ds.variables["lon"][:], dtype=float),
                        np.ma.asarray(ds.variables["z"][:]),
                    )
                )
        if not boxes:
            raise ValueError(f"No elevation NetCDF files found in {directory}")
        return cls(boxes)

    def window(self, lat: float, lon: float, half: float = CELL_HALF) -> np.ndarray:
        """Return each distinct pixel within `half` degrees of the point.

        Padded state boxes overlap and all sample one global grid, so a pixel
        can appear in several boxes with identical coordinates. Pixels are
        deduplicated by coordinate so that overlap never over-weights part of
        a window. Masked pixels are returned as NaN.
        """
        keys = []
        values = []
        for lats, lons, z in self._boxes:
            rows = np.where((lats >= lat - half) & (lats <= lat + half))[0]
            cols = np.where((lons >= lon - half) & (lons <= lon + half))[0]
            if not (rows.size and cols.size):
                continue
            lat_grid, lon_grid = np.meshgrid(lats[rows], lons[cols], indexing="ij")
            keys.append(
                np.column_stack([np.round(lat_grid.ravel(), 6), np.round(lon_grid.ravel(), 6)])
            )
            block = np.ma.asarray(z[np.ix_(rows, cols)]).astype(float)
            values.append(np.ma.filled(block, np.nan).ravel())
        if not keys:
            return np.array([])
        keys = np.vstack(keys)
        values = np.concatenate(values)
        _, first = np.unique(keys, axis=0, return_index=True)
        return values[np.sort(first)]

    def features(self, lat: float, lon: float, half: float = CELL_HALF) -> dict:
        return terrain_features(self.window(lat, lon, half))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_terrain.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/features/terrain.py tests/test_terrain.py
git commit -m "feat: add terrain feature extraction"
```

---

## Task 9: FEMA claims source

Revised after live API review: see commit message.

**Files:**
- Create: `migracoast_ml/sources/fema.py`
- Create: `tests/test_fema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fema.py`:

```python
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

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


def test_page_url_pins_a_stable_sort_order():
    """Without an explicit order, paging with $skip can silently duplicate or
    drop rows, and nothing downstream could detect it."""
    query = parse_qs(urlparse(fema.build_page_url("MA", skip=0)).query)
    assert query["$orderby"] == ["id"]


def test_empty_result_raises_a_clear_error(monkeypatch):
    class EmptyResponse:
        text = "latitude,longitude\n"

        def raise_for_status(self):
            return None

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, timeout=None):
            return EmptyResponse()

    monkeypatch.setattr(fema.requests, "Session", Session)
    monkeypatch.setattr(fema, "RAW_DIR", tmp := fema.RAW_DIR)
    with pytest.raises(ValueError, match="No claims returned"):
        fema.fetch_claims(states=("MA",), force=True)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fema.py -v`
Expected: FAIL with `ImportError: cannot import name 'fema'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/sources/fema.py`:

```python
"""FEMA NFIP redacted claims, the supervised target for the model."""

import io
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from migracoast_ml.config import RAW_DIR, STATES, ensure_dirs
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
                response = session.get(build_page_url(state, skip), timeout=300)
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fema.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/sources/fema.py tests/test_fema.py
git commit -m "feat: add FEMA NFIP claims fetcher"
```

---

## Task 10: Storm events source

NOAA appends a creation date to each filename and changes it when republishing,
so filenames must be discovered by pattern, never hardcoded.

**Files:**
- Create: `migracoast_ml/sources/storms.py`
- Create: `tests/fixtures/storm_index.html`
- Create: `tests/test_storms.py`

- [ ] **Step 1: Create the test fixture**

Create `tests/fixtures/storm_index.html`, a trimmed copy of the NOAA directory
listing with deliberately differing creation dates:

```html
<html><head><title>Index of /pub/data/swdi/stormevents/csvfiles</title></head>
<body>
<a href="StormEvents_details-ftp_v1.0_d2000_c20260323.csv.gz">StormEvents_details-ftp_v1.0_d2000_c20260323.csv.gz</a>
<a href="StormEvents_details-ftp_v1.0_d2001_c20260323.csv.gz">StormEvents_details-ftp_v1.0_d2001_c20260323.csv.gz</a>
<a href="StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz">StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz</a>
<a href="StormEvents_fatalities-ftp_v1.0_d2024_c20260728.csv.gz">StormEvents_fatalities-ftp_v1.0_d2024_c20260728.csv.gz</a>
<a href="StormEvents_locations-ftp_v1.0_d2024_c20260728.csv.gz">StormEvents_locations-ftp_v1.0_d2024_c20260728.csv.gz</a>
</body></html>
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_storms.py`:

```python
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_storms.py -v`
Expected: FAIL with `ImportError: cannot import name 'storms'`

- [ ] **Step 4: Write the implementation**

Create `migracoast_ml/sources/storms.py`:

```python
"""NOAA Storm Events, the source of hazard history features."""

import gzip
import io
import re
from pathlib import Path

import pandas as pd
import requests

from migracoast_ml.config import (
    FLOOD_EVENT_TYPES,
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

    index = parse_index(requests.get(STORM_INDEX_URL, timeout=180).text)

    frames = []
    with requests.Session() as session:
        for year in years:
            filename = index.get(year)
            if filename is None:
                continue
            response = session.get(STORM_INDEX_URL + filename, timeout=300)
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_storms.py -v`
Expected: PASS, 5 passed

- [ ] **Step 6: Commit**

```bash
git add migracoast_ml/sources/storms.py tests/fixtures/storm_index.html tests/test_storms.py
git commit -m "feat: add NOAA storm events fetcher"
```

---

## Task 11: Hazard features

Because a feature window is exactly one cell, events are binned to cells once
and looked up in constant time. A per-cell scan would be far too slow.

**Files:**
- Create: `migracoast_ml/features/hazards.py`
- Create: `tests/test_hazards.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hazards.py`:

```python
import pandas as pd
import pytest

from migracoast_ml.features.hazards import HazardIndex


@pytest.fixture
def index():
    events = pd.DataFrame(
        {
            "lat": [42.36, 42.38, 42.41, 25.77],
            "lon": [-71.06, -71.09, -71.12, -80.19],
            "kind": ["flood", "flood", "storm", "flood"],
        }
    )
    return HazardIndex(events, n_years=25)


def test_returns_both_rate_features(index):
    assert set(index.features(42.4, -71.1)) == {
        "flood_events_per_year",
        "storm_events_per_year",
    }


def test_counts_are_divided_by_the_year_span(index):
    feats = index.features(42.4, -71.1)
    # Two floods and one storm fall in the 42.4, -71.1 cell.
    assert feats["flood_events_per_year"] == pytest.approx(2 / 25)
    assert feats["storm_events_per_year"] == pytest.approx(1 / 25)


def test_a_cell_with_no_events_reports_zero(index):
    feats = index.features(44.0, -69.0)
    assert feats["flood_events_per_year"] == 0.0
    assert feats["storm_events_per_year"] == 0.0


def test_lookup_is_by_cell_not_by_radius(index):
    # Any point inside the cell yields the cell's counts.
    assert index.features(42.38, -71.07) == index.features(42.4, -71.1)


def test_zero_years_raises():
    with pytest.raises(ValueError, match="n_years"):
        HazardIndex(pd.DataFrame(columns=["lat", "lon", "kind"]), n_years=0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hazards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.features.hazards'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/features/hazards.py`:

```python
"""Historical flood and storm event rates per grid cell."""

import pandas as pd

from migracoast_ml.features.grid import snap_to_cell


class HazardIndex:
    """Event counts binned to 0.1 degree cells, exposed as annual rates."""

    def __init__(self, events: pd.DataFrame, n_years: int):
        if not n_years:
            raise ValueError("n_years must be a positive number of years")
        self._n_years = float(n_years)
        self._counts: dict[tuple[float, float], dict[str, int]] = {}

        if events.empty:
            return

        snapped = [snap_to_cell(a, o) for a, o in zip(events["lat"], events["lon"])]
        binned = pd.DataFrame(
            {
                "cell_lat": [s[0] for s in snapped],
                "cell_lon": [s[1] for s in snapped],
                "kind": list(events["kind"]),
            }
        )
        grouped = binned.groupby(["cell_lat", "cell_lon", "kind"]).size()
        for (cell_lat, cell_lon, kind), count in grouped.items():
            self._counts.setdefault((cell_lat, cell_lon), {})[kind] = int(count)

    def features(self, lat: float, lon: float) -> dict:
        """Annual flood and storm event rates for the cell containing this point."""
        cell = snap_to_cell(lat, lon)
        counts = self._counts.get(cell, {})
        return {
            "flood_events_per_year": counts.get("flood", 0) / self._n_years,
            "storm_events_per_year": counts.get("storm", 0) / self._n_years,
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_hazards.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/features/hazards.py tests/test_hazards.py
git commit -m "feat: add hazard history features"
```

---

## Task 12: Feature builder and the training/serving parity test

This is the task that makes training and serving provably identical. The builder
snaps its input to a cell centre before touching any provider, so a query
anywhere inside a cell produces that cell's exact training vector.

**Files:**
- Create: `migracoast_ml/features/builder.py`
- Create: `tests/test_builder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_builder.py`:

```python
import numpy as np
import pandas as pd
import pytest

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.features.builder import FeatureBuilder


class StubElevation:
    def features(self, lat, lon, half=0.05):
        return {
            "elev_mean": 4.0,
            "elev_min": 1.0,
            "elev_p10": 2.0,
            "frac_below_2m": 0.25,
            "frac_below_5m": 0.75,
            "land_fraction": 0.9,
        }


class StubCoast:
    def distance_km(self, lat, lon):
        return 12.5


class StubSeaLevel:
    def features(self, lat, lon):
        return {"slr_2050_ft": 1.08, "slr_2100_ft": 2.59}


class StubHazards:
    def features(self, lat, lon):
        return {"flood_events_per_year": 0.08, "storm_events_per_year": 0.16}


@pytest.fixture
def builder():
    return FeatureBuilder(StubElevation(), StubCoast(), StubSeaLevel(), StubHazards())


def test_build_returns_exactly_the_configured_features(builder):
    feats = builder.build(42.36, -71.06)
    assert set(feats) == set(FEATURE_NAMES)


def test_build_excludes_geography(builder):
    feats = builder.build(42.36, -71.06)
    for banned in ("lat", "lon", "state", "latitude", "longitude"):
        assert banned not in feats


def test_every_point_inside_a_cell_yields_an_identical_vector(builder):
    """The invariant that eliminates training and serving skew."""
    centre = builder.build(42.4, -71.1)
    for dlat in (-0.04, 0.0, 0.04):
        for dlon in (-0.04, 0.0, 0.04):
            assert builder.build(42.4 + dlat, -71.1 + dlon) == centre


def test_build_frame_preserves_row_order_and_columns(builder):
    frame = builder.build_frame([(42.4, -71.1), (25.8, -80.2)])
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert len(frame) == 2


def test_build_frame_on_empty_input_returns_empty_frame_with_columns(builder):
    frame = builder.build_frame([])
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert frame.empty


def test_values_flow_through_from_the_providers(builder):
    feats = builder.build(42.4, -71.1)
    assert feats["coast_km"] == pytest.approx(12.5)
    assert feats["slr_2100_ft"] == pytest.approx(2.59)
    assert feats["storm_events_per_year"] == pytest.approx(0.16)
    assert feats["elev_mean"] == pytest.approx(4.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.features.builder'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/features/builder.py`:

```python
"""The single feature code path shared by training and serving."""

import pandas as pd

from migracoast_ml.config import CELL_HALF, FEATURE_NAMES, RAW_DIR, SLR_SCENARIO
from migracoast_ml.features.grid import snap_to_cell


class FeatureBuilder:
    """Assembles the model's feature vector for any coordinate.

    Every request is snapped to its 0.1 degree cell centre first. Training
    rows and live queries therefore run identical arithmetic, which is what
    makes training and serving skew impossible by construction.
    """

    def __init__(self, elevation, coast, sealevel, hazards):
        self._elevation = elevation
        self._coast = coast
        self._sealevel = sealevel
        self._hazards = hazards

    @classmethod
    def load(cls, raw_dir=None) -> "FeatureBuilder":
        """Build from the files written by the fetch stage."""
        from migracoast_ml.features.coast import CoastlineIndex
        from migracoast_ml.features.hazards import HazardIndex
        from migracoast_ml.features.sealevel import SeaLevelIndex
        from migracoast_ml.features.terrain import ElevationGrid
        from migracoast_ml.sources import boundaries, sealevel as slr_source
        from migracoast_ml.sources.storms import classify_events
        from migracoast_ml.config import STORM_YEARS

        raw_dir = raw_dir or RAW_DIR

        elevation = ElevationGrid.from_dir(raw_dir / "elevation")
        coast = CoastlineIndex(boundaries.load_coastline(raw_dir / "ne_10m_coastline.zip"))
        projections = slr_source.parse_projections(
            raw_dir / "slr_projections.csv", scenario=SLR_SCENARIO
        )
        events = classify_events(pd.read_csv(raw_dir / "storm_events.csv", low_memory=False))
        hazards = HazardIndex(events, n_years=len(STORM_YEARS))
        return cls(elevation, coast, SeaLevelIndex(projections), hazards)

    def build(self, lat: float, lon: float) -> dict:
        """Return the feature vector for the cell containing this point."""
        cell_lat, cell_lon = snap_to_cell(lat, lon)

        features = {}
        features.update(self._elevation.features(cell_lat, cell_lon, CELL_HALF))
        features["coast_km"] = self._coast.distance_km(cell_lat, cell_lon)
        features.update(self._sealevel.features(cell_lat, cell_lon))
        features.update(self._hazards.features(cell_lat, cell_lon))

        return {name: features[name] for name in FEATURE_NAMES}

    def build_frame(self, points) -> pd.DataFrame:
        """Vectorised build over an iterable of (lat, lon) pairs."""
        rows = [self.build(lat, lon) for lat, lon in points]
        return pd.DataFrame(rows, columns=list(FEATURE_NAMES))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_builder.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/features/builder.py tests/test_builder.py
git commit -m "feat: add feature builder with training and serving parity"
```

---

## Task 13: Fetch orchestrator and CLI skeleton

**Files:**
- Create: `migracoast_ml/sources/fetch_all.py`
- Create: `migracoast_ml/__main__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
import pytest

from migracoast_ml.__main__ import build_parser


def test_parser_exposes_every_pipeline_stage():
    parser = build_parser()
    for command in ("fetch", "build", "train", "evaluate", "baseline", "serve"):
        assert parser.parse_args([command]).command == command


def test_fetch_accepts_a_force_flag():
    assert build_parser().parse_args(["fetch", "--force"]).force is True


def test_fetch_defaults_to_not_forcing():
    assert build_parser().parse_args(["fetch"]).force is False


def test_serve_accepts_host_and_port():
    args = build_parser().parse_args(["serve", "--port", "8080"])
    assert args.port == 8080


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nonsense"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.__main__'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/sources/fetch_all.py`:

```python
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
```

Create `migracoast_ml/__main__.py`:

```python
"""Command line entry point for the Migracoast ML pipeline."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migracoast_ml",
        description="Learned coastal flood risk pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download every external data source.")
    fetch.add_argument("--force", action="store_true", help="Re-download existing files.")

    sub.add_parser("build", help="Build the training table from raw data.")
    sub.add_parser("train", help="Fit the model ladder and save the winner.")
    sub.add_parser("evaluate", help="Print cross-validation metrics.")
    sub.add_parser("baseline", help="Compare the rule engine against the model.")

    serve = sub.add_parser("serve", help="Run the prediction API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5001)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "fetch":
        from migracoast_ml.sources.fetch_all import fetch_all

        for name, path in fetch_all(force=args.force).items():
            print(f"{name}: {path}")
        return 0

    if args.command == "build":
        from migracoast_ml.dataset import build_training_table

        table = build_training_table()
        print(f"Wrote {len(table)} cells")
        return 0

    if args.command == "train":
        from migracoast_ml.models.train import train_and_save

        result = train_and_save()
        print(f"Selected {result['winner']}")
        return 0

    if args.command == "evaluate":
        from migracoast_ml.models.train import print_metrics

        print_metrics()
        return 0

    if args.command == "baseline":
        from migracoast_ml.baseline import run_comparison

        print(run_comparison())
        return 0

    if args.command == "serve":
        from migracoast_ml.serve import run

        run(host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/sources/fetch_all.py migracoast_ml/__main__.py tests/test_cli.py
git commit -m "feat: add fetch orchestrator and CLI"
```

---

## Task 14: Download the real data

This task runs the pipeline against the live services. It downloads roughly
350 MB and takes several minutes, mostly for Florida's 448,618 claims.

- [ ] **Step 1: Run the fetch**

Run: `.venv/Scripts/python.exe -m migracoast_ml fetch`

Expected output: six lines naming the written paths, ending with a claims CSV
and a storm events CSV.

- [ ] **Step 2: Verify what landed**

```bash
ls -la data/raw/
ls -la data/raw/elevation/
```

Expected: `ne_10m_admin_1_states_provinces.zip`, `ne_10m_coastline.zip`,
`slr_projections.csv`, `nfip_claims.csv`, `storm_events.csv`, and five NetCDF
files named `etopo_CA.nc`, `etopo_FL.nc`, `etopo_MA.nc`, `etopo_NY.nc`, `etopo_PA.nc`.

- [ ] **Step 3: Check the claim counts against the verified figures**

```bash
.venv/Scripts/python.exe -c "import pandas as pd; d=pd.read_csv('data/raw/nfip_claims.csv', low_memory=False); print(d.state.value_counts()); print('total', len(d))"
```

Expected, within a small margin if FEMA has refreshed: MA about 35,375,
NY about 175,430, PA about 77,060, FL about 448,618, CA about 53,563.
If any state is drastically short, paging stopped early and `fetch_claims`
needs investigation before continuing.

- [ ] **Step 4: Confirm nothing was committed**

```bash
git status --short
```

Expected: no untracked files under `data/`, because `.gitignore` excludes them.

---

## Task 15: Dataset assembly

**Files:**
- Create: `migracoast_ml/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset.py`:

```python
import numpy as np
import pandas as pd
import pytest

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.dataset import assemble, coastal_mask, target_from_counts


def test_target_is_log1p_of_claim_count():
    counts = pd.Series([0, 1, 9])
    assert list(target_from_counts(counts)) == pytest.approx(
        [0.0, np.log(2.0), np.log(10.0)]
    )


def test_cells_with_no_claims_are_kept_as_zero_targets():
    cells = pd.DataFrame({"cell_lat": [42.4, 44.0], "cell_lon": [-71.1, -69.0], "state": ["MA", "MA"]})
    counts = pd.DataFrame({"cell_lat": [42.4], "cell_lon": [-71.1], "claim_count": [7]})
    features = pd.DataFrame(
        {name: [1.0, 2.0] for name in FEATURE_NAMES}
    )
    out = assemble(cells, counts, features)
    assert len(out) == 2
    assert int(out.loc[out.cell_lat == 44.0, "claim_count"].iloc[0]) == 0


def test_assemble_adds_block_and_target_columns():
    cells = pd.DataFrame({"cell_lat": [42.4], "cell_lon": [-71.1], "state": ["MA"]})
    counts = pd.DataFrame({"cell_lat": [42.4], "cell_lon": [-71.1], "claim_count": [3]})
    features = pd.DataFrame({name: [1.0] for name in FEATURE_NAMES})
    out = assemble(cells, counts, features)
    assert "block" in out.columns
    assert "target" in out.columns
    assert out.target.iloc[0] == pytest.approx(np.log(4.0))


def test_assemble_preserves_every_feature_column():
    cells = pd.DataFrame({"cell_lat": [42.4], "cell_lon": [-71.1], "state": ["MA"]})
    counts = pd.DataFrame(columns=["cell_lat", "cell_lon", "claim_count"])
    features = pd.DataFrame({name: [1.0] for name in FEATURE_NAMES})
    out = assemble(cells, counts, features)
    for name in FEATURE_NAMES:
        assert name in out.columns


def test_coastal_mask_uses_the_configured_threshold():
    frame = pd.DataFrame({"coast_km": [5.0, 29.9, 30.1, 500.0]})
    assert list(coastal_mask(frame)) == [True, True, False, False]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.dataset'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/dataset.py`:

```python
"""Assemble the training table: one row per grid cell."""

import numpy as np
import pandas as pd

from migracoast_ml.config import (
    COASTAL_KM,
    FEATURE_NAMES,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_dirs,
)
from migracoast_ml.features.grid import block_id

TABLE_PATH = PROCESSED_DIR / "training_table.parquet"


def target_from_counts(counts) -> np.ndarray:
    """Model the log of one plus the claim count, which tames the long tail."""
    return np.log1p(np.asarray(counts, dtype=float))


def coastal_mask(frame: pd.DataFrame) -> pd.Series:
    """True for cells within the coastal threshold of the shoreline."""
    return frame["coast_km"] < COASTAL_KM


def assemble(cells: pd.DataFrame, counts: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join the cell universe, claim counts and features into one table."""
    out = cells.reset_index(drop=True).copy()
    for name in FEATURE_NAMES:
        out[name] = features[name].to_numpy()

    if len(counts):
        out = out.merge(counts, on=["cell_lat", "cell_lon"], how="left")
    else:
        out["claim_count"] = 0

    out["claim_count"] = out["claim_count"].fillna(0).astype(int)
    out["target"] = target_from_counts(out["claim_count"])
    out["block"] = [block_id(a, o) for a, o in zip(out["cell_lat"], out["cell_lon"])]
    return out


def build_training_table(raw_dir=None, save: bool = True) -> pd.DataFrame:
    """Run the full build: cell universe, features, claim counts, join."""
    from migracoast_ml.features.builder import FeatureBuilder
    from migracoast_ml.features.grid import build_cell_universe
    from migracoast_ml.sources import boundaries, fema

    raw_dir = raw_dir or RAW_DIR
    ensure_dirs()

    states = boundaries.load_states(raw_dir / "ne_10m_admin_1_states_provinces.zip")
    cells = build_cell_universe(states)

    builder = FeatureBuilder.load(raw_dir)
    features = builder.build_frame(zip(cells["cell_lat"], cells["cell_lon"]))

    claims = pd.read_csv(raw_dir / "nfip_claims.csv", low_memory=False)
    counts = fema.count_claims_per_cell(claims)

    table = assemble(cells, counts, features)
    if save:
        table.to_parquet(TABLE_PATH, index=False)
    return table


def load_training_table(path=None) -> pd.DataFrame:
    return pd.read_parquet(path or TABLE_PATH)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dataset.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Build the real training table**

Run: `.venv/Scripts/python.exe -m migracoast_ml build`

Expected: a line reading `Wrote N cells` where N is in the low thousands to
roughly fifteen thousand. This step takes a few minutes because it computes
coastline distance for every cell.

- [ ] **Step 6: Sanity check the table**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd
t = pd.read_parquet('data/processed/training_table.parquet')
print(t.shape)
print(t.state.value_counts())
print('cells with claims:', (t.claim_count > 0).sum())
print('coastal cells:', (t.coast_km < 30).sum())
print(t[['elev_mean','coast_km','slr_2100_ft','claim_count']].describe())
"
```

Expected: every state present, a meaningful number of cells with claims, and
no column that is entirely null. If `coast_km` is all zero or `elev_mean` is
all null, stop and fix the feature providers before training.

- [ ] **Step 7: Commit**

```bash
git add migracoast_ml/dataset.py tests/test_dataset.py
git commit -m "feat: assemble the cell-level training table"
```

---

## Task 16: Model ladder

**Files:**
- Create: `migracoast_ml/models/__init__.py`
- Create: `migracoast_ml/models/ladder.py`
- Create: `tests/test_ladder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ladder.py`:

```python
import numpy as np
from sklearn.pipeline import Pipeline

from migracoast_ml.models.ladder import build_models


def test_ladder_has_three_rungs():
    models = build_models()
    assert set(models) == {"ridge", "gradient_boosting", "neural_net"}


def test_every_rung_is_a_pipeline():
    for model in build_models().values():
        assert isinstance(model, Pipeline)


def test_every_rung_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 11))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=80)
    for name, model in build_models().items():
        model.fit(X, y)
        assert model.predict(X[:5]).shape == (5,), name


def test_pipelines_tolerate_missing_values():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 11))
    X[0, 0] = np.nan
    y = rng.normal(size=60)
    for name, model in build_models().items():
        model.fit(X, y)  # must not raise
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ladder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.models'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/models/__init__.py` as an empty file. Create
`migracoast_ml/models/ladder.py`:

```python
"""The three competing models, fit on identical features.

No PyTorch, LightGBM or XGBoost. On roughly ten thousand rows and eleven
features they add installation friction without adding accuracy, and being
able to explain that choice is worth more than the library name.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_models() -> dict[str, Pipeline]:
    """Return the model ladder, each rung a self-contained pipeline.

    Imputation and scaling live inside the pipeline so that no preprocessing
    statistic can leak across a cross-validation fold.
    """
    return {
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                # HistGradientBoosting handles NaN natively, but imputing keeps
                # every rung on identical inputs so the comparison is fair.
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.06,
                        max_depth=6,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "neural_net": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=800,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ladder.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/models/ tests/test_ladder.py
git commit -m "feat: add the three-model ladder"
```

---

## Task 17: Spatial cross-validation

A random split would put a cell's neighbour in the test fold and report an
inflated score. Grouping by 1 degree block prevents that.

**Files:**
- Create: `migracoast_ml/models/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
import numpy as np
import pandas as pd
import pytest

from migracoast_ml.models.evaluate import oof_predictions, spatial_cv


@pytest.fixture
def toy():
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 11)), columns=[f"f{i}" for i in range(11)])
    y = pd.Series(X["f0"] * 1.5 + rng.normal(scale=0.2, size=n))
    groups = pd.Series([f"b{i % 10}" for i in range(n)])
    coastal = pd.Series([i % 2 == 0 for i in range(n)])
    return X, y, groups, coastal


def test_returns_one_row_per_model(toy):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    models = {
        "a": Pipeline([("m", Ridge())]),
        "b": Pipeline([("m", Ridge(alpha=10.0))]),
    }
    table = spatial_cv(*toy, models=models, n_splits=3)
    assert len(table) == 2
    assert set(table["model"]) == {"a", "b"}


def test_reports_overall_and_coastal_metrics(toy):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    table = spatial_cv(*toy, models={"a": Pipeline([("m", Ridge())])}, n_splits=3)
    for column in (
        "rmse_mean", "rmse_std", "r2_mean", "spearman_mean",
        "coastal_rmse_mean", "coastal_rmse_std", "coastal_spearman_mean",
    ):
        assert column in table.columns


def test_no_block_appears_in_both_train_and_test(toy, monkeypatch):
    """The guarantee that makes the score honest, checked on our own code path."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    import migracoast_ml.models.evaluate as evaluate

    X, y, groups, _ = toy
    seen = []

    class SpyGroupKFold(evaluate.GroupKFold):
        def split(self, X, y=None, groups=None):
            for train_idx, test_idx in super().split(X, y, groups):
                seen.append((train_idx, test_idx))
                yield train_idx, test_idx

    monkeypatch.setattr(evaluate, "GroupKFold", SpyGroupKFold)
    evaluate.oof_predictions(X, y, groups, Pipeline([("m", Ridge())]), n_splits=3)

    assert len(seen) == 3
    for train_idx, test_idx in seen:
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])


def test_oof_predictions_cover_every_row_exactly_once(toy):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    X, y, groups, _ = toy
    preds = oof_predictions(X, y, groups, Pipeline([("m", Ridge())]), n_splits=3)
    assert preds.shape == (len(X),)
    assert not np.isnan(preds).any()


def test_a_useful_model_beats_a_constant_one(toy):
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    table = spatial_cv(
        *toy,
        models={
            "real": Pipeline([("m", Ridge())]),
            "constant": Pipeline([("m", DummyRegressor(strategy="mean"))]),
        },
        n_splits=3,
    )
    real = table[table.model == "real"].rmse_mean.iloc[0]
    constant = table[table.model == "constant"].rmse_mean.iloc[0]
    assert real < constant
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.models.evaluate'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/models/evaluate.py`:

```python
"""Spatially blocked cross-validation.

Neighbouring cells are strongly correlated. A random split would place a
cell's neighbour in the test fold and report a score that cannot be
reproduced on genuinely unseen ground, so folds are split by 1 degree block.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def _safe_spearman(truth, predicted) -> float:
    if len(truth) < 3 or np.std(predicted) == 0:
        return float("nan")
    return float(spearmanr(truth, predicted).statistic)


def spatial_cv(X, y, groups, coastal, models, n_splits: int = 5) -> pd.DataFrame:
    """Score every model under blocked cross-validation.

    Metrics are reported twice: over all cells, and over coastal cells only,
    because coastal locations are the product's actual use case.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    coastal = pd.Series(coastal).reset_index(drop=True)

    splitter = GroupKFold(n_splits=n_splits)
    rows = []

    for name, model in models.items():
        fold = {"rmse": [], "r2": [], "spearman": [],
                "coastal_rmse": [], "coastal_spearman": []}

        for train_idx, test_idx in splitter.split(X, y, groups):
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            predicted = fitted.predict(X.iloc[test_idx])
            truth = y.iloc[test_idx].to_numpy()

            fold["rmse"].append(float(np.sqrt(mean_squared_error(truth, predicted))))
            fold["r2"].append(float(r2_score(truth, predicted)))
            fold["spearman"].append(_safe_spearman(truth, predicted))

            mask = coastal.iloc[test_idx].to_numpy()
            if mask.sum() >= 3:
                fold["coastal_rmse"].append(
                    float(np.sqrt(mean_squared_error(truth[mask], predicted[mask])))
                )
                fold["coastal_spearman"].append(_safe_spearman(truth[mask], predicted[mask]))

        rows.append(
            {
                "model": name,
                "rmse_mean": float(np.nanmean(fold["rmse"])),
                "rmse_std": float(np.nanstd(fold["rmse"])),
                "r2_mean": float(np.nanmean(fold["r2"])),
                "r2_std": float(np.nanstd(fold["r2"])),
                "spearman_mean": float(np.nanmean(fold["spearman"])),
                "coastal_rmse_mean": float(np.nanmean(fold["coastal_rmse"])),
                "coastal_rmse_std": float(np.nanstd(fold["coastal_rmse"])),
                "coastal_spearman_mean": float(np.nanmean(fold["coastal_spearman"])),
            }
        )

    return pd.DataFrame(rows)


def oof_predictions(X, y, groups, model, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions, the only honest input to the baseline comparison."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    out = np.full(len(X), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        out[test_idx] = fitted.predict(X.iloc[test_idx])
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluate.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/models/evaluate.py tests/test_evaluate.py
git commit -m "feat: add spatially blocked cross-validation"
```

---

## Task 18: Risk scorer

**Files:**
- Create: `migracoast_ml/models/scoring.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring.py`:

```python
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.models.scoring import RiskScorer


class Passthrough:
    """Predicts the first feature, so expected scores are easy to reason about."""

    def predict(self, X):
        return np.asarray(X)[:, 0].astype(float)


@pytest.fixture
def scorer():
    return RiskScorer(
        model=Passthrough(),
        reference=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        feature_names=FEATURE_NAMES,
        metadata={"winner": "test"},
    )


def _features(first_value):
    feats = {name: 0.0 for name in FEATURE_NAMES}
    feats[FEATURE_NAMES[0]] = first_value
    return feats


def test_score_returns_the_three_public_numbers(scorer):
    out = scorer.score_features(_features(2.0))
    assert set(out) >= {"predicted_claim_intensity", "risk_percentile", "safety_score"}


def test_percentile_is_zero_below_the_reference_range(scorer):
    assert scorer.score_features(_features(-10.0))["risk_percentile"] == pytest.approx(0.0)


def test_percentile_is_one_hundred_above_the_reference_range(scorer):
    assert scorer.score_features(_features(99.0))["risk_percentile"] == pytest.approx(100.0)


def test_percentile_is_in_the_middle_for_a_median_value(scorer):
    assert scorer.score_features(_features(2.0))["risk_percentile"] == pytest.approx(50.0, abs=15)


def test_safety_is_the_complement_of_risk(scorer):
    out = scorer.score_features(_features(1.0))
    assert out["safety_score"] == pytest.approx(100.0 - out["risk_percentile"])


def test_feature_order_is_enforced_not_dict_order(scorer):
    shuffled = dict(reversed(list(_features(3.0).items())))
    assert scorer.score_features(shuffled) == scorer.score_features(_features(3.0))


def test_missing_feature_raises(scorer):
    feats = _features(1.0)
    del feats[FEATURE_NAMES[-1]]
    with pytest.raises(ValueError, match="missing"):
        scorer.score_features(feats)


def test_round_trips_through_disk(tmp_path):
    model = Pipeline([("m", DummyRegressor(strategy="constant", constant=1.0))])
    model.fit(np.zeros((4, len(FEATURE_NAMES))), np.zeros(4))
    original = RiskScorer(model, np.array([0.0, 1.0, 2.0]), FEATURE_NAMES, {"winner": "dummy"})

    path = tmp_path / "model.joblib"
    original.save(path)
    restored = RiskScorer.load(path)

    assert restored.metadata["winner"] == "dummy"
    assert restored.feature_names == tuple(FEATURE_NAMES)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.models.scoring'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/models/scoring.py`:

```python
"""Turn a raw model prediction into the scores the product exposes."""

import numpy as np
import joblib


class RiskScorer:
    """Wraps the fitted model and the reference distribution used for ranking."""

    def __init__(self, model, reference, feature_names, metadata=None):
        self.model = model
        self.reference = np.sort(np.asarray(reference, dtype=float))
        self.feature_names = tuple(feature_names)
        self.metadata = dict(metadata or {})

    def _vector(self, features: dict) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Feature vector missing: {missing}")
        # Order by the configured feature list, never by dict insertion order.
        return np.array([[float(features[name]) for name in self.feature_names]])

    def percentile(self, prediction: float) -> float:
        """Where this prediction falls among the reference cells, 0 to 100."""
        if self.reference.size == 0:
            return float("nan")
        position = float(np.searchsorted(self.reference, prediction, side="right"))
        return float(np.clip(100.0 * position / self.reference.size, 0.0, 100.0))

    def score_features(self, features: dict) -> dict:
        """Score one already-built feature vector."""
        prediction = float(self.model.predict(self._vector(features))[0])
        risk = self.percentile(prediction)
        return {
            "predicted_claim_intensity": prediction,
            "risk_percentile": round(risk, 1),
            # The site's existing convention is higher means safer.
            "safety_score": round(100.0 - risk, 1),
        }

    def save(self, path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "reference": self.reference,
                "feature_names": self.feature_names,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "RiskScorer":
        blob = joblib.load(path)
        return cls(
            blob["model"], blob["reference"], blob["feature_names"], blob["metadata"]
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scoring.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/models/scoring.py tests/test_scoring.py
git commit -m "feat: add percentile risk scorer"
```

---

## Task 19: Training and model selection

**Files:**
- Create: `migracoast_ml/models/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train.py`:

```python
import numpy as np
import pandas as pd
import pytest

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.models.train import select_winner, train_from_table


@pytest.fixture
def table():
    rng = np.random.default_rng(3)
    n = 300
    data = {name: rng.normal(size=n) for name in FEATURE_NAMES}
    frame = pd.DataFrame(data)
    frame["target"] = frame["elev_mean"] * -1.2 + rng.normal(scale=0.2, size=n)
    frame["block"] = [f"b{i % 12}" for i in range(n)]
    frame["coast_km"] = np.abs(frame["coast_km"]) * 10.0
    frame["cell_lat"] = 42.0 + rng.normal(scale=0.5, size=n)
    frame["cell_lon"] = -71.0 + rng.normal(scale=0.5, size=n)
    frame["claim_count"] = rng.integers(0, 5, size=n)
    frame["state"] = "MA"
    return frame


def test_winner_is_chosen_on_coastal_rmse():
    metrics = pd.DataFrame(
        {
            "model": ["a", "b"],
            "rmse_mean": [0.1, 0.9],          # 'a' wins overall
            "coastal_rmse_mean": [0.8, 0.2],  # 'b' wins where it matters
        }
    )
    assert select_winner(metrics) == "b"


def test_winner_ignores_nan_rows():
    metrics = pd.DataFrame(
        {"model": ["a", "b"], "rmse_mean": [0.5, 0.5], "coastal_rmse_mean": [np.nan, 0.4]}
    )
    assert select_winner(metrics) == "b"


def test_training_returns_a_scorer_and_metrics(table, tmp_path):
    result = train_from_table(table, out_dir=tmp_path)
    assert result["winner"] in {"ridge", "gradient_boosting", "neural_net"}
    assert len(result["metrics"]) == 3


def test_training_writes_the_expected_artifacts(table, tmp_path):
    train_from_table(table, out_dir=tmp_path)
    assert (tmp_path / "risk_model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()


def test_saved_scorer_predicts_on_a_feature_dict(table, tmp_path):
    from migracoast_ml.models.scoring import RiskScorer

    train_from_table(table, out_dir=tmp_path)
    scorer = RiskScorer.load(tmp_path / "risk_model.joblib")
    out = scorer.score_features({name: 0.0 for name in FEATURE_NAMES})
    assert 0.0 <= out["risk_percentile"] <= 100.0


def test_reference_distribution_comes_from_coastal_cells(table, tmp_path):
    from migracoast_ml.models.scoring import RiskScorer

    table = table.copy()
    # Exactly 100 of the 300 cells are coastal (under 30 km).
    table["coast_km"] = [5.0 if i % 3 == 0 else 100.0 for i in range(len(table))]
    train_from_table(table, out_dir=tmp_path)
    scorer = RiskScorer.load(tmp_path / "risk_model.joblib")
    assert scorer.reference.size == 100
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.models.train'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/models/train.py`:

```python
"""Fit the ladder, choose a winner, and serialise it for serving."""

import json
from datetime import date

import numpy as np
import pandas as pd

from migracoast_ml.config import (
    FEATURE_NAMES,
    MODELS_DIR,
    SLR_SCENARIO,
    STATES,
    ensure_dirs,
)
from migracoast_ml.dataset import coastal_mask, load_training_table
from migracoast_ml.models.evaluate import spatial_cv
from migracoast_ml.models.ladder import build_models
from migracoast_ml.models.scoring import RiskScorer

MODEL_PATH = MODELS_DIR / "risk_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


def select_winner(metrics: pd.DataFrame) -> str:
    """Lowest coastal RMSE wins, because coastal cells are the use case."""
    usable = metrics.dropna(subset=["coastal_rmse_mean"])
    if usable.empty:
        usable = metrics.dropna(subset=["rmse_mean"])
        return str(usable.loc[usable["rmse_mean"].idxmin(), "model"])
    return str(usable.loc[usable["coastal_rmse_mean"].idxmin(), "model"])


def train_from_table(table: pd.DataFrame, out_dir=None) -> dict:
    """Cross-validate every model, refit the winner, and write the artifacts."""
    out_dir = out_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    X = table[list(FEATURE_NAMES)]
    y = table["target"]
    groups = table["block"]
    coastal = coastal_mask(table)

    models = build_models()
    metrics = spatial_cv(X, y, groups, coastal, models)
    winner = select_winner(metrics)

    fitted = models[winner].fit(X, y)

    # Percentiles are ranked against coastal cells only; ranking against
    # inland zeros would push every coastal answer to the top of the scale.
    reference_rows = X[coastal.to_numpy()]
    reference = fitted.predict(reference_rows if len(reference_rows) else X)

    metadata = {
        "winner": winner,
        "trained_on": date.today().isoformat(),
        "n_cells": int(len(table)),
        "n_coastal_cells": int(coastal.sum()),
        "states": list(STATES),
        "scenario": SLR_SCENARIO,
        "feature_names": list(FEATURE_NAMES),
        "metrics": metrics.to_dict(orient="records"),
    }

    RiskScorer(fitted, reference, FEATURE_NAMES, metadata).save(out_dir / "risk_model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metadata, indent=2))

    return {"winner": winner, "metrics": metrics, "metadata": metadata}


def train_and_save() -> dict:
    ensure_dirs()
    return train_from_table(load_training_table())


def print_metrics() -> None:
    """Print the saved cross-validation table."""
    metadata = json.loads(METRICS_PATH.read_text())
    print(f"Winner: {metadata['winner']}  (trained {metadata['trained_on']})")
    print(f"Cells: {metadata['n_cells']}, coastal: {metadata['n_coastal_cells']}")
    print(pd.DataFrame(metadata["metrics"]).to_string(index=False))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_train.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Train on the real data**

Run: `.venv/Scripts/python.exe -m migracoast_ml train`
Expected: a line reading `Selected <model name>`.

Then: `.venv/Scripts/python.exe -m migracoast_ml evaluate`
Expected: the three-row metrics table with overall and coastal columns.

- [ ] **Step 6: Record the honest reading of the numbers**

Look at `coastal_r2_mean` and `coastal_spearman_mean`. If R squared is near
zero or negative, the model is not beating the mean and that must be reported
plainly in the README rather than hidden. Do not tune hyperparameters to chase
a better number at this point; the comparison in Task 23 is the real test.

- [ ] **Step 7: Commit**

```bash
git add migracoast_ml/models/train.py tests/test_train.py models/risk_model.joblib models/metrics.json
git commit -m "feat: train and select the risk model"
```

---

## Task 20: Explainability

**Files:**
- Create: `migracoast_ml/models/explain.py`
- Create: `tests/test_explain.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_explain.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.models.explain import permutation_importance_table, top_contributors


@pytest.fixture
def fitted():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(150, 11)), columns=list(FEATURE_NAMES))
    # Only elev_mean carries signal.
    y = pd.Series(X["elev_mean"] * 3.0 + rng.normal(scale=0.1, size=150))
    model = Pipeline([("m", Ridge())]).fit(X, y)
    return model, X, y


def test_importance_table_covers_every_feature(fitted):
    table = permutation_importance_table(*fitted)
    assert len(table) == len(FEATURE_NAMES)
    assert set(table["feature"]) == set(FEATURE_NAMES)


def test_importance_is_sorted_with_the_strongest_first(fitted):
    table = permutation_importance_table(*fitted)
    assert table["importance_mean"].is_monotonic_decreasing
    assert table["feature"].iloc[0] == "elev_mean"


def test_top_contributors_returns_the_requested_count(fitted):
    model, X, _ = fitted
    out = top_contributors(model, X.iloc[0].to_dict(), FEATURE_NAMES, k=3)
    assert len(out) == 3


def test_contributor_entries_carry_a_name_and_value(fitted):
    model, X, _ = fitted
    entry = top_contributors(model, X.iloc[0].to_dict(), FEATURE_NAMES, k=1)[0]
    assert "feature" in entry
    assert "value" in entry


def test_shap_absence_is_reported_not_raised(fitted, monkeypatch):
    import migracoast_ml.models.explain as explain

    monkeypatch.setattr(explain, "_import_shap", lambda: None)
    model, X, _ = fitted
    assert explain.shap_contributions(model, X.iloc[0].to_dict(), FEATURE_NAMES) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_explain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.models.explain'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/models/explain.py`:

```python
"""Why the model said what it said."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(model, X, y, n_repeats: int = 10, random_state: int = 42):
    """Global feature importance, strongest first."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="r2"
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def top_contributors(model, features: dict, feature_names, k: int = 5) -> list[dict]:
    """Per-prediction attribution by single-feature ablation.

    Each feature is replaced by the reference value of zero in turn; the shift
    in prediction is that feature's contribution. This always works, unlike
    SHAP which is an optional dependency.
    """
    names = list(feature_names)
    base_row = np.array([[float(features[name]) for name in names]])
    base_prediction = float(model.predict(base_row)[0])

    contributions = []
    for i, name in enumerate(names):
        ablated = base_row.copy()
        ablated[0, i] = 0.0
        shift = base_prediction - float(model.predict(ablated)[0])
        contributions.append(
            {"feature": name, "value": float(features[name]), "contribution": shift}
        )

    contributions.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return contributions[:k]


def _import_shap():
    """Isolated so tests can simulate SHAP being absent."""
    try:
        import shap

        return shap
    except ImportError:
        return None


def shap_contributions(model, features: dict, feature_names) -> dict | None:
    """SHAP values when the optional package is installed, otherwise None."""
    shap = _import_shap()
    if shap is None:
        return None

    names = list(feature_names)
    row = np.array([[float(features[name]) for name in names]])
    try:
        explainer = shap.Explainer(model.predict, row)
        values = explainer(row).values[0]
    except Exception:
        return None
    return {name: float(value) for name, value in zip(names, values)}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_explain.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/models/explain.py tests/test_explain.py
git commit -m "feat: add permutation importance and per-prediction attribution"
```

---

## Task 21: Flask API

**Files:**
- Create: `migracoast_ml/api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
import numpy as np
import pytest
from flask import Flask

from migracoast_ml.api import MAX_GRID_CELLS, create_blueprint
from migracoast_ml.config import FEATURE_NAMES


class StubBuilder:
    def build(self, lat, lon):
        return {name: 1.0 for name in FEATURE_NAMES}


class StubScorer:
    feature_names = tuple(FEATURE_NAMES)
    metadata = {"winner": "stub", "trained_on": "2026-09-15", "metrics": []}
    model = None
    reference = np.array([0.0, 1.0, 2.0])

    def score_features(self, features):
        return {
            "predicted_claim_intensity": 1.0,
            "risk_percentile": 50.0,
            "safety_score": 50.0,
        }


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(
        create_blueprint(StubScorer(), StubBuilder(), covered_states=("MA", "FL"))
    )
    return app.test_client()


def test_risk_returns_the_score_payload(client):
    response = client.post("/ml/risk", json={"latitude": 42.36, "longitude": -71.06})
    assert response.status_code == 200
    body = response.get_json()
    assert body["safety_score"] == 50.0
    assert body["risk_percentile"] == 50.0
    assert "features" in body
    assert "top_contributors" in body


def test_risk_rejects_missing_coordinates(client):
    response = client.post("/ml/risk", json={"latitude": 42.36})
    assert response.status_code == 400
    assert "longitude" in response.get_json()["error"].lower()


def test_risk_rejects_non_numeric_coordinates(client):
    response = client.post("/ml/risk", json={"latitude": "north", "longitude": -71.0})
    assert response.status_code == 400


def test_risk_rejects_out_of_range_coordinates(client):
    response = client.post("/ml/risk", json={"latitude": 999.0, "longitude": -71.0})
    assert response.status_code == 400


def test_heatmap_returns_one_entry_per_cell(client):
    response = client.post(
        "/ml/heatmap",
        json={"north": 42.4, "south": 42.2, "east": -71.0, "west": -71.2, "resolution": 0.1},
    )
    assert response.status_code == 200
    cells = response.get_json()["cells"]
    assert len(cells) >= 1
    assert {"lat", "lon", "risk_percentile", "safety_score"} <= set(cells[0])


def test_heatmap_rejects_an_oversized_grid(client):
    response = client.post(
        "/ml/heatmap",
        json={"north": 49.0, "south": 25.0, "east": -67.0, "west": -125.0, "resolution": 0.1},
    )
    assert response.status_code == 400
    assert str(MAX_GRID_CELLS) in response.get_json()["error"]


def test_heatmap_rejects_an_inverted_box(client):
    response = client.post(
        "/ml/heatmap",
        json={"north": 42.0, "south": 43.0, "east": -71.0, "west": -70.0, "resolution": 0.1},
    )
    assert response.status_code == 400


def test_model_info_reports_the_winner(client):
    body = client.get("/ml/model-info").get_json()
    assert body["winner"] == "stub"
    assert body["feature_names"] == list(FEATURE_NAMES)


def test_errors_are_json_not_html(client):
    response = client.post("/ml/risk", json={})
    assert response.status_code == 400
    assert response.is_json
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.api'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/api.py`:

```python
"""Flask blueprint exposing the learned risk model."""

from flask import Blueprint, jsonify, request

from migracoast_ml.config import CELL_SIZE, STATES
from migracoast_ml.features.grid import snap_to_cell

# A heatmap request larger than this is rejected rather than served slowly.
MAX_GRID_CELLS = 2500


def _read_number(payload, key):
    if key not in payload or payload[key] is None:
        raise ValueError(f"Missing required field: {key}")
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"Field {key} must be a number")


def _validate_point(lat, lon):
    if not -90.0 <= lat <= 90.0:
        raise ValueError("latitude must be between -90 and 90")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("longitude must be between -180 and 180")


def create_blueprint(scorer, builder, covered_states=STATES) -> Blueprint:
    """Build the /ml blueprint around an already-loaded scorer and builder."""
    blueprint = Blueprint("migracoast_ml", __name__, url_prefix="/ml")

    @blueprint.post("/risk")
    def risk():
        payload = request.get_json(silent=True) or {}
        try:
            lat = _read_number(payload, "latitude")
            lon = _read_number(payload, "longitude")
            _validate_point(lat, lon)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        features = builder.build(lat, lon)
        result = scorer.score_features(features)

        contributors = []
        if getattr(scorer, "model", None) is not None:
            from migracoast_ml.models.explain import top_contributors

            contributors = top_contributors(scorer.model, features, scorer.feature_names)

        cell_lat, cell_lon = snap_to_cell(lat, lon)
        return jsonify(
            {
                **result,
                "latitude": lat,
                "longitude": lon,
                "cell": {"lat": cell_lat, "lon": cell_lon},
                "covered_states": list(covered_states),
                "features": features,
                "top_contributors": contributors,
            }
        )

    @blueprint.post("/heatmap")
    def heatmap():
        payload = request.get_json(silent=True) or {}
        try:
            north = _read_number(payload, "north")
            south = _read_number(payload, "south")
            east = _read_number(payload, "east")
            west = _read_number(payload, "west")
            resolution = float(payload.get("resolution", CELL_SIZE))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if north <= south or east <= west:
            return jsonify({"error": "north must exceed south and east must exceed west"}), 400
        if resolution <= 0:
            return jsonify({"error": "resolution must be positive"}), 400

        rows = int((north - south) / resolution) + 1
        cols = int((east - west) / resolution) + 1
        if rows * cols > MAX_GRID_CELLS:
            return jsonify(
                {
                    "error": f"Grid of {rows * cols} cells exceeds the limit of "
                    f"{MAX_GRID_CELLS}. Use a coarser resolution or a smaller box."
                }
            ), 400

        cells = []
        for r in range(rows):
            lat = south + r * resolution
            for c in range(cols):
                lon = west + c * resolution
                scored = scorer.score_features(builder.build(lat, lon))
                cells.append(
                    {
                        "lat": round(lat, 4),
                        "lon": round(lon, 4),
                        "risk_percentile": scored["risk_percentile"],
                        "safety_score": scored["safety_score"],
                    }
                )

        return jsonify({"count": len(cells), "resolution": resolution, "cells": cells})

    @blueprint.get("/model-info")
    def model_info():
        metadata = dict(getattr(scorer, "metadata", {}))
        return jsonify(
            {
                "winner": metadata.get("winner"),
                "trained_on": metadata.get("trained_on"),
                "states": metadata.get("states", list(covered_states)),
                "scenario": metadata.get("scenario"),
                "feature_names": list(scorer.feature_names),
                "metrics": metadata.get("metrics", []),
                "n_cells": metadata.get("n_cells"),
            }
        )

    return blueprint
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add migracoast_ml/api.py tests/test_api.py
git commit -m "feat: add risk and heatmap API endpoints"
```

---

## Task 22: Standalone server, app.py integration, and the credential fix

**Files:**
- Create: `migracoast_ml/serve.py`
- Modify: `app.py:29-35` (database credentials)
- Modify: `app.py` (register the blueprint, at the end of the file)
- Create: `tests/test_serve.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_serve.py`:

```python
import re
from pathlib import Path

import pytest

from migracoast_ml import serve

APP_PY = Path(__file__).parent.parent / "app.py"


def test_create_app_registers_the_ml_blueprint(monkeypatch):
    class StubScorer:
        feature_names = ("a",)
        metadata = {}
        model = None

    monkeypatch.setattr(serve, "_load_scorer", lambda: StubScorer())
    monkeypatch.setattr(serve, "_load_builder", lambda: object())

    app = serve.create_app()
    rules = {str(rule) for rule in app.url_map.iter_rules()}
    assert "/ml/risk" in rules
    assert "/ml/heatmap" in rules
    assert "/ml/model-info" in rules


def test_health_endpoint_responds(monkeypatch):
    class StubScorer:
        feature_names = ("a",)
        metadata = {}
        model = None

    monkeypatch.setattr(serve, "_load_scorer", lambda: StubScorer())
    monkeypatch.setattr(serve, "_load_builder", lambda: object())

    client = serve.create_app().test_client()
    assert client.get("/health").status_code == 200


def test_app_py_has_no_hardcoded_password():
    """The committed credential is replaced by an environment lookup."""
    source = APP_PY.read_text(encoding="utf-8", errors="replace")
    assert "St5SfZXpxRPS4XZy" not in source
    assert "os.environ" in source


def test_app_py_reads_every_db_field_from_the_environment():
    source = APP_PY.read_text(encoding="utf-8", errors="replace")
    block = re.search(r"db_config\s*=\s*\{.*?\}", source, re.S)
    assert block is not None
    for field in ("MIGRACOAST_DB_HOST", "MIGRACOAST_DB_USER",
                  "MIGRACOAST_DB_PASSWORD", "MIGRACOAST_DB_NAME"):
        assert field in block.group(0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_serve.py -v`
Expected: FAIL with `ImportError: cannot import name 'serve'`

- [ ] **Step 3: Write the standalone server**

Create `migracoast_ml/serve.py`:

```python
"""Standalone Flask server for the ML API.

The rule engine in app.py cannot be imported without its raster and NetCDF
data files, so the ML API must be able to run entirely on its own.
"""

from flask import Flask, jsonify
from flask_cors import CORS

from migracoast_ml.api import create_blueprint
from migracoast_ml.models.train import MODEL_PATH


def _load_scorer():
    from migracoast_ml.models.scoring import RiskScorer

    return RiskScorer.load(MODEL_PATH)


def _load_builder():
    from migracoast_ml.features.builder import FeatureBuilder

    return FeatureBuilder.load()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Loaded once at startup and held in memory, so a point query is
    # a few milliseconds rather than a few seconds.
    app.register_blueprint(create_blueprint(_load_scorer(), _load_builder()))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


def run(host: str = "127.0.0.1", port: int = 5001) -> None:
    create_app().run(host=host, port=port, debug=False)
```

- [ ] **Step 4: Fix the committed credential in app.py**

Replace lines 29 to 35 of `app.py`. The current block is:

```python
# Database connection details
db_config = {
    'host': '127.0.0.1',
    'user': 'migracoast',
    'password': 'St5SfZXpxRPS4XZy',
    'database': 'migracoast'
}
```

Replace it with:

```python
# Database connection details, read from the environment.
# The previous hardcoded password is in this repository's git history
# and must be rotated on the database server.
import os

db_config = {
    'host': os.environ.get('MIGRACOAST_DB_HOST', '127.0.0.1'),
    'user': os.environ.get('MIGRACOAST_DB_USER', 'migracoast'),
    'password': os.environ.get('MIGRACOAST_DB_PASSWORD', ''),
    'database': os.environ.get('MIGRACOAST_DB_NAME', 'migracoast'),
}
```

- [ ] **Step 5: Register the blueprint in app.py**

Append this to the end of `app.py`, immediately before the
`if __name__ == '__main__':` block:

```python
# Optional: expose the learned risk model alongside the rule engine.
# The import is guarded because the ML package may not be installed, and
# because app.py itself cannot start without its raster data files.
try:
    from migracoast_ml.api import create_blueprint as _create_ml_blueprint
    from migracoast_ml.features.builder import FeatureBuilder as _FeatureBuilder
    from migracoast_ml.models.scoring import RiskScorer as _RiskScorer
    from migracoast_ml.models.train import MODEL_PATH as _MODEL_PATH

    app.register_blueprint(
        _create_ml_blueprint(_RiskScorer.load(_MODEL_PATH), _FeatureBuilder.load())
    )
    print("Registered the ML blueprint at /ml")
except Exception as _ml_error:  # pragma: no cover - depends on local data
    print(f"ML blueprint not registered: {_ml_error}")
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_serve.py -v`
Expected: PASS, 4 passed

- [ ] **Step 7: Start the server and exercise it by hand**

Run: `.venv/Scripts/python.exe -m migracoast_ml serve`

In a second terminal:

```bash
curl -s -X POST http://127.0.0.1:5001/ml/risk -H "Content-Type: application/json" -d "{\"latitude\": 25.77, \"longitude\": -80.19}"
curl -s http://127.0.0.1:5001/ml/model-info
```

Expected: the first returns a risk percentile, a safety score, eleven features
and a contributor list. The second returns the winning model name.
Sanity check: downtown Miami should not come back as one of the safest
locations. If it does, something is inverted and must be investigated before
the work is called complete.

- [ ] **Step 8: Commit**

```bash
git add migracoast_ml/serve.py tests/test_serve.py app.py
git commit -m "feat: add standalone ML server and move DB credentials to env vars"
```

---

## Task 23: Baseline comparison against the rule engine

`score_time_series.csv` holds 480 rule-engine scores with no coordinates. The
coordinates live in the notebook's point lists, and the row counts reconcile
exactly. The mapping below is an inference from those counts, so Step 5
validates it empirically before any comparison is published.

**Files:**
- Create: `migracoast_ml/baseline.py`
- Create: `tests/test_baseline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_baseline.py`:

```python
import numpy as np
import pandas as pd
import pytest

from migracoast_ml.baseline import (
    REGION_LISTS,
    alignment_confidence,
    extract_point_lists,
    reconstruct_coordinates,
)

NOTEBOOK = "Check_inundation.ipynb"


def test_region_lists_cover_every_region_in_the_csv():
    csv = pd.read_csv("score_time_series.csv")
    assert set(REGION_LISTS) == set(csv["State"])


def test_extracted_lists_have_the_documented_lengths():
    lists = extract_point_lists(NOTEBOOK)
    expected = {
        "MA_points": 28, "FL_points": 28, "CA_points": 28, "NY_points": 40,
        "PA_points": 24, "BOS_points": 28, "PA_points_new": 16,
        "MA_points_new": 12, "BOS_points_new": 12, "FL_points_new": 12,
        "CA_points_new": 12,
    }
    for name, count in expected.items():
        assert len(lists[name]) == count, name


def test_reconstructed_row_count_matches_the_csv():
    coords = reconstruct_coordinates(NOTEBOOK, "score_time_series.csv")
    assert len(coords) == 480


def test_every_region_gets_the_right_number_of_rows():
    coords = reconstruct_coordinates(NOTEBOOK, "score_time_series.csv")
    per_scenario = coords[coords["Scenario"] == "SSP585"]
    counts = per_scenario["State"].value_counts().to_dict()
    assert counts == {"BOS": 52, "NY": 40, "PA": 40, "FL": 40, "CA": 40, "MA": 28}


def test_reconstruction_raises_if_counts_disagree(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"2020": [1], "State": ["MA"], "Scenario": ["SSP585"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="count"):
        reconstruct_coordinates(NOTEBOOK, bad)


def test_alignment_confidence_is_positive_when_score_tracks_elevation():
    frame = pd.DataFrame(
        {
            "rule_score": list(range(10, 110, 10)),
            "elev_mean": [0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60],
        }
    )
    assert alignment_confidence(frame) > 0.5


def test_alignment_confidence_is_negative_when_the_mapping_is_shuffled():
    frame = pd.DataFrame(
        {
            "rule_score": list(range(100, 0, -10)),
            "elev_mean": [0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60],
        }
    )
    assert alignment_confidence(frame) < 0


def test_alignment_confidence_is_nan_below_ten_rows():
    frame = pd.DataFrame({"rule_score": [1, 2, 3], "elev_mean": [1, 2, 3]})
    assert np.isnan(alignment_confidence(frame))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migracoast_ml.baseline'`

- [ ] **Step 3: Write the implementation**

Create `migracoast_ml/baseline.py`:

```python
"""Compare the hand-tuned rule engine against the learned model.

The rule engine's 480 saved scores carry no coordinates. They are recovered
from the notebook's point lists, whose row counts reconcile exactly with the
CSV. That reconciliation is strong evidence but not proof, so
`alignment_confidence` checks the mapping against elevation before any
comparison is reported.
"""

import ast
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Concatenation order inferred from row counts. BOS absorbs MA_points_new
# because those twelve coordinates are downtown Boston.
REGION_LISTS = {
    "MA": ("MA_points",),
    "NY": ("NY_points",),
    "PA": ("PA_points", "PA_points_new"),
    "FL": ("FL_points", "FL_points_new"),
    "CA": ("CA_points", "CA_points_new"),
    "BOS": ("BOS_points", "BOS_points_new", "MA_points_new"),
}

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\[[^\]]*\])", re.M | re.S)


def extract_point_lists(notebook_path) -> dict[str, list[tuple[float, float]]]:
    """Pull every coordinate list out of the notebook source."""
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    found = {}
    for match in _ASSIGNMENT.finditer(source):
        name, body = match.group(1), match.group(2)
        if "points" not in name.lower():
            continue
        try:
            value = ast.literal_eval(body)
        except (ValueError, SyntaxError):
            continue
        pairs = [t for t in value if isinstance(t, tuple) and len(t) == 2]
        if pairs:
            found[name] = pairs
    return found


def reconstruct_coordinates(notebook_path, scores_path) -> pd.DataFrame:
    """Attach coordinates to every row of the saved rule-engine scores."""
    lists = extract_point_lists(notebook_path)
    scores = pd.read_csv(scores_path)

    frames = []
    for scenario in sorted(scores["Scenario"].unique()):
        for region, list_names in REGION_LISTS.items():
            block = scores[
                (scores["State"] == region) & (scores["Scenario"] == scenario)
            ].reset_index(drop=True)
            if block.empty:
                continue

            points = []
            for name in list_names:
                points.extend(lists.get(name, []))

            if len(points) != len(block):
                raise ValueError(
                    f"Coordinate count mismatch for {region}/{scenario}: "
                    f"{len(points)} points against {len(block)} rows"
                )

            block = block.copy()
            block["lat"] = [p[0] for p in points]
            block["lon"] = [p[1] for p in points]
            frames.append(block)

    return pd.concat(frames, ignore_index=True)


def alignment_confidence(frame: pd.DataFrame) -> float:
    """Spearman correlation between the rule score and elevation.

    The rule score is oriented so that higher means safer, so a correct
    coordinate alignment must produce a clearly positive correlation. A value
    near zero or negative means the mapping is wrong.
    """
    usable = frame.dropna(subset=["rule_score", "elev_mean"])
    if len(usable) < 10:
        return float("nan")
    return float(spearmanr(usable["rule_score"], usable["elev_mean"]).statistic)


def run_comparison(
    notebook_path="Check_inundation.ipynb",
    scores_path="score_time_series.csv",
    min_confidence: float = 0.2,
) -> str:
    """Produce the comparison, or explain why it cannot be trusted."""
    from migracoast_ml.dataset import load_training_table
    from migracoast_ml.features.grid import snap_to_cell
    from migracoast_ml.models.evaluate import oof_predictions
    from migracoast_ml.models.ladder import build_models
    from migracoast_ml.models.train import METRICS_PATH
    from migracoast_ml.config import FEATURE_NAMES

    coords = reconstruct_coordinates(notebook_path, scores_path)
    # 2020 is the first scored year and the one closest to the claim record.
    coords["rule_score"] = coords["2020"].astype(float)

    snapped = [snap_to_cell(a, o) for a, o in zip(coords["lat"], coords["lon"])]
    coords["cell_lat"] = [s[0] for s in snapped]
    coords["cell_lon"] = [s[1] for s in snapped]

    table = load_training_table()
    winner = json.loads(METRICS_PATH.read_text())["winner"]
    table["oof"] = oof_predictions(
        table[list(FEATURE_NAMES)], table["target"], table["block"], build_models()[winner]
    )

    merged = coords.merge(
        table[["cell_lat", "cell_lon", "claim_count", "oof", "elev_mean"]],
        on=["cell_lat", "cell_lon"],
        how="inner",
    )
    # One row per cell; the same cell repeats across scenarios and years.
    merged = merged.drop_duplicates(subset=["cell_lat", "cell_lon"])

    confidence = alignment_confidence(merged)
    lines = [
        f"Matched cells: {len(merged)} of {len(coords)} scored rows",
        f"Alignment confidence (rule score vs elevation): {confidence:.3f}",
    ]

    if not np.isfinite(confidence) or confidence < min_confidence:
        lines.append(
            "ALIGNMENT REJECTED. The reconstructed coordinates do not track "
            "elevation as a higher-is-safer score must. The comparison is not "
            "reported, because publishing it would mean trusting a mapping the "
            "data contradicts."
        )
        return "\n".join(lines)

    if len(merged) < 10:
        lines.append("Too few matched cells to compare.")
        return "\n".join(lines)

    rule_rho = float(spearmanr(merged["claim_count"], merged["rule_score"]).statistic)
    model_rho = float(spearmanr(merged["claim_count"], merged["oof"]).statistic)

    lines += [
        "",
        "Spearman correlation with observed claim counts:",
        f"  rule engine score (higher = safer): {rule_rho:+.3f}",
        f"  learned model, out-of-fold:         {model_rho:+.3f}",
        "",
        "The rule score is expected to correlate negatively, since higher means",
        "safer and more claims mean more risk. Compare absolute magnitudes.",
    ]

    if abs(model_rho) > abs(rule_rho):
        lines.append("RESULT: the learned model ranks risk more accurately.")
    else:
        lines.append(
            "RESULT: the rule engine ranks risk at least as well on these cells. "
            "Reported as found, without re-tuning the model to overturn it."
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_baseline.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Run the comparison and accept whatever it says**

Run: `.venv/Scripts/python.exe -m migracoast_ml baseline`

Read the alignment confidence first. If the comparison is rejected, that is a
valid outcome: record it in the README and move on. If it runs, record both
correlation values exactly as printed.

**Do not adjust the model, the features, or `min_confidence` to change this
result.** The spec pre-registered this commitment. A documented loss with
correct methodology is worth more than a win produced by tuning against the
test.

- [ ] **Step 6: Commit**

```bash
git add migracoast_ml/baseline.py tests/test_baseline.py
git commit -m "feat: compare rule engine against the learned model"
```

---

## Task 24: Results notebook

**Files:**
- Create: `notebooks/results.ipynb`

The notebook reads saved artifacts and never recomputes the model, so it opens
in seconds and always matches the committed results.

- [ ] **Step 1: Create the notebook with these cells**

Cell 1, setup:

```python
import json
import matplotlib.pyplot as plt
import pandas as pd
from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.dataset import load_training_table
from migracoast_ml.models.train import METRICS_PATH, MODEL_PATH
from migracoast_ml.models.scoring import RiskScorer

table = load_training_table()
metadata = json.loads(METRICS_PATH.read_text())
scorer = RiskScorer.load(MODEL_PATH)
print(f"Winner: {metadata['winner']}, cells: {metadata['n_cells']}")
```

Cell 2, the model ladder:

```python
metrics = pd.DataFrame(metadata["metrics"])
metrics[["model", "rmse_mean", "rmse_std", "r2_mean",
         "coastal_rmse_mean", "coastal_spearman_mean"]]
```

Cell 3, target distribution:

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(table["claim_count"], bins=50)
axes[0].set_yscale("log")
axes[0].set_title("Claims per cell")
axes[1].hist(table["target"], bins=50)
axes[1].set_title("log1p(claims), the modelling target")
plt.tight_layout()
```

Cell 4, feature importance:

```python
from migracoast_ml.models.explain import permutation_importance_table

importance = permutation_importance_table(
    scorer.model, table[list(FEATURE_NAMES)], table["target"], n_repeats=5
)
importance.plot.barh(x="feature", y="importance_mean", figsize=(8, 5), legend=False)
plt.gca().invert_yaxis()
plt.xlabel("Drop in R squared when shuffled")
plt.tight_layout()
importance
```

Cell 5, spatial residual map:

```python
from migracoast_ml.models.evaluate import oof_predictions
from migracoast_ml.models.ladder import build_models

table["oof"] = oof_predictions(
    table[list(FEATURE_NAMES)], table["target"], table["block"],
    build_models()[metadata["winner"]],
)
table["residual"] = table["target"] - table["oof"]

coastal = table[table["coast_km"] < 30]
plt.figure(figsize=(9, 7))
plt.scatter(coastal["cell_lon"], coastal["cell_lat"],
            c=coastal["residual"], cmap="RdBu_r", s=12,
            vmin=-2, vmax=2)
plt.colorbar(label="Actual minus predicted")
plt.title("Out-of-fold residuals, coastal cells")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
```

Cell 6, predicted against actual:

```python
plt.figure(figsize=(6, 6))
plt.scatter(coastal["oof"], coastal["target"], s=8, alpha=0.4)
limit = max(coastal["target"].max(), coastal["oof"].max())
plt.plot([0, limit], [0, limit], "k--", lw=1)
plt.xlabel("Predicted log1p(claims)"); plt.ylabel("Actual log1p(claims)")
plt.title("Out-of-fold predictions, coastal cells")
```

Cell 7, baseline comparison:

```python
from migracoast_ml.baseline import run_comparison
print(run_comparison())
```

Cell 8, a markdown cell titled "Limitations" containing this text:

> The target is claim count, not flood probability. A cell with many claims may
> simply hold many insured buildings, because the public data has no policy
> count to divide by. The model predicts claim intensity and should be read as
> such. Coordinates are rounded by FEMA to 0.1 degrees, which fixes the
> resolution at roughly 11 km. Storm and flood counts cover 2000 to 2024 only.

- [ ] **Step 2: Run the notebook end to end**

Run: `.venv/Scripts/python.exe -m pytest --version` to confirm the environment,
then execute every cell in order. Every cell must run without error.

- [ ] **Step 3: Commit**

```bash
git add notebooks/results.ipynb
git commit -m "docs: add results notebook"
```

---

## Task 25: README and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the ML section to README.md**

Insert after the existing "Files" section. Fill the bracketed values from
`models/metrics.json` and the Task 23 output; do not leave brackets in place.

````markdown
## Machine Learning: Learned Risk Model

The original risk score combines five sub-scores with hand-picked weights and
normalization caps. None of those constants were fit to observed flood
outcomes. This component learns risk from data instead.

### Approach

**Target.** FEMA National Flood Insurance Program claims, about 790,000 across
the five covered states, aggregated to 0.1 degree grid cells. FEMA rounds claim
coordinates to one decimal place, which sets the resolution of the whole model.

**Features.** Eleven physical predictors: terrain statistics from ETOPO 2022
elevation, distance to coastline, NOAA 2022 sea level projections for 2050 and
2100, and flood and storm event rates from NOAA Storm Events for 2000 to 2024.

Latitude, longitude and state are deliberately excluded. Including them would
let a tree model memorize geography and post an impressive score that means
nothing.

**Models.** Ridge regression, histogram gradient boosting, and a two-layer
neural network, all fit on identical features.

**Validation.** Five-fold cross-validation grouped by 1 degree spatial block.
Neighbouring cells are strongly correlated, so a random split would leak a
cell's neighbour into the test fold and inflate the score.

### Results

Selected model: [winner]. Coastal RMSE [value], coastal Spearman [value].
Full metrics for all three models are in `models/metrics.json`.

Against the original rule engine on the [N] cells recoverable from
`score_time_series.csv`: [state both correlations and which ranked risk better,
exactly as the baseline command reported them].

### Limitations

The target is claim count, not flood probability. The public data has no policy
count to divide by, so a cell with many claims may simply have many insured
buildings. The model predicts claim intensity and must be read as such.

Coordinates are rounded to 0.1 degrees, roughly 11 km, so this is a
neighbourhood-scale model, not a property-scale one. Storm and flood history
covers 2000 to 2024.

### Usage

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m migracoast_ml fetch      # download source data
.venv/Scripts/python.exe -m migracoast_ml build      # build the training table
.venv/Scripts/python.exe -m migracoast_ml train      # fit and select
.venv/Scripts/python.exe -m migracoast_ml evaluate   # print metrics
.venv/Scripts/python.exe -m migracoast_ml baseline   # compare to the rule engine
.venv/Scripts/python.exe -m migracoast_ml serve      # run the API
```

The trained model is committed, so `serve` works immediately after cloning.
Only re-run `fetch` and `train` to rebuild from scratch.

### API

| Route | Method | Purpose |
|---|---|---|
| `/ml/risk` | POST | Score one coordinate |
| `/ml/heatmap` | POST | Score a grid, up to 2500 cells |
| `/ml/model-info` | GET | Model name, metrics, features |

### Database credentials

Connection details now come from the environment: `MIGRACOAST_DB_HOST`,
`MIGRACOAST_DB_USER`, `MIGRACOAST_DB_PASSWORD`, `MIGRACOAST_DB_NAME`.

A password was previously committed in plaintext and remains in this
repository's git history. **It should be rotated on the database server.**
````

- [ ] **Step 2: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: every test passes. Record the count. If anything fails, fix it before
continuing; do not proceed with a failing suite.

- [ ] **Step 3: Verify the pipeline runs end to end from the CLI**

```bash
.venv/Scripts/python.exe -m migracoast_ml evaluate
.venv/Scripts/python.exe -m migracoast_ml baseline
```

Expected: both print output without traceback.

- [ ] **Step 4: Verify no secret and no data file is staged**

```bash
git status --short
grep -r "St5SfZXpxRPS4XZy" --include=*.py . || echo "no plaintext password in source"
```

Expected: no files under `data/`, and the grep finds nothing.

- [ ] **Step 5: Confirm the served numbers are sane**

Start the server, then score a low-lying coastal point and an inland one:

```bash
curl -s -X POST http://127.0.0.1:5001/ml/risk -H "Content-Type: application/json" -d "{\"latitude\": 25.77, \"longitude\": -80.19}"
curl -s -X POST http://127.0.0.1:5001/ml/risk -H "Content-Type: application/json" -d "{\"latitude\": 40.44, \"longitude\": -79.99}"
```

Expected: downtown Miami scores as riskier than inland Pittsburgh. If the
ordering is reversed, the scoring direction is inverted somewhere and must be
fixed before this is called done.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document the learned risk model"
```

- [ ] **Step 7: Report honestly**

Write a short summary covering: which model won, the coastal metrics, the
baseline comparison outcome including a loss if that is what happened, the test
count, and anything left incomplete. Do not describe as verified anything that
was not actually run.

---

## Plan self-review

Checked against the design spec on 2026-09-15.

**Spec coverage.** Every section maps to at least one task:

| Spec section | Tasks |
|---|---|
| 3.1 FEMA claims | 9, 14 |
| 3.2 Storm events | 10, 14 |
| 3.3 ETOPO elevation | 7, 14 |
| 3.4 Sea level | 4, 14 |
| 3.5 Natural Earth | 3, 14 |
| 4 Spatial unit and target | 2, 15 |
| 5 Features | 5, 6, 8, 11, 12 |
| 6 Models | 16 |
| 7 Evaluation | 17, 19 |
| 8 Baseline comparison | 23 |
| 9 Serving | 21, 22 |
| 10 Security fix | 22 |
| 11 Repository layout | 1, 13 |
| Fetcher crash-safety (added after Task 3 review) | 3b, 4, 7, 9, 10 |
| 12 Testing | every task |
| 13 What is committed | 1, 19 |
| 14 Deliverables | 24, 25 |

**Placeholder scan.** No TBD, TODO, or "implement later" appears. Every code
step carries complete code. The only bracketed values are in the README
template at Task 25, where Step 1 explicitly instructs filling them from
generated artifacts.

**Type consistency.** Names verified identical across every task:
`snap_to_cell`, `block_id`, `build_cell_universe`, `haversine_km`,
`CoastlineIndex.distance_km`, `ElevationGrid.from_dir`, `terrain_features`,
`SeaLevelIndex.features`, `HazardIndex.features`, `FeatureBuilder.build`,
`FeatureBuilder.build_frame`, `FeatureBuilder.load`, `RiskScorer.score_features`,
`RiskScorer.save`, `RiskScorer.load`, `spatial_cv`, `oof_predictions`,
`select_winner`, `train_from_table`, `create_blueprint`, `MODEL_PATH`,
`METRICS_PATH`, `FEATURE_NAMES`.

Feature key agreement checked: `terrain_features` returns the six terrain keys,
`CoastlineIndex` supplies `coast_km` via the builder, `SeaLevelIndex` returns
the two sea level keys, `HazardIndex` returns the two hazard keys. Eleven keys
total, matching `FEATURE_NAMES` exactly.

**Ordering.** Every module is defined before the task that imports it.
`features/grid.py` (Task 2) precedes `sources/fema.py` (Task 9) which imports
`snap_to_cell`. Real data arrives in Task 14, before the first task that needs
it, Task 15.
