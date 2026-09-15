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
