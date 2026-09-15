# Learned Coastal Flood Risk Model for Migracoast

**Date:** 2026-09-15
**Status:** Approved design, ready for implementation planning

## 1. Problem and motivation

Migracoast today produces a coastal risk score from a hand-built rule engine in
`app.py`. Every number in that score is a human judgment:

- Five sub-scores are combined with fixed weights (`app.py:602-612`).
- Each sub-score is normalized against a hand-picked cap, for example
  `max_expected_slr=3.5` feet, `max_storm_event_count=45`, `max_flood_events=15`.
- The inundation sub-score is a step function of the projected inundation year.

None of these constants were fit to observed flood outcomes, and the system has no
learned component anywhere. It is also unrunnable in its current state: the NOAA
inundation rasters, the IPCC AR6 NetCDF files, and the NOAA Storm Events CSV are not
in the repository and are no longer available locally. `app.py` cannot even be
imported, because line 640 reads the storm CSV at module scope.

This project adds a machine learning system that learns coastal flood risk from
observed flood insurance claims, using only data that a script can download. Goals,
in priority order:

1. **Portfolio value.** The ML work must be visible, well structured, honestly
   evaluated, and defensible in a technical interview.
2. **Reproducibility.** Anyone who clones the repository must be able to fetch the
   data, train the model, and serve predictions without hunting for private files.
3. **Capability.** Predictions must work anywhere in the covered states, including
   locations the raster pipeline never covered, and must be fast enough to score a
   whole map grid in one request.

**Non-goals.** Restoring the raster pipeline. Deep learning on elevation imagery
(considered and deferred: it requires multi-gigabyte tile downloads). Changing the
website frontend.

## 2. Scope

**Geography.** Massachusetts, New York, Pennsylvania, Florida, California. These are
the five regions present in `score_time_series.csv`, which makes the baseline
comparison in section 8 possible.

**The existing rule engine is not modified or deleted.** It remains in the repository
as the documented baseline. It stays unrunnable until its data files are restored,
which is outside this project's scope.

## 3. Data sources

Every source below was verified reachable on 2026-09-15. Each has a dedicated fetcher
module under `migracoast_ml/sources/`. Fetchers write to a gitignored `data/raw/`
directory at the repository root, and every fetcher is safe to re-run: an already
downloaded file is skipped unless the caller forces a refresh.

### 3.1 FEMA NFIP redacted claims (target variable)

- **Endpoint:** `https://www.fema.gov/api/open/v3/NfipClaims`
- **Why v3:** the v2 endpoint `FimaNfipClaims` is deprecated and will be removed on
  2026-10-15. Use v3 from the start. Note the field renames: v2 `censusTract` became
  v3 `censusGeoid`, and v2 `floodZoneCurrent` became v3 `ratedFloodZone`.
- **Fields selected:** `latitude`, `longitude`, `state`, `yearOfLoss`, `dateOfLoss`,
  `censusGeoid`, `countyCode`, `reportedZipCode`, `amountPaidOnBuildingClaim`,
  `amountPaidOnContentsClaim`, `ratedFloodZone`
- **Paging:** `$top` maximum is 10000, page with `$skip`, request `$format=csv`.
  Filter per state with a `$filter` equality clause on `state`.
- **Verified record counts:**

  | State | Claims |
  |-------|--------|
  | MA | 35,375 |
  | NY | 175,430 |
  | PA | 77,060 |
  | FL | 448,618 |
  | CA | 53,563 |
  | **Total** | **790,046** |

- **Critical property:** FEMA rounds `latitude` and `longitude` to one decimal place
  for privacy. This sets the spatial resolution of the entire project. FEMA warns
  that rounding can place a point in the wrong county or state.

### 3.2 NOAA Storm Events (hazard history features)

- **Directory:** `https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/`
- **Filename pattern:** `StormEvents_details-ftp_v1.0_d{YEAR}_c{YYYYMMDD}.csv.gz`
- **Important:** the `c` suffix is a creation date that differs per file and changes
  when NOAA republishes. The fetcher must list the directory and match by year with a
  regular expression. Hardcoding filenames will break.
- **Years:** 2000 through 2024, roughly 10 to 13 MB compressed per year.
- **Columns used:** `BEGIN_LAT`, `BEGIN_LON`, `EVENT_TYPE`, `BEGIN_DATE_TIME`,
  `EVENT_ID`, `CZ_NAME`, `CZ_TYPE`, `STATE`, `YEAR`

### 3.3 ETOPO 2022 elevation (terrain features)

- **Service:** NOAA THREDDS NetCDF Subset Service, which returns a clipped box rather
  than the 1.6 GB global file.
- **URL template:** the base path is
  `https://www.ngdc.noaa.gov/thredds/ncss/grid/global/ETOPO2022/30s/30s_surface_elev_netcdf/ETOPO_2022_v1_30s_N90W180_surface.nc`
  with query parameters `var=z`, the four bounding box edges `north`, `south`, `west`,
  `east`, and `accept=netcdf`.
- **Verified constraints:**
  - `accept=netcdf` works. `accept=netcdf4` returns HTTP 400 with
    "Format netcdf4 is not supported for Grid data request".
  - The 30 arc-second dataset works. The 60 arc-second dataset returned HTTP 500 and
    must not be used.
  - A Florida-sized box of roughly 7 by 6 degrees returns 3.1 MB in under 2 seconds.
- **Resolution:** 30 arc-second, about 1 km at these latitudes.
- **Variable:** `z`, in meters, fill value -99999. Negative values are below sea level
  or ocean.

### 3.4 NOAA 2022 Sea Level Rise scenarios (sea level feature)

- **URL:** the zip at `oceanservice.noaa.gov/hazards/sealevelrise/Sea_Level_Rise_Datasets_2022.zip`,
  which redirects to `cdn.oceanservice.noaa.gov`. The fetcher must follow redirects.
- **Size:** 476 KB.
- **File used:** the sea level projections CSV inside the archive, 17,220 data rows.
- **Parsing:** the real header is on line 18 of the file, index 17 counting from zero.
  The first 17 lines are citation and notes. Rows are padded with many trailing empty
  columns which must be dropped.
- **Columns:** `PSMSL Site`, `PSMSL ID`, `NOAA ID`, `NOAA Name`, `RSL GridNum`, `Lat`,
  `Long`, `Regional Classification`, a US coastline intersect flag, `Scenario`, a
  vertical land motion trend, a 1992-to-2000 offset, a 2000-to-2005 offset, then
  relative sea level columns for 2005 and each decade from 2020 through 2150.
- **Scenario encoding:** five global scenarios named by their 2100 rise in meters
  (0.3, 0.5, 1.0, 1.5, 2.0), each with LOW, MED and HIGH variants for the 17th, 50th
  and 83rd percentiles. Fifteen rows per site. The project default is the 1.0 metre
  median, the Intermediate scenario, configurable in `config.py`.
- **Units:** centimeters. Convert to feet for consistency with the rest of the
  project. Per the file's own note 5, add the 2000-to-2005 offset to each projection
  value to initiate projections at year 2000.
- **Why this instead of the AR6 NetCDF:** the Zenodo AR6 archive is a 9.2 GB zip. This
  NOAA product is derived from the same AR6 projections, is US-specific, and is 476 KB.

### 3.5 Natural Earth boundaries (cell universe and coastline distance)

- **Coastline:** the 10m physical coastline zip on `naciscdn.org`, 3.0 MB
- **States:** the 10m cultural admin-1 states and provinces zip on `naciscdn.org`, 14.9 MB

## 4. Spatial unit and dataset construction

**Grid cells.** FEMA rounds coordinates to 0.1 degree, so the training unit is a
0.1 degree cell centered on that lattice, with a half-width of 0.05 degree. Every
claim maps to exactly one cell by construction. No finer resolution is defensible,
because smaller cells would invent precision the target does not have.

**Cell universe.** Every 0.1 degree cell whose center falls inside one of the five
state polygons. Estimated at roughly 9,000 cells. Inland cells are retained as
genuine low-risk examples rather than filtered out, because a model trained only on
coastal cells would have no negative examples and would learn nothing about what makes
a place safe. Evaluation additionally reports a coastal-only subset.

**Target variable.** The natural log of one plus the claim count per cell, summed over
all available years.

**Documented limitation, to be stated in the README and the notebook.** There is no
policy-count denominator in this dataset. A cell with many claims may simply have many
insured buildings. The model therefore predicts *claim intensity*, not the probability
that a given house floods. This is a real limitation of the public data and must not
be papered over. Total dollars paid is carried through for analysis but is not the
modeling target, because dollar amounts span decades of uncorrected inflation.

## 5. Features

All features are computed by a single function that takes a center point and a window
half-width, so training cells and live inference requests share one code path. This
eliminates training and serving skew by construction.

**Terrain,** from the ETOPO window covering the cell:
- mean elevation over land pixels
- minimum elevation
- 10th percentile elevation
- fraction of land pixels below 2 meters
- fraction of land pixels below 5 meters
- fraction of the window that is land at all

**Coast:**
- distance in kilometers from the cell center to the nearest coastline geometry

**Sea level,** from the nearest NOAA projection point under the configured scenario:
- projected relative sea level rise at 2050, in feet
- projected relative sea level rise at 2100, in feet

**Hazard history,** within the cell window, using the same event type lists as the
rule engine so the comparison is fair:
- flood-type events per year (Flood, Flash Flood, Coastal Flood)
- storm-type events per year (Thunderstorm Wind, Tornado, Hail, Hurricane)

Events lacking coordinates are excluded. The rule engine additionally matched events
by county name; that path is dropped here because it introduces a population-size bias
that the model would exploit.

**Deliberately excluded:** latitude, longitude, and state. Including them lets a
tree model memorize geography and produce an impressive but meaningless test score.
Their exclusion forces the model to learn physical drivers and is what makes the
feature importances interpretable.

## 6. Models

Three scikit-learn models are fit on identical features, forming a ladder that makes
the model selection itself part of the story:

| Model | Role |
|-------|------|
| `Ridge` | Linear baseline. Establishes what a simple model achieves. |
| `HistGradientBoostingRegressor` | Primary candidate. Handles nonlinearity and missing values natively. |
| `MLPRegressor`, two hidden layers | Neural comparison on tabular data. |

Deliberately no PyTorch, LightGBM, or XGBoost. On a dataset of roughly 9,000 rows and
11 features they would add installation friction on Windows without adding accuracy,
and a reviewer would reasonably ask why a neural network was used on tabular data this
small. Being able to explain that choice is worth more than the library name.

Numeric features are imputed and scaled inside a scikit-learn `Pipeline`, so no
preprocessing step can leak across a fold.

## 7. Evaluation

**Spatial cross-validation is mandatory.** Neighboring cells are strongly correlated;
a random split would place a cell's neighbor in the test set and report an inflated
score. Cells are grouped into 1 degree blocks and split with five-fold `GroupKFold` on
that block identifier.

**Metrics,** each reported as mean and standard deviation across folds:
- root mean squared error in log space
- R squared
- Spearman rank correlation

Each metric is reported twice: over all cells, and over the coastal subset within
30 km of the coastline.

**Selection criterion:** lowest coastal root mean squared error. Coastal locations are
the product's actual use case, so a model that wins only by predicting inland zeros
well is not the better model.

**Artifacts written by training:**
- the winning pipeline, refit on all cells, serialized with joblib
- feature names, training date, scenario, and state list
- the full cross-validation metrics table for all three models
- the distribution of predictions over coastal training cells, needed for percentile
  scoring at serve time

**Explainability:**
- Global permutation importance, always computed.
- Per-prediction SHAP contributions when the `shap` package is installed. When it is
  absent the API returns a clear "unavailable" marker rather than failing. SHAP is an
  optional dependency.

## 8. Baseline comparison against the rule engine

`score_time_series.csv` holds 480 rows of rule-engine output with no coordinates
attached. The coordinates are recoverable from the point lists in
`Check_inundation.ipynb`, and the row counts reconcile exactly:

| Region | Rows per scenario | Source lists (in concatenation order) |
|--------|-------------------|----------------------------------------|
| MA | 28 | `MA_points` |
| NY | 40 | `NY_points` |
| PA | 40 | `PA_points` + `PA_points_new` |
| FL | 40 | `FL_points` + `FL_points_new` |
| CA | 40 | `CA_points` + `CA_points_new` |
| BOS | 52 | `BOS_points` + `BOS_points_new` + `MA_points_new` |

That totals 240 rows per scenario and 480 rows across the two SSP scenarios, matching
the file exactly. The BOS grouping absorbs `MA_points_new` because those twelve
coordinates are downtown Boston, near 42.34 N, 71.07 W.

**This mapping is an inference from row counts, not a certainty.** Concatenation order
within a region is assumed to follow the order above. The implementation must validate
it empirically before the comparison is reported: under a correct alignment, the
rule-engine score should correlate positively with cell elevation, since the score is
oriented so that higher means safer. If that correlation is absent or negative, the
alignment is wrong and the comparison must be reported as not possible rather than
published on a guess.

**The comparison.** On the reconstructed cells, compute the Spearman correlation
between observed claim counts and each of two predictors: the rule-engine score, and
the learned model's out-of-fold prediction. Only out-of-fold predictions may be used,
otherwise the learned model is being evaluated on its own training data.

**Pre-registered honesty commitment.** The learned model may lose on this small set of
cells. If it does, that result is reported as the finding. The model will not be
re-tuned until it wins, and the comparison will not be quietly dropped. A documented
negative result with correct methodology is stronger evidence of competence than an
unexplained win.

## 9. Serving

`app.py` currently cannot be imported, because line 640 reads a missing CSV at module
scope. The ML API is therefore standalone, and integration with the rule engine is
optional rather than required.

A Flask blueprint in `migracoast_ml/api.py` exposes three routes:

**`POST /ml/risk`** — body carries a latitude and a longitude. Returns the predicted
claim intensity, the risk percentile from 0 to 100 where higher is riskier, a safety
score of 100 minus that percentile to match the site's existing higher-is-safer
convention, every computed input feature, and the top contributing features.

**`POST /ml/heatmap`** — body carries a bounding box and a resolution. Returns a grid
of scored cells. Hard cap of 2500 cells per request, rejected with a clear message
above that. This is the capability the current website does not have.

**`GET /ml/model-info`** — returns the winning model name, its cross-validation
metrics, the feature list, the scenario, and the training date.

Errors return a descriptive message with a 4xx status, never a stack trace. The cases
that must be handled explicitly are: coordinates outside the five covered states,
missing or non-numeric coordinates, and a grid request over the cap.

The model and serving bundle load once at process start and are held in memory. A
point query is expected to take a few milliseconds.

`serve.py` runs the blueprint standalone. A small guarded block in `app.py` registers
the same blueprint when the package imports successfully and logs a warning otherwise,
so the two systems coexist if the rule engine's data is ever restored.

## 10. Security fix

`app.py:30-35` contains a MySQL password in plaintext in a public repository. As part
of this work the credentials move to environment variables, and `.env` is added to
`.gitignore`. The committed history still contains the password, so the README will
note that the credential should be rotated. Rotating it is the repository owner's
action, not something this project can do.

## 11. Repository layout

```
migracoast_ml/
  __init__.py
  __main__.py          single CLI entry point
  config.py            states, paths, scenario, constants
  sources/             downloaders, one per external source
    fema.py            NFIP claims
    storms.py          NOAA Storm Events
    elevation.py       ETOPO via THREDDS
    sealevel.py        NOAA 2022 SLR
    boundaries.py      Natural Earth
    fetch_all.py       orchestrator
  features/
    grid.py            cell universe from state polygons
    terrain.py
    coast.py
    sealevel.py
    hazards.py
    builder.py         assembles one feature vector for any center point
  dataset.py           joins claims to features, writes the training table
  models/
    ladder.py          the three model definitions
    evaluate.py        spatial cross-validation
    train.py
    scoring.py         percentile scoring
    explain.py         permutation importance and SHAP
  baseline.py          rule score versus learned score
  api.py               Flask blueprint
  serve.py             standalone server
tests/
notebooks/
  results.ipynb        reads saved artifacts, recomputes nothing
docs/superpowers/specs/
```

Downloaded and derived files live outside the package, at the repository root:

```
data/raw/              downloaded source files, gitignored
data/processed/        the training table and serving bundle, gitignored
models/                the trained model, metrics, and figures, committed
```

The package directory `migracoast_ml/sources/` holds the code that downloads, and the
root `data/` directory holds what was downloaded. They are deliberately named
differently so the two are never confused.

The command line is one entry point with subcommands: `fetch`, `build`, `train`,
`evaluate`, `baseline`, `serve`.

## 12. Testing

Pytest, with small fixtures committed so the suite runs with no network access.

- Grid generation: cells fall inside state polygons, count is in the expected range.
- Terrain features: computed against synthetic elevation arrays with known answers,
  including an all-ocean window and an all-missing window.
- Coastline distance: a known point with an independently computed distance.
- Sea level parsing: the 17 junk header lines are skipped, units convert correctly,
  the scenario filter selects one row per site.
- Storm filename matching: the year regular expression matches regardless of the
  creation-date suffix.
- Claim-to-cell join: a claim lands in exactly one cell, boundary cases included.
- Percentile scoring: values below, above, and inside the reference distribution.
- API routes: all three, against a stub model, plus every error case in section 9.
- Feature parity: the training path and the serving path produce identical vectors for
  the same point. This is the test that prevents training and serving skew.

Network-dependent fetchers are tested against recorded responses and marked so they can
be deselected.

## 13. What is committed

Committed: all source, tests, fixtures, the trained model artifact, the metrics JSON,
generated figures, and the results notebook. Committing the trained model means the API
works immediately after a clone, which matters when a reviewer has five minutes.

Gitignored: `data/raw/`, `data/processed/`, `.env`, virtual environments, and Python
build artifacts. The data directories are regenerated by the `fetch` subcommand.

The repository has no `.gitignore` at all today, so creating one is part of this work
and must happen before any data is downloaded into the working tree.

## 14. Deliverables

1. The `migracoast_ml` package as laid out in section 11.
2. A passing test suite.
3. A trained, committed model with its metrics.
4. A results notebook: learning curves, residual maps, feature importance, and the
   baseline comparison.
5. A rewritten README section covering the problem framing, data sources, the model
   ladder, spatial validation, the honest limitations from section 4, and the
   credential rotation note from section 10.

## 15. Known risks

| Risk | Mitigation |
|------|------------|
| The learned model loses to the rule engine on the 480 comparison cells | Report it as the finding, per section 8. Do not re-tune to win. |
| Claim counts reflect insurance uptake, not flood hazard | Documented prominently in README and notebook. It is the central caveat. |
| The `score_time_series.csv` coordinate alignment is wrong | Validate against elevation correlation before publishing the comparison; drop the comparison if it fails. |
| NOAA republishes Storm Events files with new creation dates | Fetcher matches by year with a regular expression, never hardcoded names. |
| FEMA v2 endpoint removal on 2026-10-15 | v3 is used from the start. |
| The local Python 3.13 install is missing numpy | Environment setup is the first implementation step, with a pinned requirements file. |
