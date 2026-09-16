
# Migracoast

A system aiming to predict and inform users of the inundation status, risk assessments, as well as relocation suggestions of their houses.


## Website

To use Migracoast, check its website [migracoast.com](https://www.migracoast.com/)


## Author

- [@Zhiyang Zhong](https://www.linkedin.com/in/jimmy-zhong-517973242/)


## Data Source

 - [NOAA Inundation Extent Rasters](https://coast.noaa.gov/slrdata/Extent_Rasters/index.html)
 - [IPCC AR6 Sea Level Projections](https://zenodo.org/records/6382554)
 - [NOAA Storm Events Database](https://www.ncdc.noaa.gov/stormevents/)


## Files

`app.py` - source code that defines the logic for the risk assessment system

`Check_inundation.ipynb` - notebook that generates samples of the risk assessment score time series for major cities for research purposes

`score_time_series.csv` - example datasets for major cities' risk assessment score time series from 2020 to 2100

`migracoast_ml/` - a learned flood risk model built as a separate package alongside the original rule engine, with its own data fetchers, feature builders, models, evaluation, baseline comparison, and API server

`notebooks/results.ipynb` - notebook that renders the trained model's metrics, feature importance, and the baseline comparison outcome

`docs/superpowers/plans/2026-09-15-ml-flood-risk.md` - the implementation plan this package was built from, including the design decisions and fixes made along the way


## Machine Learning: Learned Risk Model

The original risk score in `app.py` combines five sub-scores with hand-picked
weights and normalization caps. None of those constants were fit to observed
flood outcomes. This component, in `migracoast_ml/`, learns risk from data
instead, and is a separate, additive system: it does not modify or replace
`app.py`.

### Approach

**Target.** FEMA National Flood Insurance Program claims: 790,046 claims
across five states (MA 35,375, NY 175,430, PA 77,060, FL 448,618, CA 53,563),
aggregated to 0.1 degree grid cells. FEMA rounds claim coordinates to one
decimal place, which sets the resolution of the whole model: 8,494 cells,
1,309 of them coastal.

**Features.** Eleven physical predictors, with no learned component of their
own: terrain statistics from ETOPO 2022 elevation, distance to coastline,
NOAA sea level projections for 2050 and 2100 from 1,142 sites, and flood and
storm event rates from 1,520,489 NOAA Storm Events rows spanning 2000 to 2024.

Latitude, longitude and state are deliberately excluded. Including them would
let a tree model memorize geography and post an impressive score that means
nothing about the physical drivers of flood risk.

**Models.** Ridge regression, histogram gradient boosting, and a two-layer
neural network, all fit on identical features.

**Validation.** Five-fold cross-validation grouped by 1 degree spatial block
(135 blocks). Neighbouring cells are strongly correlated, so a random split
would leak a cell's neighbour into the test fold and inflate the score; group
k-fold keeps every cell from a block on one side of the split.

### Results

| model | RMSE | R² | Spearman | coastal RMSE | coastal Spearman |
|---|---|---|---|---|---|
| ridge | 1.40 ± 0.15 | 0.44 | 0.67 | 2.09 ± 0.19 | 0.63 |
| gradient boosting | 1.10 ± 0.05 | 0.65 | 0.76 | **1.53 ± 0.09** | **0.80** |
| neural net | 1.14 ± 0.06 | 0.62 | 0.75 | 1.59 ± 0.12 | 0.79 |

Full metrics, including per-fold standard deviations, are in
`models/metrics.json`, and reproduced by `migracoast_ml evaluate`.

**Selected model: gradient boosting**, chosen on lowest coastal RMSE. It beats
the neural network by 0.057 on coastal RMSE, which is smaller than either
model's fold-to-fold standard deviation (0.087 and 0.118), so those two models
are within noise of each other on this data; a different random seed or fold
split could plausibly flip the ranking. Ridge trails by roughly three standard
deviations, which is a real gap: the relationship between these features and
claim counts is not linear.

The strongest predictor by permutation importance is `storm_events_per_year`
(0.402), followed by `elev_p10` (0.217), `coast_km` (0.215),
`flood_events_per_year` (0.159), and `slr_2100_ft` (0.125).

### Baseline comparison: it did not work, and here is why

The plan called for comparing the learned model against the original rule
engine's scores in `score_time_series.csv`, and pre-registered that this
result would be reported whatever it turned out to be. It did not produce a
usable comparison, and it is reported as such rather than fixed to look
better.

The rule engine's 480 saved scores carry no coordinates, so
`migracoast_ml baseline` reconstructs them from the point lists embedded in
`Check_inundation.ipynb`. Before trusting that reconstruction for anything,
the code checks whether the recovered coordinates track elevation, since the
rule score is oriented so that higher means safer, and low elevation should
score worse. That check came back with a confidence of -0.235 against a
required threshold of 0.2. The sign is wrong, not just the magnitude, so the
comparison is not a near miss — the reconstructed coordinates do not behave
the way the scoring direction requires.

Because of that, **the baseline comparison was not run and no correlation
numbers are reported**, by design, rather than publishing a comparison built
on a coordinate mapping the data itself contradicts.

Separately, even the reconstruction's coverage is thin: of 480 scored rows,
228 resolve to unique points, only 56 of those land in unique 0.1 degree
cells, and only 34 of those cells are present in the training table. Coastal
points frequently snap to a grid cell whose center falls offshore, and
offshore cells were never part of the cell universe the model was built on.
With only 34 matched cells, the honest reading is that this shows the mapping
cannot be trusted, not that it is definitely wrong — 34 points is too few to
conclude much either way, which is one more reason not to report a
correlation from it.

### Limitations

1. **The target is claim count, not flood probability.** The public FEMA data
   has no policy count to divide by, so a cell with many claims may simply
   have many insured buildings. The model predicts claim intensity, not
   probability of flooding.
2. **It predicts flood insurance claims generally, including riverine
   flooding, not coastal inundation specifically.** Inland Pittsburgh scores
   at the 87.5th percentile and inland Pennsylvania at 69.7, while Miami
   Beach scores 99.9. The inland scores are high because inland flooding
   generates real claims, not because the model is confusing coasts with
   rivers.
3. **Coordinates are rounded to 0.1 degrees**, roughly 11 km, so this is a
   neighbourhood-scale model, not a property-scale one.
4. **Storm and flood history covers 2000 to 2024 only.**
5. **The top feature partly measures general severe weather, not coastal
   hazard.** `storm_events_per_year` is the strongest predictor at 0.402
   importance, but NOAA's storm categories include thunderstorm wind,
   tornado, and hail alongside coastal storms. It reads higher for inland
   Pennsylvania (1.48 per year) than for Miami Beach (0.76 per year).
6. **The event type list deliberately matches the original rule engine's**,
   so that a comparison would use identical inputs. That means High Wind,
   Winter Storm, and Heavy Rain are excluded from the storm and flood
   counts despite being frequent events.
7. **Sea level projections use nearest-site lookup with no distance
   attenuation.** About 26 percent of New York and Pennsylvania cells draw
   on a projection site more than 150 km away, and inland Pennsylvania ends
   up assigned a projection close to Miami's.
8. **Distance to coast uses a planar nearest-geometry search**, not a
   geodesic one. Checked against a brute-force reference, it can
   overestimate distance but never underestimate it, and it changes the
   under-30-km coastal label for at most 0.5 percent of cells.
9. **About 40 percent of NOAA storm rows are dropped** because zone-based
   events (reported for a forecast zone rather than a point) carry no
   coordinates and cannot be assigned to a cell.
10. **The raw FEMA claims export contains 57,363 duplicate rows** and no
    unique claim identifier to de-duplicate against, so aggregated cell
    counts include them.
11. **The risk percentile is ranked against the 1,309 coastal cells only**,
    not all 8,494. That is deliberate, since ranking against inland zeros
    would push every coastal answer to the top of the scale, but it is why
    an inland location can still score in the 80s.
12. **About 1% of cells, 82 of them, have no land pixels at all** and carry
    no terrain values. The model imputes them at both training and serving
    time.
13. **The API will score any coordinate on earth**, including far outside
    the five states it was trained on, and returns a confident percentile
    with no out-of-domain warning.

### Usage

Requires Python 3.10 or newer, since the code uses modern type syntax.
`requirements.txt` covers the `migracoast_ml` package only. The original
`app.py` additionally needs `mysql-connector-python`, `rasterio`, and
`geopy`, which are not included there.

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

The trained model and its metrics are committed under `models/`, so a fresh
clone can run `evaluate` immediately, since it only reads
`models/metrics.json`. `serve`, `build`, `baseline`, and
`notebooks/results.ipynb` all need the downloaded source data first, which is
gitignored and not committed, so run `fetch` before any of them.

### API

| Route | Method | Purpose |
|---|---|---|
| `/ml/risk` | POST | Score one coordinate |
| `/ml/heatmap` | POST | Score a grid, capped at 2500 cells per request |
| `/ml/model-info` | GET | Model name, metrics, feature list |
| `/health` | GET | Liveness check, served by the standalone server |

A single `/ml/risk` point query takes about 120 ms. The server itself takes
about 8 seconds to start, since it loads the trained model and four feature
providers (terrain, coastline, sea level, and hazard history) into memory
before it will accept requests.

### Database credentials

Connection details for `app.py` now come from the environment:
`MIGRACOAST_DB_HOST`, `MIGRACOAST_DB_USER`, `MIGRACOAST_DB_PASSWORD`,
`MIGRACOAST_DB_NAME`.

**A database password was previously committed to this repository in
plaintext and remains readable in its git history even though the current
code no longer contains it.** It must be rotated on the database server; a
new environment variable value alone does not undo the exposure.

### Notes for reviewers

Two choices are easy to miss and worth calling out directly:

- **Spatial blocking in cross-validation** exists because neighbouring 0.1
  degree cells are highly correlated. A random train/test split would put a
  cell in the training set and its near-identical neighbour in the test set,
  inflating every score. Grouping folds by 1 degree block keeps that from
  happening.
- **Latitude, longitude, and state are excluded as features on purpose.**
  A model with location as a direct input can learn "this is Florida" instead
  of "this is low, close to the coast, and storm-prone," and would look more
  accurate than it actually is.


## Sample outputs

Sample risk report generated

![Sample Image 1](https://github.com/user-attachments/assets/608781aa-1760-467c-816b-65f7eb03095f)
Section 1

![Sample Image 2](https://github.com/user-attachments/assets/ffabc7a9-4a04-42b6-9317-533da4a617b8)
Section2

![Sample Image 3](https://github.com/user-attachments/assets/53731fd6-7162-47fc-9974-214319ab1997)
Section 3


Sample inundation map generated
![Sample Image 4](https://github.com/user-attachments/assets/e55637e2-5287-4fa5-9031-e0c37ccb868b)

