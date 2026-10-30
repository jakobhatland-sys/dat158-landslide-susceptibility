# Landslide Susceptibility – Voss / Granvin, Vestland

DAT158 Machine Learning – final project (HVL, autumn 2026)
Author: Jakob Hatland

A Streamlit web app that estimates **susceptibility to soil slides and debris flows
(jord- og flomskred)** at a user-selected location in the Voss/Granvin area, using a
machine-learning model trained on registered landslide events and terrain derivatives
from a 10 m digital terrain model.

The task is framed as **binary classification** (landslide susceptibility mapping):
cells where landslides have been registered (label 1) versus pseudo-absence cells
(label 0), described by terrain features. The model output is a **relative
susceptibility score** per 10 m cell.

---

## Study area

The study area is the extent of Kartverket DTM10 tile **6703_2** (approx. 50 × 50 km,
ETRS89 / UTM zone 32), which covers Voss, Granvin and Ulvik in inner Hardanger.

The area was limited deliberately:
- typical western-Norwegian fjord-valley topography with frequent soil slides and debris flows,
- enough registered events for training and spatial validation,
- manageable raster size (≈ 25 million cells) for processing on a laptop.

The model is **not** expected to generalise outside this area without retraining.

## Data

Raw data is not committed (size and licence). Download manually:

| Dataset | Source | Licence | Location in repo |
|---|---|---|---|
| Skredhendelser (landslide events) | NVE – ArcGIS REST service / https://nedlasting.nve.no/gis/ | NLOD 2.0 | `data/raw/skredhendelser_voss.geojson` |
| DTM 10 Terrengmodell, tile 6703_2 (UTM32) | Kartverket – https://hoydedata.no | CC BY 4.0 | `data/raw/dtm10/6703_2_10m_z32.tif` |

Landslide events are fetched for a bounding box (WGS84: 5.9–7.1 °E, 60.35–60.95 °N)
that covers the DTM tile, and filtered to soil slides and debris flows. Quick clay slides
are excluded, as they follow a different failure mechanism.

## Method

**Terrain features** (computed from the DTM at 10 m):

| Feature | Description |
|---|---|
| `elevation` | Terrain height (m) |
| `slope` | Slope angle (degrees) |
| `northness`, `eastness` | Cosine and sine of aspect (avoids the 0°/360° discontinuity) |
| `curvature` | Laplacian of elevation; positive = concave (flow convergence) |
| `roughness` | Local standard deviation of elevation, 3 × 3 window |
| `tpi` | Topographic position index: elevation minus mean in a 150 m window |

Cells below 1 m elevation (fjord and sea surface) are masked out, so that trivially
flat water cells do not inflate the scores.

**Samples**
- Positives: landslide events, snapped to the DTM grid (one sample per cell).
- Negatives: **pseudo-absence** points sampled randomly from valid cells at least
  200 m from any registered event, ratio 3:1 to positives.

**Models**
- Logistic regression (standardised features, baseline)
- Random forest (300 trees, `min_samples_leaf=5`)

Both use balanced class weights.

**Validation – spatial block cross-validation.**
Samples are grouped into 2 × 2 km blocks, and `GroupKFold` ensures that no block
appears in both training and test folds. A random-split CV is also reported for the
random forest to quantify the optimistic bias caused by spatial autocorrelation:
with a random split, test points lie next to training points and the score is inflated.

Metrics: ROC-AUC and PR-AUC (average precision). The model with the highest spatial
ROC-AUC is refitted on all data and used in the app.

## Results

<!-- Fill in from models/metrics.json after running pipeline.py -->

| Model | Spatial CV ROC-AUC | Spatial CV PR-AUC | Random CV ROC-AUC |
|---|---|---|---|
| Logistic regression | – | – | – |
| Random forest | – | – | – |

Training data: – events, – pseudo-absence points, – spatial blocks.

![Susceptibility map](report/figures/susceptibility_map.png)
![Feature importance](report/figures/feature_importance.png)

## Limitations

- **Pseudo-absence:** negative samples are not confirmed non-landslide locations; some
  may be susceptible terrain where no event has been registered yet. The score is
  therefore a relative susceptibility index, not a probability of occurrence, and its
  calibration depends on the class ratio.
- **Positional accuracy:** older events in the NVE database can have large location
  uncertainty, which adds label noise.
- **Terrain only:** no precipitation, soil type (løsmasser), vegetation or land use.
  These are major drivers of soil slides and debris flows.
- **Reporting bias:** events are registered more often near roads and settlements.
- **Not for decision-making:** this is a course project and not a substitute for NVE's
  official hazard and awareness maps (aktsomhetskart).

## Repository structure

```
.
├── app.py                   Streamlit web app
├── pipeline.py              Features, dataset, training, evaluation, maps
├── fetch_skred_rest.py      Fetch landslide events from NVE ArcGIS REST
├── explore_skred.py         Inspect and filter events (soil slides / debris flows)
├── data/
│   ├── raw/                 Source data (not in git)
│   └── processed/           Filtered events + dataset.csv (not in git)
├── models/                  model.joblib, metrics.json, features.tif, susceptibility.tif
├── report/figures/          Susceptibility map and feature importance
└── requirements.txt
```

## Reproduce

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt geopandas requests scipy matplotlib

python fetch_skred_rest.py      # download landslide events
python explore_skred.py         # filter to soil slides / debris flows
python pipeline.py              # features, training, evaluation, maps
streamlit run app.py
```

Run all commands from the repository root. The trained model and the rasters the app
needs are committed in `models/`, so the app can be run directly with
`pip install -r requirements.txt` and `streamlit run app.py`.

## Sources and AI use

- NVE – Skredhendelser (national landslide database).
- Kartverket – DTM 10 Terrengmodell, via hoydedata.no.

AI coding assistance (Claude, Anthropic) was used for code scaffolding and review.
All design decisions, data processing and results were verified by the author.
