# Landslide Susceptibility – Vestland, Norway

DAT158 Machine Learning – final project (HVL, autumn 2026)
Author: Jakob Hatland

A web app where the user clicks a location in Vestland county and gets an ML-based
estimate of **susceptibility to soil slides and debris flows (jord- og flomskred)**,
together with the terrain factors driving the prediction.

- **Labels:** registered landslide events from NVE's national landslide database (Skredhendelser).
- **Features:** terrain derivatives computed from Kartverket's DTM10 (slope, aspect, curvature,
  topographic wetness index, flow accumulation, roughness, elevation).
- **Model:** gradient-boosted trees vs. logistic-regression and random-forest baselines,
  evaluated with **spatial block cross-validation**.
- **App:** Streamlit + Folium map.

> Live app: _TBD_  
> Report: [`report/REPORT.md`](report/REPORT.md)

---

## Repository structure

```
.
├── app/                 Streamlit web app
├── data/
│   ├── raw/             Downloaded source data (not in git – see "Data")
│   └── processed/       Generated features / datasets (not in git)
├── models/              Trained model + metadata
├── notebooks/           EDA and analysis notebooks
├── report/              Report (Markdown) + figures
├── src/                 Pipeline scripts (run in numbered order)
├── environment.yml      Conda environment (full pipeline)
└── requirements.txt     Minimal dependencies for the deployed app
```

## Reproduce

### 1. Environment

```bash
conda env create -f environment.yml
conda activate dat158-skred
```

### 2. Data

Raw data is not committed (size + licence). Download manually:

| Dataset | Source | Licence | Put in |
|---|---|---|---|
| Skredhendelser (landslide events) | NVE – https://nedlasting.nve.no/gis/ | NLOD / CC BY 4.0 | `data/raw/skredhendelser/` |
| DTM 10 Terrengmodell (UTM33) | Kartverket via Geonorge | CC BY 4.0 | `data/raw/dtm10/` |

Exact download settings are documented in [`data/README.md`](data/README.md).

### 3. Pipeline

```bash
python src/01_prepare_events.py      # filter + clean landslide events
python src/02_terrain_features.py    # compute terrain derivative rasters
python src/03_build_dataset.py       # sample positives/negatives, extract features
python src/04_train.py               # train + evaluate models (spatial CV)
python src/05_export_app_assets.py   # export model + rasters used by the app
```

### 4. Run the app locally

```bash
streamlit run app/app.py
```

## Sources and AI use

See [`report/REPORT.md` – References](report/REPORT.md#references).
AI coding assistance (Claude, Anthropic) was used for code scaffolding and review;
all design decisions, data processing and results were verified by the author.
