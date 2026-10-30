# Landslide Susceptibility Mapping for Vestland with Machine Learning

DAT158 – Final project · Jakob Hatland · HVL, autumn 2026

> Status: draft – sections are filled in as the project progresses.

## 1. Problem framing and business objective

- **Problem:** Soil slides and debris flows (jord- og flomskred) threaten roads, buildings and people in
  Vestland. Event frequency is expected to increase with more intense precipitation.
- **Objective:** Give a quick, location-specific estimate of terrain-driven susceptibility for any point
  in Vestland, with an explanation of which terrain factors drive it.
- **Users:** planners, surveyors, students, the general public. Screening tool – **not** a replacement
  for NVE's official hazard maps (aktsomhetskart/faresonekart).
- **ML task:** binary classification (event / no event) on terrain features → calibrated probability.
- **Success metrics:** ROC-AUC and PR-AUC under spatial cross-validation; baseline = logistic regression
  on slope only.

## 2. Data

### 2.1 Sources
### 2.2 Data quality and filtering
### 2.3 Exploratory data analysis

## 3. Data preparation and feature engineering

### 3.1 Terrain derivatives
### 3.2 Negative sampling
### 3.3 Train/test split – spatial blocks

## 4. Modelling

### 4.1 Baselines
### 4.2 Candidate models
### 4.3 Hyperparameter tuning

## 5. Evaluation

### 5.1 Metrics
### 5.2 Model interpretation (SHAP)
### 5.3 Error analysis and limitations

## 6. Deployment

### 6.1 Web app
### 6.2 Hosting

## 7. Conclusion and further work

## References

1. NVE – Skredhendelser (national landslide database). https://nedlasting.nve.no/gis/
2. Kartverket – DTM 10 Terrengmodell (UTM33). https://kartkatalog.geonorge.no
3. _(add: literature on landslide susceptibility / TWI / spatial CV)_

### Use of AI tools

Claude (Anthropic) was used as a coding assistant for scaffolding scripts, debugging and
reviewing text. All outputs were reviewed, run and verified by the author.
