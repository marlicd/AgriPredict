# AgriPredict

**Machine Learning-Based Crop Yield Prediction Using Environmental and Agricultural Data**

A complete data science project answering one question: *can we predict crop yield in Kenya from county, area, rainfall, temperature, and soil?* — built from raw government PDF reports through to a deployed, interactive prediction tool.

---

## Live App

Run locally with `streamlit run app.py`, or deploy to Streamlit Community Cloud (see [Deployment](#deployment) below).

---

## Table of Contents

- [Overview](#overview)
- [The Question](#the-question)
- [Data Sources](#data-sources)
- [Project Pipeline](#project-pipeline)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Credits](#credits)

---

## Overview

Kenyan farmers, county governments, and agricultural planners have no easy, data-driven way to estimate expected crop yield for a given county and season — even though the underlying data exists. It sits scattered across government PDF reports that nobody had previously extracted into a usable form.

This project does that work end to end:

1. Extracts real county-level crop data directly from KNBS PDF reports
2. Cleans and validates it
3. Explores it to understand real patterns (drought years, land classification, county variation)
4. Enriches it with real weather and soil data
5. Trains and rigorously compares five machine learning algorithms per crop
6. Deploys the results as an interactive Streamlit application — including where the honest answer turned out to be "we can't predict this reliably"

---

## The Question

> Can we predict crop yield using historical agricultural, environmental, and weather data for Kenyan counties?

**Prediction scenario, defined explicitly to avoid data leakage:** this project predicts **end-of-season yield** for a given county-crop-year using that season's own area planted, plus that season's weather and static soil characteristics. It never uses a crop's own production or yield as a feature to predict itself, and never uses future data to predict a past or current season.

---

## Data Sources

| Source | What it provides | Access |
|---|---|---|
| **KNBS National Agriculture Production Report 2024** | County-level area & production, 2019–2023 | Extracted from PDF |
| **KNBS National Agriculture Production Report 2025** | County-level area & production, 2020–2024 | Extracted from PDF |
| **NASA POWER** | Daily rainfall & temperature, aggregated to annual, per county centroid | Free API, no key required |
| **iSDAsoil** | Soil pH per county centroid | Free API, registration required |
| **National Drought Management Authority (NDMA)** | Official ASAL (Arid and Semi-Arid Lands) county classification | Public bulletins, cross-checked against real production data |

**Crops covered:** Maize, Beans, Sorghum, Green Grams, Irish Potatoes — the five crops with full area + production coverage across all six years. Tea and Coffee were investigated during data collection but excluded from modeling (Tea has production data only, no area; Coffee has area only, and just three crop-years of coverage). Rice and Sugarcane were investigated and dropped entirely — neither has a genuine county-level breakdown in the source reports.

**Coverage:** 2019–2024, up to 47 Kenyan counties (varies by crop — not every county grows every crop).

---

## Project Pipeline

### 1. Data Collection
County-level tables were extracted directly from the two KNBS PDF reports using `pdftotext -layout`, then parsed with a custom **county-name-anchored parser** — built specifically because a naive line-by-line parser misreads county names that wrap across two lines in the source PDF (e.g. "Elgeyo Marakwet"). Every extracted value was spot-checked against the source PDF text before being trusted.

### 2. Data Cleaning
- County name inconsistencies from merging two different extraction passes (e.g. `"Muranga"` → `"Murang'a"`) were identified by inspecting every distinct value and fixed before merging.
- Missing values were handled deliberately, not uniformly: a missing value for one of the five modeled crops means the county genuinely doesn't grow it, and was filled with `0`.
- Every row was validated against one specific impossibility — positive production with zero hectares planted. Zero such rows were found across all five crops.
- Yield (tonnes/hectare) — the actual prediction target — was engineered as `production_tons / area_ha`, with division-by-zero results cleaned to `NaN` rather than left as `inf`.

### 3. Exploratory Data Analysis
- **Land classification:** built from NDMA's official ASAL county list, then refined using actual production data — a handful of nominally "arid" counties showing genuinely minimal output were split into their own tier. Nearly 90% of Kenya's land area is Arid or Semi-Arid, though irrigation keeps parts of even the driest counties producing.
- **The 2022 drought:** yield trends revealed a sharp drop across multiple crops in 2022, independently confirmed against the real 2021–22 Horn of Africa drought (national maize production fell 23% year-on-year).
- **County variation:** maize shows dramatic county-level variation (a handful of highland counties dominate); other crops are more uniform.
- **Area vs. yield:** tested directly — for beans, sorghum, and green grams, planting more land does not meaningfully raise yield per hectare. Maize is the exception.

### 4. Feature Engineering
- **Weather (NASA POWER):** rainfall and temperature pulled per county centroid, aggregated to annual figures.
- **Soil (iSDAsoil):** soil pH per county centroid, static (doesn't vary by year).
- **Centroid caveat:** one county (Kisumu) geocoded to a point over Lake Victoria and was manually corrected to a coordinate in Kisumu town.
- Correlation testing confirmed real signal: maize's yield correlates with rainfall at roughly +0.57 — genuine evidence weather drives yield, not coincidence. Beans showed almost no correlation with any weather or soil feature — an early sign of a finding confirmed more rigorously during modeling.

### 5. Modeling
Five algorithms were tested per crop, not one:

1. **Linear Regression**, **Random Forest**, **XGBoost** — baseline round.
2. **Ridge**, **Lasso** — tested specifically after the baseline round revealed an overfitting problem (~50 features, mostly one-hot encoded counties, against as few as 28 validation rows per crop).

**Diagnosis:** Adjusted R² returned undefined for every single model and crop — direct evidence of the overfitting problem, not a guess.

**Fix, tested not assumed:** Ridge and Lasso improved results for Maize, Sorghum, and Irish Potatoes (real signal recovered), but made Beans worse and left Green Grams flat — ruling out overfitting as the full explanation for those two. Lasso's coefficients on Beans confirmed this directly: it kept exactly **one** feature out of ~50, zeroing out rainfall, temperature, soil pH, area, and every other county.

**Model selected: Ridge, trained separately per crop** — chosen because it directly targeted the diagnosed problem, improved results wherever real signal existed, and outperformed both tree-based models given how little data each crop has.

Full metric suite used: **MAE, MSE, RMSE, R², MAPE** (Adjusted R² is discussed but not usable — see above). Precision and recall were not used, as this is a regression problem, not classification.

### 6. Deployment
Results are served through a Streamlit application (`app.py`) covering the full project story across seven sections, plus an interactive predictor.

---

## Repository Structure

```
AgriPredict/
├── AgriPredict.ipynb              # Full notebook: collection → cleaning → EDA → features → modeling
├── app.py                         # Streamlit application (single file)
├── requirements.txt               # App dependencies
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
├── DATA/
│   ├── maize.csv, beans.csv, sorghum.csv, green_grams.csv, irish_potatoes.csv
│   ├── tea_production.csv, coffee_area.csv
│   ├── agripredict_wide.csv       # Final merged, cleaned, feature-engineered dataset
│   ├── county_centroids.csv       # Geocoded county coordinates
│   ├── county_land_classification_v2.csv
│   ├── weather_annual.csv         # NASA POWER pull
│   ├── soil.csv                   # iSDAsoil pull
│   └── ken_admin1.*                # Kenya county boundary shapefile (HDX/OCHA)
└── models/
    └── {crop}_ridge.joblib        # Trained Ridge model + scaler, per crop
```

---

## Setup

### Requirements
- Python 3.10+
- A free [iSDAsoil](https://www.isda-africa.com/isdasoil/) account (only needed to re-pull soil data)

### Install

```bash
git clone https://github.com/<your-username>/AgriPredict.git
cd AgriPredict
pip install -r requirements.txt
```

### Notebook environment
The notebook additionally needs `geopandas`, `geopy`, `xgboost`, `seaborn`, and `jupyter` — these are development dependencies, not required to run the deployed app:

```bash
pip install geopandas geopy xgboost seaborn jupyter
```

---

## Usage

### Run the notebook
```bash
jupyter notebook AgriPredict.ipynb
```
Runs the full pipeline from raw PDF extraction through model training. Requires the two source KNBS PDFs in the working directory for a from-scratch run, or the pre-built `DATA/agripredict_wide.csv` to skip straight to modeling.

### Run the app locally
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Requires `DATA/` and `models/` populated (see [Repository Structure](#repository-structure)).

### Predict a yield
In the app's **Predict a Yield** tab: select a county and crop (Maize, Sorghum, or Irish Potatoes — the three crops with a usable model), adjust season conditions if desired, and click Predict. The result includes the top features that drove that specific prediction, computed directly from the Ridge model's coefficients.

---

## Deployment

Deployed via [Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repository to GitHub (including `DATA/` and `models/`).
2. On share.streamlit.io, create a new app pointing at `app.py` on your `main` branch.
3. Streamlit Cloud installs `requirements.txt` automatically.

---

## Results

| Crop | Best Model | R² | Status |
|---|---|---|---|
| Maize | Ridge | 0.754 | Served live |
| Irish Potatoes | Lasso | 0.411 | Served live |
| Sorghum | Ridge | 0.095 | Served live, flagged as weaker |
| Green Grams | Random Forest | 0.031 | Excluded — no usable accuracy |
| Beans | — | -0.456 (best) | Excluded — confirmed no usable signal |

Full per-model, per-metric breakdown available in the app's **Model Performance** tab, and in `AgriPredict.ipynb`.

---

## Known Limitations

- **No fertilizer or irrigation data.** Identified as a gap at project scoping and never filled — no consistent county-level time series exists for either in Kenya's public data. Very likely the missing piece for Beans and Green Grams.
- **Ridge's regularization strength (`alpha=10.0`) was not tuned** via cross-validation — a fixed value was used throughout.
- **Centroid-based weather/soil** represents one point per county, not true spatial variation across a county's full area.
- **Tea and Coffee** have incomplete data (production-only and area-only respectively) and were excluded from modeling as a result.
- **Beans and Green Grams** are not reliably predictable with the current feature set — confirmed directly via Lasso's coefficients, not assumed.

---

## Future Work

- Acquire real fertilizer/irrigation data to test whether it resolves the Beans/Green Grams gap.
- Tune Ridge's `alpha` via cross-validated grid search.
- Test the 4-tier land classification (built in Feature Engineering) as a replacement for the ~46 raw county dummy columns, to reduce the feature-to-row ratio directly.
- Extend the county-year panel as more KNBS reports become available.

---

## Credits

**Data:** Kenya National Bureau of Statistics, NASA POWER, iSDAsoil, Kenya National Drought Management Authority, IEBC/OCHA (county boundaries).

**Built by:** Marlic Demetrius Chege