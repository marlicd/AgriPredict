"""
AgriPredict — Streamlit deployment
Run with: streamlit run app.py

Expects, in the same folder:
  DATA/agripredict_wide.csv          (built in the notebook's Feature Engineering section)
  DATA/county_land_classification_v2.csv
  models/{crop}_ridge.joblib         (dict with keys "model" and "scaler")
    for crop in ["maize", "sorghum", "irish_potatoes", "beans", "green_grams"]

If a model file for a crop is missing, that crop is shown in the predictor
as unavailable rather than crashing the app — see load_model() below.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Page config — set first, before anything else renders
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriPredict — Kenya Crop Yield",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Theme — grounded in the project's own land-classification palette, not a
# generic SaaS default. Parchment field-report background, forest-soil ink,
# maize gold as the single accent. Farmable/semi-arid/arid colors reuse the
# exact hexes from the Kenya map built earlier in the project, so the visual
# language stays consistent end to end rather than inventing a second palette.
# ----------------------------------------------------------------------------
CROP_LABELS = {
    "maize": "Maize",
    "sorghum": "Sorghum",
    "irish_potatoes": "Irish Potatoes",
    "beans": "Beans",
    "green_grams": "Green Grams",
}
RELIABLE_CROPS = ["maize", "sorghum", "irish_potatoes"]
LOW_CONFIDENCE_CROPS = ["beans", "green_grams"]

CLASS_COLORS = {
    "Non-ASAL (higher agricultural potential)": "#3F6B3F",
    "Semi-Arid (ASAL)": "#C97C3D",
    "Arid (ASAL core)": "#8B3A3A",
    "Minimal/No Crop Farming (confirmed by production data)": "#5C2A2A",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3, .display {
    font-family: 'Spectral', serif;
}

.stApp {
    background-color: #F0EBDD;
    color: #2B3A28;
}

/* Kill Streamlit's default top padding / chrome for a tighter, less templated feel */
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

/* Tabs — flat, underline style, not rounded pill chips */
.stTabs [data-baseweb="tab-list"] {
    gap: 28px;
    border-bottom: 1px solid #CFC6AE;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Spectral', serif;
    font-size: 1.05rem;
    color: #6B6350;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    color: #2B3A28 !important;
    border-bottom: 2px solid #C89B3C !important;
}

/* Buttons — one accent color, doing one job */
.stButton>button {
    background-color: #2B3A28;
    color: #F0EBDD;
    border: none;
    border-radius: 3px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    padding: 0.55rem 1.4rem;
}
.stButton>button:hover {
    background-color: #C89B3C;
    color: #2B3A28;
}

/* A quiet divider that reads as a field-report rule, not a decorative line */
hr {
    border: none;
    border-top: 1px solid #CFC6AE;
    margin: 1.6rem 0;
}

.stat-line {
    font-size: 0.95rem;
    color: #6B6350;
    border-top: 1px solid #CFC6AE;
    border-bottom: 1px solid #CFC6AE;
    padding: 0.6rem 0;
    margin: 1.2rem 0 1.8rem 0;
}

.crop-flag {
    display: inline-block;
    font-size: 0.8rem;
    padding: 2px 9px;
    border-radius: 3px;
    margin-left: 8px;
    font-family: 'IBM Plex Sans', sans-serif;
}
.flag-reliable { background-color: #3F6B3F; color: #F0EBDD; }
.flag-caution { background-color: #C97C3D; color: #F0EBDD; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data & model loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_wide():
    path = "DATA/agripredict_wide.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_classification():
    path = "DATA/county_land_classification_v2.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model(crop):
    path = f"models/{crop}_ridge.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


wide = load_wide()
classification = load_classification()


# ----------------------------------------------------------------------------
# Header — the research question stated plainly, not a decorative hero
# ----------------------------------------------------------------------------
st.markdown("<h1 class='display' style='margin-bottom:0;'>AgriPredict</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='display' style='font-size:1.3rem; color:#6B6350; margin-top:4px;'>"
    "Can we predict crop yield in Kenya from county, area, rainfall, temperature, and soil?"
    "</p>", unsafe_allow_html=True
)

if wide is not None:
    n_counties = wide["county"].nunique()
    n_years = f"{int(wide['year'].min())}–{int(wide['year'].max())}"
    st.markdown(
        f"<div class='stat-line'>{n_counties} counties &nbsp;·&nbsp; {n_years} &nbsp;·&nbsp; "
        f"7 crops tracked, 5 modeled &nbsp;·&nbsp; source: KNBS National Agriculture Production Reports</div>",
        unsafe_allow_html=True
    )

tab1, tab2, tab3, tab4 = st.tabs(["The Question", "What the Data Shows", "Predict a Yield", "How This Was Built"])


# ----------------------------------------------------------------------------
# TAB 1 — The Question
# ----------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
Kenyan farmers, county governments, and agricultural planners have no easy,
data-driven way to estimate what yield to expect from a given crop in a given
county, under a given season's conditions — even though the underlying data
to answer this exists. It sits scattered across government PDF reports that
nobody has ever stitched together into something usable.

This project does that work: county-level tables were extracted directly
from two Kenya National Bureau of Statistics reports, joined with real
rainfall and temperature records and soil characteristics per county, and
used to train a yield model for each crop — tested honestly, not just once.
""")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
**Three questions this project set out to answer, in order:**

1. Does the land itself explain the gap — is Kenya mostly farmable, or mostly arid?
2. Does planting more land produce a higher yield per hectare, or does it not matter?
3. Can yield actually be predicted — and for which crops is that claim honest?
""")

    with col2:
        if classification is not None:
            counts = classification["land_classification_refined"].value_counts()
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            colors = [CLASS_COLORS.get(k, "#999") for k in counts.index]
            ax.pie(counts.values, colors=colors, startangle=90,
                   wedgeprops={"edgecolor": "#F0EBDD", "linewidth": 2})
            ax.set_title("Kenya's 47 counties, by land type", fontsize=10, fontfamily="serif", color="#2B3A28")
            fig.patch.set_facecolor("#F0EBDD")
            st.pyplot(fig)
            st.caption(
                "Full breakdown, sourced from the National Drought Management "
                "Authority's official ASAL classification, is in the next tab."
            )


# ----------------------------------------------------------------------------
# TAB 2 — What the Data Shows
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("<h3 class='display'>Does more land mean more land?</h3>", unsafe_allow_html=True)
    st.markdown(
        "Nearly 90% of Kenya's land area is Arid or Semi-Arid — but 'arid' does not mean "
        "'no farming.' Irrigation along the Tana, Turkwel, and Kerio rivers keeps parts of "
        "even the driest counties producing. Four counties — Garissa, Isiolo, Marsabit, and "
        "Wajir — show genuinely minimal crop production, confirmed directly from KNBS "
        "production figures rather than assumed from climate alone."
    )

    if classification is not None:
        counts = classification["land_classification_refined"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 1.6))
        left = 0
        for label, val in counts.items():
            ax.barh(0, val, left=left, color=CLASS_COLORS.get(label, "#999"), height=0.6)
            left += val
        ax.set_xlim(0, counts.sum())
        ax.axis("off")
        fig.patch.set_facecolor("#F0EBDD")
        ax.set_facecolor("#F0EBDD")
        st.pyplot(fig)
        legend_cols = st.columns(len(counts))
        for i, (label, val) in enumerate(counts.items()):
            with legend_cols[i]:
                st.markdown(
                    f"<span style='color:{CLASS_COLORS.get(label,'#999')}'>■</span> "
                    f"{label.split(' (')[0]} ({val})",
                    unsafe_allow_html=True
                )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>Does planting more land raise yield?</h3>", unsafe_allow_html=True)
    st.markdown(
        "Tested directly, per crop, by binning counties into five area ranges and comparing "
        "average yield per bin: for beans, sorghum, and green grams, the answer is essentially "
        "no — a county farming 80,000 hectares of beans yields barely more per hectare than one "
        "farming 5,000. Maize is the exception, where the largest-area counties do show "
        "meaningfully higher yield — most likely a marker of better-resourced, larger-scale "
        "operations rather than area itself causing the difference."
    )

    if wide is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3 class='display'>What happened in 2022?</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            for crop, color in [("maize", "#C89B3C"), ("sorghum", "#8B3A3A"), ("green_grams", "#3F6B3F")]:
                col = f"{crop}_yield_t_per_ha"
                if col in wide.columns:
                    wide.groupby("year")[col].mean().plot(ax=ax, marker="o", label=CROP_LABELS[crop], color=color)
            ax.set_ylabel("Average yield (t/ha)")
            ax.set_xlabel("")
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            fig.patch.set_facecolor("#F0EBDD")
            ax.set_facecolor("#F0EBDD")
            st.pyplot(fig)
        with c2:
            st.markdown(
                "The 2021–22 short rains failed — the third consecutive below-average season "
                "across eastern and northern Kenya. National maize production dropped 23% "
                "year-on-year. Sorghum and green grams, concentrated in drier counties, fell "
                "even harder that year — a pattern confirmed independently in this project's "
                "correlation analysis, not just visible in the chart."
            )


# ----------------------------------------------------------------------------
# TAB 3 — Predict a Yield
# ----------------------------------------------------------------------------
with tab3:
    if wide is None:
        st.warning("DATA/agripredict_wide.csv not found — place it alongside this app to enable prediction.")
    else:
        st.markdown(
            "Maize, sorghum, and Irish potatoes have models that reach meaningful accuracy "
            "(R² of 0.10 to 0.75, confirmed across five algorithms). Beans and green grams "
            "are shown too, but flagged — no algorithm tested reached usable accuracy for "
            "either, and that has a concrete cause rather than a mysterious one (see the last tab)."
        )
        st.markdown("<hr>", unsafe_allow_html=True)

        step1, step2 = st.columns(2)
        with step1:
            st.markdown("**1. County**")
            county = st.selectbox("", sorted(wide["county"].dropna().unique()), label_visibility="collapsed")
        with step2:
            st.markdown("**2. Crop**")
            crop_display = st.selectbox(
                "",
                [f"{CROP_LABELS[c]}" + (" — low confidence" if c in LOW_CONFIDENCE_CROPS else "")
                 for c in CROP_LABELS],
                label_visibility="collapsed"
            )
            crop = [c for c, label in CROP_LABELS.items() if crop_display.startswith(label)][0]

        county_rows = wide[wide["county"] == county]
        default_rain = county_rows["rainfall_mm"].mean() if "rainfall_mm" in wide.columns else 900.0
        default_temp = county_rows["avg_temp_c"].mean() if "avg_temp_c" in wide.columns else 21.0
        default_soil = county_rows["soil_ph"].mean() if "soil_ph" in wide.columns else 6.2
        default_area = county_rows[f"{crop}_area_ha"].mean() if f"{crop}_area_ha" in wide.columns else 1000.0
        default_area = 0 if pd.isna(default_area) else default_area

        st.markdown("**3. Season conditions**  ·  pre-filled with this county's historical average — adjust to test a scenario")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            area = st.number_input("Area planted (ha)", min_value=0.0, value=float(round(default_area)), step=10.0)
        with c2:
            rainfall = st.number_input("Rainfall (mm/yr)", min_value=0.0, value=float(round(default_rain)), step=10.0)
        with c3:
            temp = st.number_input("Avg. temperature (°C)", min_value=0.0, value=float(round(default_temp, 1)), step=0.5)
        with c4:
            soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=float(round(default_soil, 1)), step=0.1)

        st.markdown("**4. Predict**")
        if st.button("Predict yield"):
            bundle = load_model(crop)
            if bundle is None:
                st.error(f"No trained model found at models/{crop}_ridge.joblib — train and save it first.")
            else:
                model, scaler = bundle["model"], bundle["scaler"]
                feature_names = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None

                row = pd.DataFrame([{
                    f"{crop}_area_ha": area,
                    "rainfall_mm": rainfall,
                    "avg_temp_c": temp,
                    "soil_ph": soil_ph,
                }])
                county_col = f"county_{county}"
                if feature_names is not None:
                    for col in feature_names:
                        if col not in row.columns:
                            row[col] = 1 if col == county_col else 0
                    row = row[feature_names]

                scaled = scaler.transform(row)
                pred = model.predict(scaled)[0]

                st.markdown("<hr>", unsafe_allow_html=True)
                r1, r2 = st.columns([1, 2])
                with r1:
                    st.markdown(
                        f"<div style='font-family:Spectral,serif; font-size:2.6rem; color:#2B3A28;'>"
                        f"{pred:.2f} <span style='font-size:1.1rem; color:#6B6350;'>t/ha</span></div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"Predicted {CROP_LABELS[crop]} yield, {county}")
                    if crop in LOW_CONFIDENCE_CROPS:
                        st.markdown(
                            "<span class='crop-flag flag-caution'>Low confidence — treat as indicative only</span>",
                            unsafe_allow_html=True
                        )
                with r2:
                    st.markdown("**What drove this number**")
                    coefs = pd.Series(model.coef_, index=feature_names) if feature_names is not None else None
                    if coefs is not None:
                        contributions = coefs * scaled[0]
                        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(5)
                        for feat, val in top.items():
                            label = feat.replace("county_", "").replace("_", " ")
                            direction = "pushed the prediction up" if val > 0 else "pulled the prediction down"
                            st.markdown(f"- **{label}** {direction}")


# ----------------------------------------------------------------------------
# TAB 4 — How This Was Built
# ----------------------------------------------------------------------------
with tab4:
    st.markdown("""
Five algorithms were tested, not one: Linear Regression, Random Forest, and
XGBoost first, then Ridge and Lasso specifically to address an overfitting
problem the first round revealed — roughly 50 features against as few as 28
validation rows per crop.
""")

    results = pd.DataFrame([
        {"Crop": "Maize", "Best model": "Ridge", "R²": 0.754, "Status": "Reliable"},
        {"Crop": "Irish Potatoes", "Best model": "Lasso", "R²": 0.411, "Status": "Reliable"},
        {"Crop": "Sorghum", "Best model": "Ridge", "R²": 0.095, "Status": "Weak, usable"},
        {"Crop": "Green Grams", "Best model": "—", "R²": -0.032, "Status": "No usable signal"},
        {"Crop": "Beans", "Best model": "—", "R²": -0.456, "Status": "No usable signal"},
    ])
    st.dataframe(results, hide_index=True, use_container_width=True)

    st.markdown("""
**Why Ridge was selected as the deployed model:** it directly targets the
overfitting diagnosed in testing, it improved results wherever real signal
existed (maize, sorghum, Irish potatoes all gained R² over plain Linear
Regression), and it outperformed both tree-based models — Random Forest and
XGBoost overfit harder given how few rows each crop has to train on.

**Why beans and green grams are flagged, not hidden:** tested directly, not
assumed. Lasso — which can shrink a feature's weight to exactly zero — kept
exactly one feature out of roughly 50 for beans: a single county effect.
Rainfall, temperature, soil pH, and every other county were zeroed out
entirely. That is direct evidence the current features do not explain these
two crops' yield, most likely because they are grown in far fewer counties
than maize, leaving too little data to learn from.

**What would improve this next:** real fertilizer and irrigation data — a
gap identified at the very start of this project and never filled, since no
consistent county-level time series exists for either in Kenya's public
data. Tuning Ridge's penalty strength via cross-validation, rather than the
fixed value used here, is the other concrete next step.
""")

    st.caption(
        "Data: KNBS National Agriculture Production Reports (2024, 2025) · "
        "NASA POWER (weather) · iSDAsoil (soil) · National Drought Management "
        "Authority (land classification)"
    )
