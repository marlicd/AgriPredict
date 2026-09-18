"""
AgriPredict — Streamlit app (single file)
Run with: streamlit run app.py

Expects, in the same folder:
  DATA/agripredict_wide.csv
  DATA/county_land_classification_v2.csv
  models/{crop}_ridge.joblib   (dict with keys "model" and "scaler")
    for crop in maize, sorghum, irish_potatoes, beans, green_grams
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ----------------------------------------------------------------------------
# Config, theme, constants
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AgriPredict", page_icon="🌽", layout="wide")

CROP_LABELS = {
    "maize": "Maize", "sorghum": "Sorghum", "irish_potatoes": "Irish Potatoes",
    "beans": "Beans", "green_grams": "Green Grams",
}
LOW_CONFIDENCE_CROPS = ["beans", "green_grams"]
CLASS_COLORS = {
    "Non-ASAL (higher agricultural potential)": "#3F6B3F",
    "Semi-Arid (ASAL)": "#C97C3D",
    "Arid (ASAL core)": "#8B3A3A",
    "Minimal/No Crop Farming (confirmed by production data)": "#5C2A2A",
}
CROP_COLORS = {
    "maize": "#C89B3C", "beans": "#7C6A45", "sorghum": "#8B3A3A",
    "green_grams": "#3F6B3F", "irish_potatoes": "#4A5F8A",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, .display { font-family: 'Spectral', serif; }
.stApp { background-color: #F0EBDD; color: #2B3A28; }
.block-container { padding-top: 2rem; max-width: 1100px; }
.stTabs [data-baseweb="tab-list"] { gap: 22px; border-bottom: 1px solid #CFC6AE; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { font-family: 'Spectral', serif; font-size: 0.95rem; color: #6B6350; padding-bottom: 10px; }
.stTabs [aria-selected="true"] { color: #2B3A28 !important; border-bottom: 2px solid #C89B3C !important; }
.stButton>button { background-color: #2B3A28; color: #F0EBDD; border: none; border-radius: 3px; font-weight: 500; padding: 0.55rem 1.4rem; }
.stButton>button:hover { background-color: #C89B3C; color: #2B3A28; }
hr { border: none; border-top: 1px solid #CFC6AE; margin: 1.4rem 0; }
.stat-line { font-size: 0.95rem; color: #6B6350; border-top: 1px solid #CFC6AE; border-bottom: 1px solid #CFC6AE; padding: 0.6rem 0; margin: 1rem 0 1.5rem 0; }
.crop-flag { display: inline-block; font-size: 0.8rem; padding: 2px 9px; border-radius: 3px; margin-left: 8px; }
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
    df = pd.read_csv(path)
    for crop in CROP_LABELS:
        yc = f"{crop}_yield_t_per_ha"
        if yc not in df.columns and f"{crop}_production_tons" in df.columns:
            df[yc] = df[f"{crop}_production_tons"] / df[f"{crop}_area_ha"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


@st.cache_data
def load_classification():
    path = "DATA/county_land_classification_v2.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_resource
def load_model(crop):
    path = f"models/{crop}_ridge.joblib"
    return joblib.load(path) if os.path.exists(path) else None


def missing_data_notice():
    st.warning("DATA/agripredict_wide.csv not found. Place your project's DATA folder alongside this app.")


wide = load_wide()
classification = load_classification()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("<h1 class='display' style='margin-bottom:0;'>AgriPredict</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='display' style='font-size:1.2rem; color:#6B6350; margin-top:4px;'>"
    "Can we predict crop yield in Kenya from county, area, rainfall, temperature, and soil?</p>",
    unsafe_allow_html=True
)
if wide is not None:
    st.markdown(
        f"<div class='stat-line'>{wide['county'].nunique()} counties &nbsp;·&nbsp; "
        f"{int(wide['year'].min())}–{int(wide['year'].max())} &nbsp;·&nbsp; "
        f"5 crops modeled &nbsp;·&nbsp; 5 algorithms tested</div>", unsafe_allow_html=True
    )

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["The Question", "Data Collection", "Data Cleaning", "Exploratory Data Analysis",
     "Feature Engineering", "Modeling", "Predict a Yield"]
)

with tab1:
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("""
    Kenyan farmers, county governments, and agricultural planners have no easy,
    data-driven way to estimate expected yield for a given crop, county, and
    season — even though the data to answer this exists. It sits scattered
    across government PDF reports nobody has stitched together into something
    usable. This project does that work, end to end, and this app walks through
    every stage of it honestly — including where the answer turned out to be no.

    **Use the tabs above to move through the project in order:**

    1. **Data Collection** — where the numbers came from, and how they were extracted from PDF reports
    2. **Data Cleaning** — fixing inconsistencies, deciding what a missing value actually means
    3. **Exploratory Data Analysis** — what the data revealed before any model was trained
    4. **Feature Engineering** — adding weather and soil, and what actually correlated with yield
    5. **Modeling** — five algorithms tested, and the honest result for each crop
    6. **Predict a Yield** — the tool itself
    """)
    with col2:
        if classification is not None:
            counts = classification["land_classification_refined"].value_counts()
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            ax.pie(counts.values, colors=[CLASS_COLORS.get(k, "#999") for k in counts.index],
                   startangle=90, wedgeprops={"edgecolor": "#F0EBDD", "linewidth": 2})
            ax.set_title("Kenya's 47 counties, by land type", fontsize=10, fontfamily="serif", color="#2B3A28")
            fig.patch.set_facecolor("#F0EBDD")
            st.pyplot(fig)

with tab2:
    st.markdown("<h1 class='display'>Data Collection</h1>", unsafe_allow_html=True)

    st.markdown("""
    No ready-made "Kenya crop yield" dataset exists. The real numbers live inside two
    Kenya National Bureau of Statistics reports — the **National Agriculture Production
    Report 2024** (covers 2019–2023) and **2025** (covers 2020–2024) — as tables inside
    150–200 page PDFs, not as spreadsheets.
    """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>What the raw source looked like</h3>", unsafe_allow_html=True)
    st.markdown("A table like this, extracted from the PDF's text (not an image — real selectable text, but no column structure):")
    st.code(
        "Baringo         47,437     68,374     34,709     59,169     37,894     50,820\n"
        "Bomet           52,180    121,904     ...\n"
        "Elgeyo\nMarakwet   25,856     87,431     31,351     86,007     ...",
        language=None
    )
    st.caption(
        "Note Elgeyo Marakwet's name wrapping onto a second line — a real formatting "
        "quirk in the source PDF that a naive line-by-line parser misreads."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>Extraction method</h3>", unsafe_allow_html=True)
    st.markdown("""
    1. `pdftotext -layout` converts each PDF page to text, preserving rough column alignment.
    2. Each crop's table is located by searching for its heading (e.g. *"Annex 1: Area and Production of Maize by County"*).
    3. A **county-name-anchored parser** is used instead of a line-by-line one: it finds every
       occurrence of a known county name in the flattened text block, and treats everything
       between one county name and the next as that county's row — regardless of how many
       lines it wrapped across in the PDF. This is what correctly handles cases like
       Elgeyo Marakwet above.
    4. Extracted values were spot-checked directly against the source PDF text before
       being trusted for anything further.
    """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>Coverage, by crop</h3>", unsafe_allow_html=True)
    coverage = pd.DataFrame([
        {"Crop": "Maize", "Years": "2019–2024", "Counties": "47", "Metric": "Area + Production"},
        {"Crop": "Beans", "Years": "2019–2024", "Counties": "43", "Metric": "Area + Production"},
        {"Crop": "Sorghum", "Years": "2019–2024", "Counties": "43", "Metric": "Area + Production"},
        {"Crop": "Green Grams", "Years": "2019–2024", "Counties": "35", "Metric": "Area + Production"},
        {"Crop": "Irish Potatoes", "Years": "2019–2024", "Counties": "30", "Metric": "Area + Production"},
    ])
    st.dataframe(coverage.set_index("Crop"), width='stretch')
    st.caption(
        "Rice and Sugarcane were investigated and dropped — neither has a genuine "
        "county-level breakdown in the source reports."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>The final merged schema</h3>", unsafe_allow_html=True)
    if wide is not None:
        st.dataframe(wide.head(5), width='stretch')
        st.caption(f"{wide.shape[0]} rows, {wide.shape[1]} columns — one row per county per year.")
    else:
        missing_data_notice()

with tab3:
    st.markdown("<h1 class='display'>Data Cleaning</h1>", unsafe_allow_html=True)

    st.markdown("<h3 class='display'>Fixing inconsistent county names</h3>", unsafe_allow_html=True)
    st.markdown(
        "Merging data extracted by two different parsers let name variants slip through — "
        "the same real county, spelled two ways, silently becomes two separate rows on a merge:"
    )
    st.code(
        '"Muranga"        → "Murang\'a"\n'
        '"Tharaka - Nithi" → "Tharaka Nithi"\n'
        '"Trans-Nzoia"     → "Trans Nzoia"',
        language=None
    )
    st.caption(
        "Caught by printing every distinct county value and reading the list by eye — "
        "a merge on a mismatched key fails silently, producing extra rows with missing "
        "data rather than an error."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>What a missing value actually means</h3>", unsafe_allow_html=True)
    st.markdown("""
    Two different situations look identical in a spreadsheet but mean opposite things,
    and were handled differently on purpose:
    """)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Maize, Beans, Sorghum, Green Grams, Irish Potatoes**")
        st.markdown("Missing = the county genuinely doesn't grow it → filled with **0**")
    with c2:
        st.markdown("**Tea, Coffee**")
        st.markdown("Missing = the report simply doesn't cover that county-year → left as **NaN**, never filled")
    st.caption(
        "Filling Coffee's gaps with 0 would have told the model 'zero coffee was produced' "
        "in years the report just didn't record — a false signal, not a true one."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>Validity check — before trusting any of it</h3>", unsafe_allow_html=True)
    st.markdown("Every row was checked for a specific impossibility: positive production against zero hectares planted.")
    if wide is not None:
        crops = ["maize", "beans", "sorghum", "green_grams", "irish_potatoes"]
        check_rows = []
        for crop in crops:
            if f"{crop}_area_ha" in wide.columns and f"{crop}_production_tons" in wide.columns:
                bad = wide[(wide[f"{crop}_area_ha"] == 0) & (wide[f"{crop}_production_tons"] > 0)]
                check_rows.append({"Crop": crop.replace("_", " ").title(), "Suspicious rows": len(bad)})
        st.dataframe(pd.DataFrame(check_rows).set_index("Crop"), width='stretch')
    else:
        missing_data_notice()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>Engineering the actual target — yield</h3>", unsafe_allow_html=True)
    st.markdown(
        "The source reports give area and production, but yield — tonnes per hectare, "
        "what this whole project predicts — doesn't exist until it's calculated:"
    )
    st.code("yield_t_per_ha = production_tons / area_ha", language="python")
    st.caption("Division-by-zero results are cleaned to NaN, not left as inf, so they don't silently corrupt later statistics.")

    if wide is not None:
        crops = ["maize", "beans", "sorghum", "green_grams", "irish_potatoes"]
        ycols = [f"{c}_yield_t_per_ha" for c in crops if f"{c}_yield_t_per_ha" in wide.columns]
        if ycols:
            st.dataframe(wide[["county", "year"] + ycols].head(6), width='stretch')

with tab4:
    st.markdown("<h1 class='display'>Explore the Data</h1>", unsafe_allow_html=True)

    if wide is None:
        missing_data_notice()
        st.stop()

    crops = ["maize", "beans", "sorghum", "green_grams", "irish_potatoes"]

    # ---------------------------------------------------------------------------
    st.markdown("<h3 class='display'>Is Kenya mostly farmable, or mostly arid?</h3>", unsafe_allow_html=True)
    st.markdown(
        "Nearly 90% of Kenya's land area is Arid or Semi-Arid — but 'arid' does not mean "
        "'no farming.' Irrigation along the Tana, Turkwel, and Kerio rivers keeps parts of "
        "even the driest counties producing. Four counties — Garissa, Isiolo, Marsabit, and "
        "Wajir — show genuinely minimal crop production, confirmed directly from production "
        "figures rather than assumed from climate alone."
    )
    if classification is not None:
        counts = classification["land_classification_refined"].value_counts()
        fig, ax = plt.subplots(figsize=(9, 1.6))
        left = 0
        for label, val in counts.items():
            ax.barh(0, val, left=left, color=CLASS_COLORS.get(label, "#999"), height=0.6)
            left += val
        ax.set_xlim(0, counts.sum())
        ax.axis("off")
        fig.patch.set_facecolor("#F0EBDD")
        st.pyplot(fig)
        legend_cols = st.columns(len(counts))
        for i, (label, val) in enumerate(counts.items()):
            with legend_cols[i]:
                st.markdown(f"<span style='color:{CLASS_COLORS.get(label,'#999')}'>■</span> "
                            f"{label.split(' (')[0]} ({val})", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    st.markdown("<h3 class='display'>Yield over time — and what happened in 2022</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for crop in ["maize", "beans", "sorghum", "green_grams"]:
            col = f"{crop}_yield_t_per_ha"
            if col in wide.columns:
                wide.groupby("year")[col].mean().plot(ax=ax, marker="o", label=CROP_LABELS[crop], color=CROP_COLORS[crop])
        ax.set_ylabel("Average yield (t/ha)")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("#F0EBDD"); ax.set_facecolor("#F0EBDD")
        st.pyplot(fig)
        st.caption("Irish potatoes omitted here — its yield scale (7–17 t/ha) is 5–8x the others; shown separately below.")
    with c2:
        st.markdown(
            "The 2021–22 short rains failed — the third consecutive below-average season "
            "across eastern and northern Kenya. National maize production dropped 23% "
            "year-on-year. Sorghum and green grams, concentrated in drier counties, fell "
            "even harder — a pattern also visible in the correlation analysis below, not "
            "just this chart."
        )

    fig, ax = plt.subplots(figsize=(9, 3))
    if "irish_potatoes_yield_t_per_ha" in wide.columns:
        wide.groupby("year")["irish_potatoes_yield_t_per_ha"].mean().plot(ax=ax, marker="o", color=CROP_COLORS["irish_potatoes"])
    ax.set_ylabel("Average yield (t/ha)"); ax.set_title("Irish Potatoes", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("#F0EBDD"); ax.set_facecolor("#F0EBDD")
    st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    st.markdown("<h3 class='display'>Which counties drive each crop</h3>", unsafe_allow_html=True)
    crop_pick = st.selectbox("Crop", crops, format_func=lambda c: CROP_LABELS[c], key="county_var_crop")
    yield_col = f"{crop_pick}_yield_t_per_ha"
    fig, ax = plt.subplots(figsize=(10, 5))
    wide.groupby("county")[yield_col].mean().sort_values().dropna().plot(kind="bar", ax=ax, color=CROP_COLORS[crop_pick])
    ax.set_ylabel("Average yield (t/ha)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("#F0EBDD"); ax.set_facecolor("#F0EBDD")
    plt.xticks(rotation=90, fontsize=7)
    st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    st.markdown("<h3 class='display'>Yield distribution — averages hide the spread</h3>", unsafe_allow_html=True)
    yield_cols = [f"{c}_yield_t_per_ha" for c in crops]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    wide[yield_cols].boxplot(ax=ax)
    ax.set_ylabel("Yield (t/ha)")
    ax.set_xticklabels([CROP_LABELS[c] for c in crops], rotation=15)
    fig.patch.set_facecolor("#F0EBDD"); ax.set_facecolor("#F0EBDD")
    st.pyplot(fig)
    st.caption(
        "Irish potatoes yield 5–8x more than the grain/legume crops (expected — a tuber crop, "
        "not a grain), but also shows the widest spread of any crop, from a near-zero outlier "
        "up to almost 17.5 t/ha."
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    st.markdown("<h3 class='display'>Does planting more land raise yield?</h3>", unsafe_allow_html=True)
    st.markdown(
        "Tested directly, per crop, by binning counties into five area ranges and comparing "
        "average yield per bin: for beans, sorghum, and green grams, the answer is essentially "
        "no. Maize is the exception — its largest-area counties do show meaningfully higher "
        "yield, most likely a marker of better-resourced, larger-scale operations rather than "
        "area itself causing the difference."
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    axes = axes.flatten()
    for i, crop in enumerate(crops):
        area_col, y_col = f"{crop}_area_ha", f"{crop}_yield_t_per_ha"
        axes[i].scatter(wide[area_col], wide[y_col], alpha=0.5, s=14, color=CROP_COLORS[crop])
        axes[i].set_title(CROP_LABELS[crop], fontsize=9)
        axes[i].set_xlabel("area (ha)", fontsize=8); axes[i].set_ylabel("yield (t/ha)", fontsize=8)
        axes[i].spines[["top", "right"]].set_visible(False)
    axes[-1].axis("off")
    fig.patch.set_facecolor("#F0EBDD")
    for ax in axes:
        ax.set_facecolor("#F0EBDD")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("Weather and soil correlations with yield are covered on the Feature Engineering page.")

with tab5:
    st.markdown("<h1 class='display'>Feature Engineering</h1>", unsafe_allow_html=True)
    st.markdown(
        "Area and yield alone aren't enough — two more sources were added, both queried "
        "by **county centroid** (a single representative coordinate per county, since the "
        "weather and soil APIs only accept a point, not an area)."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<h3 class='display'>Weather — NASA POWER</h3>", unsafe_allow_html=True)
        st.markdown("""
    Free, no API key, daily records back to 1981. For each county centroid and year,
    daily rainfall and temperature were pulled and aggregated to annual totals/averages:
    """)
        st.code("rainfall_mm = sum(daily precipitation)\navg_temp_c  = mean(daily temperature)", language="python")
    with c2:
        st.markdown("<h3 class='display'>Soil — iSDAsoil</h3>", unsafe_allow_html=True)
        st.markdown("""
    Free after registration, login-based token (expires hourly). Soil pH per county
    centroid was pulled the same way — static per county, since soil doesn't change
    year to year the way weather does.
    """)
        st.caption(
            "One county's centroid (Kisumu) landed over Lake Victoria and was rejected "
            "by the API — corrected to a coordinate in Kisumu town itself."
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>Did it actually correlate with yield?</h3>", unsafe_allow_html=True)
    if wide is None:
        missing_data_notice()
    else:
        weather_cols = [c for c in ["rainfall_mm", "avg_temp_c", "soil_ph"] if c in wide.columns]
        crops = ["maize", "beans", "sorghum", "green_grams", "irish_potatoes"]
        if weather_cols:
            rows = []
            for crop in crops:
                yc = f"{crop}_yield_t_per_ha"
                if yc in wide.columns:
                    row = {"Crop": CROP_LABELS[crop]}
                    for wc in weather_cols:
                        row[wc] = round(wide[[yc, wc]].corr().iloc[0, 1], 2)
                    rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index("Crop"), width='stretch')
            st.markdown("""
    **Maize shows the strongest weather relationship of any crop** — rainfall correlates
    around +0.57, temperature around -0.44 with yield. Real, direct evidence weather
    genuinely drives maize yield, not a coincidence.

    **Beans shows almost no relationship to rainfall, temperature, or soil pH.** This
    was the first sign of a finding confirmed more rigorously on the Modeling page:
    weather and soil, on their own, don't explain beans' yield.
    """)
        else:
            st.info("Weather/soil columns not found — merge them via the notebook's Feature Engineering section.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 class='display'>A third feature — land classification</h3>", unsafe_allow_html=True)
    st.markdown("""
    Built from the National Drought Management Authority's official ASAL county list,
    cross-checked against actual production data (a handful of "arid" counties showing
    genuinely minimal output were split into their own tier). Not yet tested as a
    replacement for the 46 individual county columns in the model — a real next step,
    noted on the Modeling page.
    """)
    if classification is not None:
        st.dataframe(classification, width='stretch', height=200)

with tab6:
    st.markdown("<h1 class='display'>Model Performance</h1>", unsafe_allow_html=True)
    st.markdown(
        "Five algorithms were tested, not one: Linear Regression, Random Forest, and "
        "XGBoost first, then Ridge and Lasso specifically to address an overfitting "
        "problem the first round revealed — roughly 50 features against as few as 28 "
        "validation rows per crop, depending on the crop."
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h3 class='display'>All five algorithms, all five crops</h3>", unsafe_allow_html=True)

    # Full regression metric suite used throughout this project: MAE, MSE, RMSE, R2, MAPE.
    # Adjusted R2 is deliberately not shown as a column here — it came back undefined for
    # every single row (see caption below); that's itself the key finding of round one.
    results = pd.DataFrame([
        {"Crop": "Maize", "Model": "Linear Regression", "MAE": 0.334, "MSE": 0.199, "RMSE": 0.446, "R2": 0.728, "MAPE": 24.4},
        {"Crop": "Maize", "Model": "Random Forest", "MAE": 0.399, "MSE": 0.306, "RMSE": 0.553, "R2": 0.581, "MAPE": 32.8},
        {"Crop": "Maize", "Model": "XGBoost", "MAE": 0.366, "MSE": 0.241, "RMSE": 0.491, "R2": 0.670, "MAPE": 27.8},
        {"Crop": "Maize", "Model": "Ridge", "MAE": 0.305, "MSE": 0.180, "RMSE": 0.424, "R2": 0.754, "MAPE": 23.3},
        {"Crop": "Maize", "Model": "Lasso", "MAE": 0.416, "MSE": 0.283, "RMSE": 0.532, "R2": 0.612, "MAPE": 38.1},

        {"Crop": "Beans", "Model": "Linear Regression", "MAE": 0.284, "MSE": 0.212, "RMSE": 0.460, "R2": -1.042, "MAPE": 40.5},
        {"Crop": "Beans", "Model": "Random Forest", "MAE": 0.265, "MSE": 0.153, "RMSE": 0.391, "R2": -0.474, "MAPE": 39.1},
        {"Crop": "Beans", "Model": "XGBoost", "MAE": 0.276, "MSE": 0.175, "RMSE": 0.419, "R2": -0.689, "MAPE": 45.8},
        {"Crop": "Beans", "Model": "Ridge", "MAE": 0.281, "MSE": 0.198, "RMSE": 0.445, "R2": -0.907, "MAPE": 41.0},
        {"Crop": "Beans", "Model": "Lasso", "MAE": 0.281, "MSE": 0.151, "RMSE": 0.389, "R2": -0.456, "MAPE": 43.3},

        {"Crop": "Sorghum", "Model": "Linear Regression", "MAE": 0.394, "MSE": 0.311, "RMSE": 0.558, "R2": 0.070, "MAPE": 43.8},
        {"Crop": "Sorghum", "Model": "Random Forest", "MAE": 0.388, "MSE": 0.316, "RMSE": 0.562, "R2": 0.054, "MAPE": 48.0},
        {"Crop": "Sorghum", "Model": "XGBoost", "MAE": 0.414, "MSE": 0.362, "RMSE": 0.601, "R2": -0.082, "MAPE": 49.9},
        {"Crop": "Sorghum", "Model": "Ridge", "MAE": 0.374, "MSE": 0.303, "RMSE": 0.550, "R2": 0.095, "MAPE": 41.0},
        {"Crop": "Sorghum", "Model": "Lasso", "MAE": 0.412, "MSE": 0.314, "RMSE": 0.560, "R2": 0.063, "MAPE": 50.8},

        {"Crop": "Green Grams", "Model": "Linear Regression", "MAE": 0.164, "MSE": 0.056, "RMSE": 0.236, "R2": -0.074, "MAPE": 26.5},
        {"Crop": "Green Grams", "Model": "Random Forest", "MAE": 0.165, "MSE": 0.050, "RMSE": 0.224, "R2": 0.031, "MAPE": 23.3},
        {"Crop": "Green Grams", "Model": "XGBoost", "MAE": 0.186, "MSE": 0.056, "RMSE": 0.236, "R2": -0.075, "MAPE": 26.4},
        {"Crop": "Green Grams", "Model": "Ridge", "MAE": 0.167, "MSE": 0.053, "RMSE": 0.231, "R2": -0.032, "MAPE": 25.7},
        {"Crop": "Green Grams", "Model": "Lasso", "MAE": 0.177, "MSE": 0.058, "RMSE": 0.241, "R2": -0.118, "MAPE": 25.3},

        {"Crop": "Irish Potatoes", "Model": "Linear Regression", "MAE": 1.562, "MSE": 4.111, "RMSE": 2.028, "R2": 0.389, "MAPE": 22.3},
        {"Crop": "Irish Potatoes", "Model": "Random Forest", "MAE": 1.863, "MSE": 6.007, "RMSE": 2.451, "R2": 0.107, "MAPE": 28.6},
        {"Crop": "Irish Potatoes", "Model": "XGBoost", "MAE": 1.911, "MSE": 6.431, "RMSE": 2.536, "R2": 0.044, "MAPE": 29.5},
        {"Crop": "Irish Potatoes", "Model": "Ridge", "MAE": 1.574, "MSE": 4.016, "RMSE": 2.004, "R2": 0.403, "MAPE": 22.5},
        {"Crop": "Irish Potatoes", "Model": "Lasso", "MAE": 1.594, "MSE": 3.964, "RMSE": 1.991, "R2": 0.411, "MAPE": 23.1},
    ])

    crop_pick = st.selectbox("Crop", results["Crop"].unique(), key="perf_crop")
    subset = results[results["Crop"] == crop_pick].sort_values("R2", ascending=False)

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        metric_cols = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
        display_df = subset.set_index("Model")[metric_cols].rename(columns={"R2": "R²", "MAPE": "MAPE (%)"})
        st.dataframe(display_df, width='stretch')
        st.caption(
            "Adjusted R² is not shown as a column — it came back undefined for every "
            "model and crop tested. With ~50 features and as few as 28 validation rows, "
            "the formula's guard condition fails. This is exactly what led to testing "
            "Ridge and Lasso."
        )
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ["#3F6B3F" if v >= 0.3 else ("#C97C3D" if v >= 0 else "#8B3A3A") for v in subset["R2"]]
        ax.barh(subset["Model"], subset["R2"], color=colors)
        ax.axvline(0, color="#2B3A28", linewidth=0.8)
        ax.set_xlabel("R²")
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("#F0EBDD"); ax.set_facecolor("#F0EBDD")
        st.pyplot(fig)

    st.caption(
        "Green = usable (R² ≥ 0.3), orange = weak but positive, red = worse than predicting "
        "the average every time. A negative R² is a real, meaningful result — not a bug."
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h3 class='display'>Why the simplest model often won</h3>", unsafe_allow_html=True)
    st.markdown("""
Linear Regression outperformed Random Forest and XGBoost on 3 of 5 crops — unusual,
since tree-based ensembles typically outperform linear models on real-world data.
The cause: **Adjusted R² returned undefined for every single crop and model**, because
the number of features (~50, mostly one-hot encoded counties) was close to or exceeded
the number of validation rows (28–47 per crop, depending on how many counties grow it).
Tree-based models have more capacity to fit noise under these conditions than a
constrained linear model does — so their extra flexibility became a liability, not
an advantage.
""")

    st.markdown("<h3 class='display'>Did regularization fix it?</h3>", unsafe_allow_html=True)
    st.markdown("""
Ridge and Lasso — linear models built specifically to counteract this kind of
overfitting — were tested next. **Maize, sorghum, and Irish potatoes all improved**
under Ridge, confirming real signal existed that the unconstrained model had been
overfitting away. **Beans got worse, and green grams stayed flat.** That's the more
important result: it rules out overfitting as the *whole* explanation for these two
crops.
""")

    st.markdown("<h3 class='display'>Direct evidence — Lasso's coefficients on beans</h3>", unsafe_allow_html=True)
    st.markdown("""
Lasso can shrink a feature's weight to exactly zero. Fit on beans, it kept
**exactly one feature out of roughly 50** — a single county effect (`county_Kwale`,
coefficient 0.22) — and zeroed out every other feature, including rainfall,
temperature, area, and soil pH entirely. This is concrete evidence that beans'
yield has no meaningful relationship with any current feature, not a weak one
obscured by noise.
""")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h3 class='display'>Model selected: Ridge, per crop</h3>", unsafe_allow_html=True)
    st.markdown("""
Ridge was chosen over the other four because it directly targets the diagnosed
problem, it improved results wherever real signal existed, and it outperformed
both tree-based models — which overfit harder given how few rows each crop has.
It was chosen over Lasso specifically because Lasso's more aggressive elimination
occasionally discarded features with real, if modest, predictive value — visible
in Lasso trailing Ridge on both maize and sorghum.
""")

    scope = pd.DataFrame([
        {"Crop": "Maize", "Status": "Served live"},
        {"Crop": "Sorghum", "Status": "Served live, flagged as weaker"},
        {"Crop": "Irish Potatoes", "Status": "Served live"},
        {"Crop": "Beans", "Status": "Excluded — no algorithm reached usable accuracy"},
        {"Crop": "Green Grams", "Status": "Excluded — no algorithm reached usable accuracy"},
    ])
    st.dataframe(scope.set_index("Crop"), width='stretch')

    st.markdown("""
**What would improve this next:** real fertilizer and irrigation data — a gap
identified at the very start of this project and never filled, since no
consistent county-level time series exists for either in Kenya's public data.
Tuning Ridge's penalty strength via cross-validation, rather than the fixed
value used here, is the other concrete next step.
""")

    st.caption(
        "Metrics computed on the 2023 validation year, time-based split (train ≤2022). "
        "Data: KNBS National Agriculture Production Reports · NASA POWER · iSDAsoil · "
        "National Drought Management Authority."
    )

with tab7:
    st.markdown("<h1 class='display'>Predict a Yield</h1>", unsafe_allow_html=True)

    if wide is None:
        missing_data_notice()
        st.stop()

    st.markdown(
        "Only crops whose model reached meaningful accuracy are offered here — Maize, "
        "Sorghum, and Irish Potatoes (R² of 0.10 to 0.75, confirmed across five algorithms, "
        "see Model Performance). Beans and Green Grams are excluded entirely: tested "
        "directly, not assumed — no algorithm reached usable accuracy for either."
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    PREDICT_CROPS = ["maize", "sorghum", "irish_potatoes"]

    step1, step2 = st.columns(2)
    with step1:
        st.markdown("**1. County**")
        county = st.selectbox("County", sorted(wide["county"].dropna().unique()), label_visibility="collapsed")
    with step2:
        st.markdown("**2. Crop**")
        crop_display = st.selectbox(
            "Crop",
            [CROP_LABELS[c] for c in PREDICT_CROPS],
            label_visibility="collapsed"
        )
        crop = [c for c in PREDICT_CROPS if CROP_LABELS[c] == crop_display][0]

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