# =============================================================
# app/dashboard_data.py
# Week 4 — Shared Data Loading Module
# =============================================================
# What this file does:
#   Loads all data files and ML models needed by the Streamlit
#   dashboard, using Streamlit's caching system to ensure they
#   are only loaded ONCE per session, not on every interaction.
#
# Why a separate file?
#   streamlit_app.py imports from here. Keeping data loading
#   separate from UI code makes both files easier to read and
#   means we can test loading independently of the app.
#
# How to use in streamlit_app.py:
#   from app.dashboard_data import (
#       load_segments, load_cohort_retention,
#       load_churn_data, load_models
#   )
# =============================================================

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path

# ── Root path ──────────────────────────────────────────────────
# This file lives in app/, so ROOT is one level up
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# =============================================================
# WHY CACHING MATTERS IN STREAMLIT
# =============================================================
# Streamlit works by re-running your ENTIRE script top to bottom
# every time the user interacts with anything — clicks a button,
# moves a slider, selects from a dropdown. Without caching, every
# slider move would reload all CSV files and all ML models from
# disk. On this project that means:
#   - 3 CSV files (~5MB total) → ~1-2s reload each time
#   - 4 joblib model files     → ~0.5s reload each time
#   Total: ~3-5 seconds of freezing on EVERY interaction
#
# Streamlit provides two caching decorators:
#
# @st.cache_data     → for DATA (DataFrames, arrays, dicts)
#   - Caches the RETURN VALUE of the function
#   - Safe to use for data that doesn't change during the session
#   - Each unique set of function arguments gets its own cache
#
# @st.cache_resource → for RESOURCES (ML models, DB connections)
#   - Caches the OBJECT ITSELF (not a copy)
#   - Used for things that are expensive to create AND should be
#     shared across all users/sessions (models, scalers)
#   - Never copies the object — all parts of the app share one
#     instance, which is correct for read-only model inference
#
# With caching:
#   - First load: ~3-5 seconds (one time only)
#   - Every subsequent interaction: near-instant (~0ms)


# =============================================================
# FEATURE COLUMNS
# =============================================================
# Must match FEATURE_COLS in src/churn_model.py exactly.
# Defined here once so both dashboard_data.py and streamlit_app.py
# reference the same list without duplication.

CHURN_FEATURE_COLS = [
    "Frequency",
    "Monetary",
    "F_Score",
    "M_Score",
    "avg_days_between_orders",
    "order_gap_std",
    "product_diversity",
    "CLV_capped",
]

# Human-readable labels for the churn predictor sliders
# Maps feature column name → (display label, description, unit)
FEATURE_DISPLAY = {
    "Frequency"               : ("Order Frequency",            "Total number of orders placed",               "orders"),
    "Monetary"                : ("Total Spend",                "Total £ spent across all orders",             "£"),
    "F_Score"                 : ("Frequency Score",            "Quintile score 1-5 (5 = most frequent)",      ""),
    "M_Score"                 : ("Monetary Score",             "Quintile score 1-5 (5 = highest spend)",      ""),
    "avg_days_between_orders" : ("Avg Days Between Orders",    "Average gap between consecutive orders",      "days"),
    "order_gap_std"           : ("Order Gap Variability",      "How erratic is the ordering pattern?",        "days"),
    "product_diversity"       : ("Product Diversity",          "Number of unique products ordered",           "products"),
    "CLV_capped"              : ("Customer Lifetime Value",    "Estimated CLV (capped at 99th pct)",          "£"),
}

# Segment colour palette — consistent with notebook plots
SEGMENT_COLOURS = {
    "Loyal VIP"       : "#185FA5",
    "Champions"       : "#0F6E56",
    "New / Promising" : "#854F0B",
    "At Risk"         : "#993C1D",
    "Needs Attention" : "#534AB7",
    "Lost / Inactive" : "#888882",
}


# =============================================================
# DATA LOADING FUNCTIONS
# =============================================================

@st.cache_data
def load_segments() -> pd.DataFrame:
    """
    Load rfm_segments.csv — one row per customer with RFM scores,
    K-Means cluster label, segment name, and CLV.

    Produced by: src/train.py
    Used by   : Tab 1 (Segment Explorer)

    @st.cache_data means this DataFrame is computed once and
    stored in memory. Every subsequent call to load_segments()
    returns the cached copy instantly without re-reading the file.
    """
    filepath = DATA / "rfm_segments.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find {filepath}. "
            f"Run src/train.py first."
        )

    df = pd.read_csv(filepath)

    # Ensure Customer ID is always a string
    # (may read as int if all IDs are numeric)
    df["Customer ID"] = df["Customer ID"].astype(str)

    return df


@st.cache_data
def load_cohort_retention() -> pd.DataFrame:
    """
    Load cohort_retention.csv — the monthly retention matrix.
    Rows = acquisition cohort months, columns = months since acquisition.
    Values = % of cohort still active.

    Produced by: src/cohorts.py
    Used by   : Tab 2 (Retention Viewer)
    """
    filepath = DATA / "cohort_retention.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find {filepath}. "
            f"Run src/cohorts.py first."
        )

    df = pd.read_csv(filepath, index_col=0)

    # Ensure columns are integers (months: 0, 1, 2 ...)
    # They may read as strings ("0", "1", "2") from CSV
    df.columns = df.columns.astype(int)

    return df


@st.cache_data
def load_churn_data() -> pd.DataFrame:
    """
    Load churn_labels.csv — every customer with:
    - Engineered churn features (Frequency, Monetary, etc.)
    - Churn label (Churned: 0/1)
    - Model prediction (Churn_Pred: 0/1)
    - Model probability (Churn_Prob: 0.0-1.0)
    - Split ('train' / 'test')
    - KMeans_Segment (merged from rfm_segments in churn_model.py)

    Produced by: src/churn_model.py
    Used by   : Tab 3 (Churn Predictor — existing customer lookup)
    """
    filepath = DATA / "churn_labels.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find {filepath}. "
            f"Run src/churn_model.py first."
        )

    df = pd.read_csv(filepath)
    df["Customer ID"] = df["Customer ID"].astype(str)

    # Round Churn_Prob to 3 decimal places for clean display
    if "Churn_Prob" in df.columns:
        df["Churn_Prob"] = df["Churn_Prob"].round(3)

    return df


@st.cache_resource
def load_models() -> dict:
    """
    Load all four ML model / scaler objects from joblib files.

    Returns a dictionary with keys:
        "kmeans"        → fitted KMeans model (Week 2 clustering)
        "rfm_scaler"    → StandardScaler fitted on RFM features
        "churn_model"   → fitted XGBoost churn classifier
        "churn_scaler"  → StandardScaler fitted on churn features

    WHY @st.cache_resource not @st.cache_data?
    Models are not simple data — they are Python objects with
    internal state. @st.cache_resource:
    - Keeps the SAME object in memory (not a copy per call)
    - Shares ONE instance across all users (safe for read-only inference)
    - Never serialises/deserialises the object (unlike cache_data)

    IMPORTANT: Two separate scalers exist in this project.
    They are NOT interchangeable:
        rfm_scaler   → fitted on [Recency, Frequency, Monetary]
                        used by the K-Means model for clustering
        churn_scaler → fitted on CHURN_FEATURE_COLS (8 features)
                        used by the XGBoost model for churn prediction
    Using the wrong scaler on the wrong model produces silently
    wrong predictions — no error is raised, the numbers are just
    meaningless. Always use models["churn_scaler"] with the churn
    model, and models["rfm_scaler"] with K-Means.
    """
    required_files = {
        "kmeans"      : DATA / "kmeans_model.joblib",
        "rfm_scaler"  : DATA / "scaler.joblib",
        "churn_model" : DATA / "churn_model.joblib",
        "churn_scaler": DATA / "churn_scaler.joblib",
    }

    # Check all files exist before attempting to load
    missing = [
        str(path) for key, path in required_files.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model files:\n" + "\n".join(missing) +
            f"\nRun src/train.py and src/churn_model.py first."
        )

    return {
        key: joblib.load(path)
        for key, path in required_files.items()
    }


# =============================================================
# HELPER FUNCTIONS
# =============================================================
# These are utility functions used by multiple tabs in the app.
# They operate on already-loaded data (no file I/O here).

def get_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics per KMeans_Segment.
    Used in the Tab 1 sidebar and summary table.
    """
    return (
        df.groupby("KMeans_Segment")
        .agg(
            Customers    = ("Customer ID",   "count"),
            Avg_Recency  = ("Recency",       "mean"),
            Avg_Frequency= ("Frequency",     "mean"),
            Avg_Monetary = ("Monetary",      "mean"),
            Avg_CLV      = ("CLV_capped",    "mean"),
            Churn_Rate   = ("Churned",       "mean"),
        )
        .round(1)
        .reset_index()
        .sort_values("Customers", ascending=False)
    )


def predict_churn_single(
    feature_values: dict,
    models: dict
) -> tuple:
    """
    Run the churn model on a single customer's feature values.

    Parameters:
        feature_values : dict mapping feature name → value
                         Keys must match CHURN_FEATURE_COLS exactly
        models         : dict returned by load_models()

    Returns:
        (churn_probability: float, churn_prediction: int, X_scaled: np.ndarray)
        churn_probability : 0.0 to 1.0
        churn_prediction  : 0 (active) or 1 (churned)
        X_scaled          : scaled feature array (used for SHAP)
    """
    # Build feature array in the exact column order expected by the model
    X = np.array([[feature_values[col] for col in CHURN_FEATURE_COLS]])

    # Scale using the CHURN scaler (not the RFM scaler — see load_models comment)
    X_scaled = models["churn_scaler"].transform(X)

    # Predict
    churn_prob = models["churn_model"].predict_proba(X_scaled)[0, 1]
    churn_pred = int(churn_prob >= 0.5)

    return round(float(churn_prob), 4), churn_pred, X_scaled


def get_churn_risk_label(probability: float) -> tuple:
    """
    Convert a churn probability to a human-readable risk label and colour.

    Returns:
        (label: str, colour: str, emoji: str)
    """
    if probability >= 0.75:
        return "High Risk",    "#993C1D", "🔴"
    elif probability >= 0.50:
        return "Medium Risk",  "#854F0B", "🟠"
    elif probability >= 0.25:
        return "Low Risk",     "#0F6E56", "🟡"
    else:
        return "Very Low Risk","#185FA5", "🟢"


def get_feature_ranges(churn_df: pd.DataFrame) -> dict:
    """
    Compute min, max, mean, and median for each churn feature.
    Used to set meaningful slider ranges and default values
    in the manual-entry mode of Tab 3.

    Returns:
        dict mapping feature name → {"min", "max", "mean", "median", "std"}
    """
    ranges = {}
    for col in CHURN_FEATURE_COLS:
        if col in churn_df.columns:
            ranges[col] = {
                "min"   : float(churn_df[col].min()),
                "max"   : float(churn_df[col].quantile(0.99)),  # cap at 99th pct
                "mean"  : float(churn_df[col].mean()),
                "median": float(churn_df[col].median()),
                "std"   : float(churn_df[col].std()),
            }
    return ranges


# =============================================================
# QUICK LOAD TEST
# =============================================================
# Run this file directly to verify all data and models load
# without errors before starting the Streamlit app:
#   python app/dashboard_data.py

if __name__ == "__main__":
    print("Testing data loading ...\n")

    print("Loading segments ...", end=" ")
    seg = load_segments()
    print(f"OK — {len(seg):,} customers, "
          f"segments: {seg['KMeans_Segment'].unique().tolist()}")

    print("Loading cohort retention ...", end=" ")
    ret = load_cohort_retention()
    print(f"OK — {ret.shape[0]} cohorts × {ret.shape[1]} months")

    print("Loading churn data ...", end=" ")
    churn = load_churn_data()
    print(f"OK — {len(churn):,} customers, "
          f"churn rate: {churn['Churned'].mean()*100:.1f}%")

    print("Loading models ...", end=" ")
    models = load_models()
    print(f"OK — loaded: {list(models.keys())}")

    print("\nFeature ranges:")
    ranges = get_feature_ranges(churn)
    for feat, r in ranges.items():
        print(f"  {feat:<30} min={r['min']:.1f}  "
              f"median={r['median']:.1f}  max={r['max']:.1f}")

    print("\nAll checks passed. Ready to run: streamlit run app/streamlit_app.py")
