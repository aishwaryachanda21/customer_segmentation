# =============================================================
# src/train.py
# Week 2 — K-Means Customer Segmentation + MLflow Tracking
# =============================================================
# What this script does:
#   1. Loads RFM scores from data/rfm_scores.csv
#   2. Scales R, F, M features using StandardScaler
#   3. Runs K-Means for k=2 to k=8 — logs every run to MLflow
#   4. Plots and saves the elbow curve + silhouette score chart
#   5. Identifies the best k by silhouette score
#   6. Fits the final model with best k
#   7. Assigns cluster labels back to the dataframe
#   8. Saves rfm_segments.csv, the scaler, and the model
#
# Run from your project root:
#   python src/train.py
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn
import joblib
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
IN_FILE     = ROOT / "data" / "rfm_scores.csv"
OUT_FILE    = ROOT / "data" / "rfm_segments.csv"
SCALER_PATH = ROOT / "data" / "scaler.joblib"
MODEL_PATH  = ROOT / "data" / "kmeans_model.joblib"
RPT_DIR     = ROOT / "reports"
RPT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────
# The three RFM columns we cluster on — raw values, not scores.
# We use raw R/F/M because StandardScaler needs continuous numbers.
# The 1-5 scores are ordinal categories and lose information when scaled.
FEATURES     = ["Recency", "Frequency", "Monetary"]
K_RANGE      = range(2, 9)      # try k = 2, 3, 4, 5, 6, 7, 8
RANDOM_STATE = 42               # fixed seed → reproducible clusters
N_INIT       = 10               # how many times K-Means reruns with
                                # different starting points per k value
                                # (takes the best result each time)


# =============================================================
# STEP 1: Load RFM data
# =============================================================

def load_data(filepath: Path) -> pd.DataFrame:
    print("=" * 58)
    print("  STEP 1: Loading RFM scores")
    print("=" * 58)

    if not filepath.exists():
        raise FileNotFoundError(
            f"\n  Could not find: {filepath}"
            f"\n  Run src/features.py first to generate rfm_scores.csv"
        )

    df = pd.read_csv(filepath)
    print(f"  Loaded  : {len(df):,} customers × {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")
    print(f"\n  RFM feature summary:")
    print(df[FEATURES].describe().round(2).to_string())
    print()
    return df


def remove_outliers(df: pd.DataFrame) -> tuple:
    """
    Remove extreme outlier customers before clustering.
    These are typically bulk B2B wholesale accounts that are so
    different from regular retail customers that they form their
    own clusters and prevent meaningful segmentation of everyone else.

    We cap at the 99th percentile of Monetary value — customers
    above this threshold are separated out, not deleted entirely.
    They're saved separately so the business still knows who they are.

    Why 99th percentile?
    Removes the top 1% of spenders — a conservative cut that only
    removes genuine outliers, not just high-value customers.
    """
    print("=" * 58)
    print("  STEP 1b: Removing outliers before clustering")
    print("=" * 58)

    monetary_cap = df["Monetary"].quantile(0.99)
    freq_cap     = df["Frequency"].quantile(0.99)

    mask_outlier = (df["Monetary"]  > monetary_cap) | \
                   (df["Frequency"] > freq_cap)

    df_outliers = df[mask_outlier].copy()
    df_clean    = df[~mask_outlier].copy()

    print(f"  Monetary 99th pct cap  : £{monetary_cap:,.0f}")
    print(f"  Frequency 99th pct cap : {freq_cap:.0f} orders")
    print(f"  Outliers removed       : {len(df_outliers)} customers")
    print(f"  Remaining for clustering: {len(df_clean):,} customers")

    if len(df_outliers) > 0:
        print(f"\n  Outlier profiles (these are your bulk B2B accounts):")
        print(df_outliers[["Customer ID", "Recency", "Frequency",
                            "Monetary"]].to_string(index=False))

    # Save outliers separately — they're VIP accounts worth tracking
    outlier_path = ROOT / "data" / "vip_outliers.csv"
    df_outliers.to_csv(outlier_path, index=False)
    print(f"\n  VIP outliers saved → {outlier_path}\n")

    return df_clean, df_outliers

# =============================================================
# STEP 2: Scale features
# =============================================================

def scale_features(df: pd.DataFrame) -> tuple:
    """
    Why scale?
    K-Means calculates distances between points. Without scaling,
    a feature with large values (Monetary: £0–£280,000) dominates
    over one with small values (Frequency: 1–200 orders). The model
    would effectively only cluster on Monetary and ignore the others.

    StandardScaler transforms each feature to mean=0, std=1.
    After scaling, all three features contribute equally to distances.

    We return BOTH the scaled array AND the fitted scaler object.
    The scaler must be saved so the Streamlit app can transform
    new input values the same way before predicting a cluster.
    """
    print("=" * 58)
    print("  STEP 2: Scaling RFM features")
    print("=" * 58)

    scaler = StandardScaler()

    # fit_transform: learns the mean/std from the data (fit),
    # then applies the transformation (transform) in one step.
    # Returns a numpy array, shape (n_customers, 3)
    X_scaled = scaler.fit_transform(df[FEATURES])

    print(f"  Features scaled : {FEATURES}")
    print(f"  Scaler means    : {dict(zip(FEATURES, scaler.mean_.round(2)))}")
    print(f"  Scaler stds     : {dict(zip(FEATURES, scaler.scale_.round(2)))}")
    print(f"  Scaled shape    : {X_scaled.shape}")
    print(f"  Scaled mean     ≈ 0: {X_scaled.mean(axis=0).round(4)}")
    print(f"  Scaled std      ≈ 1: {X_scaled.std(axis=0).round(4)}\n")

    print(X_scaled)
    print(scaler)
    return X_scaled, scaler


# =============================================================
# STEP 3: Run K-Means for k=2 to k=8, log each to MLflow
# =============================================================

def run_experiments(X_scaled: np.ndarray) -> dict:
    """
    Run K-Means for each k value and collect two key metrics:

    Inertia (elbow method):
        Sum of squared distances from each point to its cluster centre.
        Lower = tighter clusters. Always decreases as k increases.
        We look for the "elbow" — where adding more clusters stops
        helping much. That elbow is a good candidate for best k.

    Silhouette score:
        Measures how similar each point is to its own cluster vs
        neighbouring clusters. Range: -1 (bad) to +1 (perfect).
        Higher = better separated clusters.
        Unlike inertia, this has a natural maximum — we pick the k
        that MAXIMISES silhouette score.

    We log both metrics to MLflow so we can compare runs in the UI.
    """
    print("=" * 58)
    print("  STEP 3: Running K-Means experiments (k=2 to k=8)")
    print("=" * 58)

    # Allow file-store backend (MLflow >= 2.x requires explicit opt-in)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # Point MLflow at the SQLite database in the project root
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("customer_segmentation")

    results = {}   # store {k: {"inertia": x, "silhouette": y, "labels": [...]}}

    for k in K_RANGE:
        print(f"  Running k={k} ...", end=" ")

        # ── Fit K-Means ──────────────────────────────────────
        # n_init: runs K-Means n_init times with different random
        # starting centroids. Keeps the run with lowest inertia.
        # This avoids getting stuck in a bad local minimum.
        km = KMeans(
            n_clusters=k,
            n_init=N_INIT,
            random_state=RANDOM_STATE
        )
        labels   = km.fit_predict(X_scaled)   # fit + assign cluster labels
        inertia  = km.inertia_                 # total within-cluster variance
        sil      = silhouette_score(X_scaled, labels, sample_size=5000,
                                    random_state=RANDOM_STATE)
        # sample_size=5000: silhouette is expensive on large datasets.
        # Computing on a random sample of 5000 points is fast and accurate.

        # Cluster sizes: how many customers in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes  = dict(zip(unique.tolist(), counts.tolist()))

        print(f"inertia={inertia:,.0f}  silhouette={sil:.4f}  "
              f"sizes={cluster_sizes}")

        # ── Log to MLflow ────────────────────────────────────
        with mlflow.start_run(run_name=f"kmeans_k{k}"):

            # Params: the settings we chose (not computed from data)
            mlflow.log_param("k",            k)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_init",       N_INIT)
            mlflow.log_param("features",     str(FEATURES))

            # Metrics: the numbers computed from the model fit
            mlflow.log_metric("inertia",         inertia)
            mlflow.log_metric("silhouette_score", sil)
            mlflow.log_metric("n_customers",      len(labels))

            # Log each cluster size as a separate metric
            # e.g. cluster_size_0, cluster_size_1, ...
            for cluster_id, size in cluster_sizes.items():
                mlflow.log_metric(f"cluster_size_{cluster_id}", size)

        # Store results for plotting and final model selection
        results[k] = {
            "inertia"    : inertia,
            "silhouette" : sil,
            "labels"     : labels,
            "cluster_sizes": cluster_sizes,
            "model"      : km,
        }
    print()
    return results

# =============================================================
# STEP 4: Plot elbow curve + silhouette scores
# =============================================================

def plot_evaluation_charts(results: dict) -> None:
    """
    Two charts to help choose the best k visually:

    Left — Elbow curve:
        Inertia vs k. Look for the bend/elbow where the line
        starts to flatten. That k is usually a good choice.
        The elbow is subjective — silhouette score is more objective.

    Right — Silhouette scores:
        Silhouette vs k. The peak (highest bar) is the best k
        by this metric. More objective than the elbow.

    We annotate both charts clearly to be portfolio-ready.
    """
    print("=" * 58)
    print("  STEP 4: Plotting evaluation charts")
    print("=" * 58)

    ks          = list(results.keys())
    inertias    = [results[k]["inertia"]   for k in ks]
    silhouettes = [results[k]["silhouette"] for k in ks]
    best_k      = max(results, key=lambda k: results[k]["silhouette"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("K-Means Evaluation: Analyzing for the Best k",
                 fontsize=14, fontweight="500", y=1.01)

    # ── Left: Elbow curve ────────────────────────────────────
    ax = axes[0]
    ax.plot(ks, inertias, marker="o", color="#185FA5",
            linewidth=2, markersize=7)

    # Highlight the best k (by silhouette) on the elbow chart too
    ax.axvline(best_k, color="#993C1D", linestyle="--",
               linewidth=1.2, label=f"Best k={best_k} (by silhouette)")

    ax.set_title("Elbow Curve — Inertia vs k", pad=10)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster variance)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
                              else f"{x/1e3:.0f}k"))
    ax.set_xticks(ks)
    ax.legend(fontsize=9)

    # ── Right: Silhouette scores ─────────────────────────────
    ax = axes[1]
    bar_colors = ["#993C1D" if k == best_k else "#185FA5" for k in ks]
    bars = ax.bar(ks, silhouettes, color=bar_colors,
                  alpha=0.8, width=0.6, edgecolor="white")

    # Annotate each bar with its value
    ax.bar_label(bars,
                 labels=[f"{s:.3f}" for s in silhouettes],
                 padding=4, fontsize=9)

    ax.set_title("Silhouette Score vs k (generally higher = better)", pad=10)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(silhouettes) * 1.15)

    # Label the best k
    ax.annotate(f"Best k={best_k}",
                xy=(best_k, results[best_k]["silhouette"]),
                xytext=(best_k + 0.4, results[best_k]["silhouette"] + 0.01),
                fontsize=9, color="#993C1D",
                arrowprops=dict(arrowstyle="->", color="#993C1D", lw=1))

    plt.tight_layout()
    out_path = RPT_DIR / "plot_08_kmeans_evaluation.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.show()
    print(f"  Saved → {out_path}\n")


# =============================================================
# STEP 5: Select best k and print comparison table
# =============================================================
def select_best_k(results: dict, override_k: int = None) -> int:
    """
        Print a comparison table of all k values and select the best.

        We use silhouette score as the primary criterion because:
        - It has a natural maximum (unlike inertia which always decreases)
        - It measures cluster quality, not just tightness
        - It's more robust to differences in cluster size

        The elbow chart is also shown as supporting evidence, as the final best_k selection would depend on a balanced interpretation
        from elbow and sil score, and domain knowledge based decision.
        """
    print("=" * 58)
    print("  STEP 5: Comparing runs — selecting best k")
    print("=" * 58)

    print(f"\n  {'k':>4}  {'Inertia':>14}  {'Silhouette':>12}  {'Cluster sizes'}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*12}  {'─'*30}")

    metric_best_k  = max(results, key=lambda k: results[k]["silhouette"])
    metric_best_sil = results[metric_best_k]["silhouette"]

    for k in sorted(results.keys()):
        r      = results[k]
        marker = "  ← BEST (by silhouette)" if k == metric_best_k else ""
        print(f"  {k:>4}  {r['inertia']:>14,.0f}  "
              f"{r['silhouette']:>12.4f}  {r['cluster_sizes']}{marker}")

    print(f"\n  Metric selection → k={metric_best_k}  "
          f"(silhouette = {metric_best_sil:.4f})")

    if override_k and override_k != metric_best_k:
        print(f"\n  ⚠  Domain override applied → k={override_k}")
        print(f"     Reason: k={metric_best_k} produces a trivial split")
        print(f"     (bulk B2B buyers vs retail customers) with limited")
        print(f"     business utility. Elbow curve inflects at k=3.")
        print(f"     k={override_k} selected for four interpretable retention")
        print(f"     segments aligned with marketing strategy, maintaining")
        print(f"     a silhouette score of "
              f"{results[override_k]['silhouette']:.4f}.")
        final_k = override_k
    else:
        final_k = metric_best_k

    print(f"\n  Final k = {final_k}\n")
    return final_k

# =============================================================
# STEP 6: Fit final model with best k
# =============================================================

def fit_final_model(X_scaled: np.ndarray, best_k: int) -> KMeans:
    """
    Refit K-Means with the chosen best k.

    Why refit instead of reusing the stored model from the loop?
    Cleanliness and explicitness — we want the final model to be
    a fresh, clearly intentional fit, not a leftover from the loop.
    We also increase n_init to 20 for the final fit to maximise the
    chance of finding the global optimum (costs little extra time).
    """
    print("=" * 58)
    print(f"  STEP 6: Fitting final model with k={best_k}")
    print("=" * 58)

    final_model = KMeans(
        n_clusters=best_k,
        n_init=20,                  # more restarts for the final model
        random_state=RANDOM_STATE
    )
    final_model.fit(X_scaled)

    print(f"  Final model fitted  ✓")
    print(f"  Inertia : {final_model.inertia_:,.0f}")
    print(f"  Cluster centres (scaled):")

    centres_df = pd.DataFrame(
        final_model.cluster_centers_,
        columns=FEATURES
    )
    centres_df.index.name = "Cluster"
    print(centres_df.round(3).to_string())
    print()

    return final_model



# =============================================================
# STEP 7: Assign cluster labels + name segments
# =============================================================
def assign_clusters(df: pd.DataFrame,
                    X_scaled: np.ndarray,
                    model: KMeans) -> pd.DataFrame:
    """
    Both Cluster 1 and Cluster 3 are above median on R, F, and M so both hit the Champions branch.
    The earlier logic has no way to distinguish "moderately above median" from "extremely above median."
    The fix — name from scaled cluster centres, not raw medians
    The scaled centres already tell you everything you need. Use them directly.
    """
    print("=" * 58)
    print("  STEP 7: Assigning cluster labels and segment names")
    print("=" * 58)

    # Predict cluster for every customer
    df = df.copy()
    df["Cluster"] = model.predict(X_scaled)

    # Build cluster profiles from SCALED centres — more reliable
    # than raw medians for naming because they're already normalised
    centres_scaled = pd.DataFrame(
        model.cluster_centers_,
        columns=FEATURES
    )
    centres_scaled.index.name = "Cluster"

    print("\n  Cluster centres (scaled):")
    print(centres_scaled.round(3).to_string())

    # Also show unscaled averages for interpretability
    cluster_profile = df.groupby("Cluster")[FEATURES].mean()
    print("\n  Cluster RFM profiles (unscaled averages):")
    print(cluster_profile.round(1).to_string())

    def name_cluster(row):
        """
        Name clusters using scaled centre values.
        Scaled values: 0 = average, positive = above average, negative = below.
        Recency is inverted: negative scaled recency = bought recently = good.

        Thresholds:
            |value| < 0.5  → near average
            value > 0.5    → notably above average
            value < -0.5   → notably below average
        """
        r = row["Recency"]  # negative = recent = good
        f = row["Frequency"]
        m = row["Monetary"]

        recent = r < -0.3  # bought recently
        dormant = r > 0.3  # hasn't bought in a while
        frequent = f > 0.3
        low_freq = f < -0.3
        high_val = m > 0.3
        low_val = m < -0.3

        if recent and frequent and high_val and (f > 1.5 or m > 1.5):
            return "Loyal VIP"  # Cluster 3: extreme on all dimensions
        elif recent and frequent and high_val:
            return "Champions"  # Cluster 1: solidly above average
        elif recent and low_freq and low_val:
            return "New / Promising"  # Cluster 2: recent but not yet loyal
        elif recent and not frequent:
            return "New / Promising"
        elif dormant and low_freq:
            return "Lost / Inactive"  # Cluster 0: gone quiet
        elif dormant and frequent:
            return "At Risk"  # were loyal, now lapsing
        else:
            return "Needs Attention"

    segment_map = centres_scaled.apply(name_cluster, axis=1).to_dict()

    print(f"\n  Cluster → Segment mapping:")
    for cluster_id, seg_name in segment_map.items():
        size = (df["Cluster"] == cluster_id).sum()
        print(f"    Cluster {cluster_id} → {seg_name:<22} ({size:,} customers)")

    df["KMeans_Segment"] = df["Cluster"].map(segment_map)

    print()
    return df, segment_map

# =============================================================
# STEP 8: Save outputs
# =============================================================

def save_outputs(df: pd.DataFrame,
                 scaler: StandardScaler,
                 model: KMeans,
                 segment_map: dict,
                 results: dict,
                 best_k: int) -> None:
    """
    Save three things:

    1. rfm_segments.csv  — one row per customer, includes cluster
                           label, segment name, and all RFM scores.
                           Used by the Streamlit dashboard and Week 3.

    2. scaler.joblib     — the fitted StandardScaler.
                           MUST be saved: the Streamlit app needs it
                           to transform new input values before
                           predicting a cluster. If you retrain with
                           new data, you must resave this too.

    3. kmeans_model.joblib — the fitted KMeans model.
                           Used for predicting the cluster of a new
                           customer in the Streamlit churn predictor.

    joblib is the standard way to save sklearn models — it's faster
    than pickle for objects with large numpy arrays (like cluster centres).
    """
    print("=" * 58)
    print("  STEP 8: Saving outputs")
    print("=" * 58)

    # ── rfm_segments.csv ────────────────────────────────────
    df.to_csv(OUT_FILE, index=False)
    print(f"  rfm_segments.csv saved → {OUT_FILE}")
    print(f"    Rows    : {len(df):,}")
    print(f"    Columns : {list(df.columns)}")

    # ── scaler.joblib ────────────────────────────────────────
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  scaler.joblib saved    → {SCALER_PATH}")

    # ── kmeans_model.joblib ──────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    print(f"  kmeans_model.joblib    → {MODEL_PATH}")

    # ── Log final model artifacts to MLflow ──────────────────
    # Log the final model and its outputs as a separate MLflow run
    # so it's clearly marked as the chosen production model.
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("customer_segmentation")

    with mlflow.start_run(run_name=f"FINAL_MODEL_k{best_k}"):
        mlflow.log_param("best_k",        best_k)
        mlflow.log_param("random_state",  RANDOM_STATE)
        mlflow.log_param("features",      str(FEATURES))
        mlflow.log_param("segment_map",   str(segment_map))

        mlflow.log_metric("final_inertia",
                          results[best_k]["inertia"])
        mlflow.log_metric("final_silhouette",
                          results[best_k]["silhouette"])

        # Log the saved model and scaler files as artifacts
        mlflow.log_artifact(str(MODEL_PATH),  artifact_path="model")
        mlflow.log_artifact(str(SCALER_PATH), artifact_path="model")
        mlflow.log_artifact(str(OUT_FILE),    artifact_path="data")

        # Log evaluation chart
        chart_path = RPT_DIR / "plot_08_kmeans_evaluation.png"
        if chart_path.exists():
            mlflow.log_artifact(str(chart_path), artifact_path="charts")

        print(f"\n  Final model logged to MLflow as 'FINAL_MODEL_k{best_k}'")


# =============================================================
# STEP 9: Print final summary
# =============================================================

def print_summary(df: pd.DataFrame, best_k: int) -> None:
    print("\n" + "=" * 58)
    print("  FINAL SUMMARY")
    print("=" * 58)
    print(f"  Best k selected      : {best_k}")
    print(f"  Total customers      : {len(df):,}")
    print()

    seg_summary = (df.groupby("KMeans_Segment")
                     .agg(
                         Customers   = ("Customer ID", "count"),
                         Avg_Recency = ("Recency",    "mean"),
                         Avg_Freq    = ("Frequency",  "mean"),
                         Avg_Monetary= ("Monetary",   "mean"),
                         Avg_CLV     = ("CLV_capped", "mean"),
                     )
                     .round(1)
                     .sort_values("Customers", ascending=False))

    print("  Segment profiles:")
    print(seg_summary.to_string())

    print(f"\n  Files saved:")
    print(f"    data/rfm_segments.csv")
    print(f"    data/scaler.joblib")
    print(f"    data/kmeans_model.joblib")
    print(f"    reports/plot_08_kmeans_evaluation.png")
    print(f"\n  MLflow UI: run 'mlflow ui --backend-store-uri "
          f"sqlite:///mlflow.db'")
    print(f"  Then open: http://127.0.0.1:5000")
    print(f"\n  Next step: open notebooks/02_rfm_clustering.ipynb\n")


# =============================================================
# Main execution
# =============================================================
if __name__ == "__main__":
    print("\n  UCI Online Retail II — K-Means Clustering Pipeline")
    print("  " + "─" * 54 + "\n")

    df                   = load_data(IN_FILE)
    df_clean, df_outliers = remove_outliers(df)      # ← add this line
    X_scaled, scaler     = scale_features(df_clean)  # ← use df_clean
    # ── Save full-dataset scaled array for K-Medoids comparison ──
    # Scale the FULL df (including outliers) using the SAME scaler
    # fitted on df_clean. We don't refit — using the same scaler
    # ensures fair comparison. transform() applies without refitting.
    X_scaled_full = scaler.transform(df[FEATURES])
    np.save(ROOT / "data" / "X_scaled_full.npy", X_scaled_full)
    print(f"  X_scaled_full saved → data/X_scaled_full.npy  "
          f"shape={X_scaled_full.shape}\n")

    # Also save the full df with Customer IDs aligned to X_scaled_full
    df[["Customer ID"] + FEATURES].to_csv(
        ROOT / "data" / "rfm_full.csv", index=False
    )
    print(f"  rfm_full.csv saved  → data/rfm_full.csv\n")
    results              = run_experiments(X_scaled)
    plot_evaluation_charts(results)
    best_k = select_best_k(results, override_k=4)
                             # ← override: domain judgement over pure metric
                             # "Although k=2 achieved the highest silhouette score (0.927),
                             # this reflects a bulk-buyer vs retail-customer split with limited business utility.
                             # The elbow curve identifies k=3 as the inflection point.
                             # k=4 was selected as it provides four interpretable customer segments
                             # aligned with retention strategy, while maintaining a strong silhouette score of 0.592."
    final_model          = fit_final_model(X_scaled, best_k)
    df_clean, segment_map = assign_clusters(df_clean, X_scaled, final_model)
    save_outputs(df_clean, scaler, final_model, segment_map, results, best_k)
    print_summary(df_clean, best_k)