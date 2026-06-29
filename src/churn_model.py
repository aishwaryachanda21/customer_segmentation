# =============================================================
# src/churn_model.py
# Week 3 — Churn Prediction Model
# =============================================================
# What this script does:
#   1. Loads clean transaction data + RFM segments
#   2. Derives the churn threshold from the data itself
#      (median inter-purchase gap + 1.5 x IQR)
#   3. Engineers behavioural features beyond basic RFM
#   4. Labels each customer as churned (1) or active (0)
#   5. Checks class balance — applies SMOTE only if needed
#   6. Trains an XGBoost classifier
#   7. Evaluates: ROC-AUC, F1, precision, recall, confusion matrix
#   8. Logs everything to MLflow
#   9. Generates SHAP feature importance plots
#  10. Saves model, scaler, and churn_labels.csv
#
# Run from your project root:
#   python src/churn_model.py
# =============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import mlflow
import mlflow.sklearn
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
TXN_FILE      = ROOT / "data" / "clean_retail.csv"
RFM_FILE      = ROOT / "data" / "rfm_segments.csv"
OUT_LABELS    = ROOT / "data" / "churn_labels.csv"
MODEL_PATH    = ROOT / "data" / "churn_model.joblib"
SCALER_PATH   = ROOT / "data" / "churn_scaler.joblib"
RPT_DIR       = ROOT / "reports"
RPT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────
RANDOM_STATE       = 42
TEST_SIZE          = 0.2      # 80% train, 20% test
IMBALANCE_THRESHOLD = 0.30   # apply SMOTE if minority class < 30%
# Features used for training (engineered below)
FEATURE_COLS = [
    # REMOVED: "Recency"       ← directly encodes the churn label
    # REMOVED: "R_Score"       ← derived from Recency
    # REMOVED: "recency_trend" ← derived from days since last purchase
    "Frequency",
    "Monetary",
    "F_Score",
    "M_Score",
    #"RFM_Total",
    "avg_days_between_orders",   # avg gap between past orders — not the current gap
    "order_gap_std",
    "product_diversity",
    "CLV_capped",
]

# =============================================================
# STEP 1: Load data
# =============================================================

def load_data() -> tuple:
    print("=" * 60)
    print("  STEP 1: Loading data")
    print("=" * 60)

    for f in [TXN_FILE, RFM_FILE]:
        if not f.exists():
            script = "ingest.py" if "clean" in f.name else "features.py + train.py"
            raise FileNotFoundError(
                f"\n  Could not find: {f}"
                f"\n  Run src/{script} first."
            )

    txn = pd.read_csv(TXN_FILE, parse_dates=["InvoiceDate"])
    rfm = pd.read_csv(RFM_FILE)

    print(f"  Transactions : {len(txn):,} rows")
    print(f"  RFM segments : {len(rfm):,} customers")
    print(f"  Date range   : {txn['InvoiceDate'].min().date()} → "
          f"{txn['InvoiceDate'].max().date()}\n")
    return txn, rfm


# =============================================================
# STEP 2: Derive churn threshold from the data
# =============================================================

def derive_churn_threshold(txn: pd.DataFrame) -> float:
    """
    Compute a data-driven churn threshold rather than assuming 90 days.

    Why not use a hardcoded 90-day threshold?
    This dataset has a B2B / wholesale character (flat retention curve,
    weekday-heavy ordering pattern). Some customers naturally go 60-120+
    days between orders and are still active. A fixed 90-day threshold
    would mis-label many of them as churned.

    Method: for each customer with >= 2 purchases, compute the gap
    (in days) between each consecutive pair of purchases. Then take:

        threshold = median(all gaps) + 1.5 x IQR(all gaps)

    This means "a customer is churned if their current silence is longer
    than 95%+ of all historically normal gaps between purchases."
    It is derived purely from observed behaviour, not an assumption.

    Why 1.5 x IQR?
    The same multiplier used in box-plot outlier detection (Tukey's fence).
    It is a well-established, defensible statistical threshold that
    captures the natural upper bound of "normal" behaviour.
    """
    print("=" * 60)
    print("  STEP 2: Deriving churn threshold from data")
    print("=" * 60)

    # Sort by customer and date to compute consecutive gaps
    txn_sorted = txn.sort_values(["Customer ID", "InvoiceDate"])

    # For each customer, group purchase dates by invoice
    # (one date per invoice, ignoring multiple items in one order)
    invoice_dates = (
        txn_sorted.groupby(["Customer ID", "Invoice"])["InvoiceDate"]
        .min()
        .reset_index()
    )

    # Compute gap between consecutive orders per customer
    # shift(1) gets the previous row — so gap = current date - previous date
    invoice_dates = invoice_dates.sort_values(["Customer ID", "InvoiceDate"])
    invoice_dates["prev_date"] = invoice_dates.groupby(
        "Customer ID")["InvoiceDate"].shift(1)
    invoice_dates["gap_days"] = (
        invoice_dates["InvoiceDate"] - invoice_dates["prev_date"]
    ).dt.days

    # Drop first purchases (no previous date → NaN gap)
    gaps = invoice_dates["gap_days"].dropna()

    # Compute threshold
    median_gap = gaps.median()
    q75        = gaps.quantile(0.75)
    q25        = gaps.quantile(0.25)
    iqr        = q75 - q25
    threshold  = median_gap + 1.5 * iqr

    print(f"  Inter-purchase gap distribution:")
    print(f"    Count  : {len(gaps):,} gaps")
    print(f"    Min    : {gaps.min():.0f} days")
    print(f"    Median : {median_gap:.0f} days")
    print(f"    Mean   : {gaps.mean():.0f} days")
    print(f"    Q75    : {q75:.0f} days")
    print(f"    IQR    : {iqr:.0f} days")
    print(f"    Max    : {gaps.max():.0f} days")
    print(f"\n  Derived churn threshold = "
          f"median({median_gap:.0f}) + 1.5 x IQR({iqr:.0f})")
    print(f"  Threshold = {threshold:.0f} days  "
          f"(customers silent > {threshold:.0f} days = churned)\n")

    return round(threshold, 0)


# =============================================================
# STEP 3: Engineer behavioural features
# =============================================================

def engineer_features(txn: pd.DataFrame,
                      rfm: pd.DataFrame,
                      threshold: float) -> pd.DataFrame:
    """
    Build a richer feature set beyond basic RFM for the churn model.

    Why go beyond RFM?
    Recency, Frequency, and Monetary capture WHAT customers did, but
    not HOW CONSISTENTLY or IN WHAT DIRECTION. A customer who ordered
    10 times with very irregular gaps (gap_std=60 days) is a different
    churn risk than one who ordered 10 times like clockwork every 2 weeks
    (gap_std=3 days). These additional features capture that nuance.

    Features engineered:
    - avg_days_between_orders : average gap between consecutive orders
    - order_gap_std           : standard deviation of gaps — high std
                                means erratic purchasing = higher churn risk
    - product_diversity       : number of unique products ordered — breadth
                                of engagement with the catalogue
    - recency_trend           : slope of order dates over time — positive
                                means buying more recently = lower churn risk.
                                Computed as: (date of last order - date of
                                first order) / frequency, normalised by days.
    """
    print("=" * 60)
    print("  STEP 3: Engineering behavioural features")
    print("=" * 60)

    snapshot_date = txn["InvoiceDate"].max() + pd.Timedelta(days=1)

    # ── Per-invoice date table ────────────────────────────────
    invoice_dates = (
        txn.sort_values(["Customer ID", "InvoiceDate"])
        .groupby(["Customer ID", "Invoice"])["InvoiceDate"]
        .min()
        .reset_index()
    )
    invoice_dates = invoice_dates.sort_values(["Customer ID", "InvoiceDate"])
    invoice_dates["prev_date"] = invoice_dates.groupby(
        "Customer ID")["InvoiceDate"].shift(1)
    invoice_dates["gap_days"] = (
        invoice_dates["InvoiceDate"] - invoice_dates["prev_date"]
    ).dt.days

    # ── avg_days_between_orders ──────────────────────────────
    avg_gap = (
        invoice_dates.groupby("Customer ID")["gap_days"]
        .mean()
        .reset_index()
        .rename(columns={"gap_days": "avg_days_between_orders"})
    )

    # ── order_gap_std ────────────────────────────────────────
    gap_std = (
        invoice_dates.groupby("Customer ID")["gap_days"]
        .std()
        .reset_index()
        .rename(columns={"gap_days": "order_gap_std"})
    )
    # Customers with only 1 order have NaN std — fill with 0
    # (no variation = perfectly consistent, even if only one data point)
    gap_std["order_gap_std"] = gap_std["order_gap_std"].fillna(0)

    # ── product_diversity ────────────────────────────────────
    diversity = (
        txn.groupby("Customer ID")["StockCode"]
        .nunique()
        .reset_index()
        .rename(columns={"StockCode": "product_diversity"})
    )

    # ── recency_trend ────────────────────────────────────────
    # Measure whether the customer's purchase activity is accelerating
    # or decelerating. Positive = buying more recently = lower churn risk.
    # Computed as: (last_purchase_date - first_purchase_date) / frequency
    # Normalised by total lifespan so short and long-tenure customers
    # are comparable.
    first_last = txn.groupby("Customer ID")["InvoiceDate"].agg(
        first_purchase="min",
        last_purchase="max"
    ).reset_index()
    first_last["lifespan_days"] = (
        (first_last["last_purchase"] - first_last["first_purchase"]).dt.days
    ).clip(lower=1)   # minimum 1 day to avoid division by zero
    first_last["recency_days"] = (
        snapshot_date - first_last["last_purchase"]
    ).dt.days

    # Trend: if lifespan is long but last purchase was recent,
    # recency_trend is low (buying right up to "now" = good)
    # If lifespan is short and last purchase was long ago, trend is high
    # We flip sign so HIGHER trend = MORE likely to churn (aligns with target)
    first_last["recency_trend"] = (
        first_last["recency_days"] / first_last["lifespan_days"]
    )

    # ── Churn label ──────────────────────────────────────────
    # A customer is "churned" if their most recent purchase was more
    # than `threshold` days before the snapshot date.
    # Label 1 = churned, 0 = active.
    first_last["Churned"] = (
        first_last["recency_days"] > threshold
    ).astype(int)

    # ── Merge all features ────────────────────────────────────
    features = rfm.copy()
    for extra_df in [avg_gap, gap_std, diversity,
                     first_last[["Customer ID", "recency_trend", "Churned"]]]:
        features = features.merge(extra_df, on="Customer ID", how="left")

    # Fill NaN avg_gap for single-purchase customers
    # (no gap exists — fill with the threshold itself as a conservative estimate)
    features["avg_days_between_orders"] = features[
        "avg_days_between_orders"].fillna(threshold)

    # Cap extreme gap values at 99th percentile (same outlier logic as Week 2)
    gap_cap = features["avg_days_between_orders"].quantile(0.99)
    features["avg_days_between_orders"] = features[
        "avg_days_between_orders"].clip(upper=gap_cap)
    features["order_gap_std"] = features["order_gap_std"].clip(
        upper=features["order_gap_std"].quantile(0.99))

    print(f"  Features engineered:")
    for col in FEATURE_COLS:
        n_null = features[col].isnull().sum()
        null_str = f"  ← {n_null} nulls" if n_null > 0 else ""
        print(f"    {col:<35} {null_str}")

    # Drop any remaining nulls
    before = len(features)
    features = features.dropna(subset=FEATURE_COLS + ["Churned"])
    dropped = before - len(features)
    if dropped > 0:
        print(f"\n  Dropped {dropped} rows with remaining nulls")

    print(f"\n  Final feature dataset: {len(features):,} customers\n")
    return features


# =============================================================
# STEP 4: Check class balance + apply SMOTE if needed
# =============================================================

def check_and_balance(X: pd.DataFrame,
                      y: pd.Series) -> tuple:
    """
    Check whether the churn/active classes are imbalanced.

    Why does class imbalance matter?
    If 90% of customers are "active" and only 10% "churned", a naive
    model can achieve 90% accuracy by predicting EVERYONE as active —
    but it would catch 0 churners. We need the model to learn from
    both classes equally.

    SMOTE (Synthetic Minority Over-sampling TEchnique):
    Generates synthetic examples of the minority class (churned)
    by interpolating between real minority examples. This is better
    than simple duplication because it adds variety, not just copies.

    We only apply SMOTE if minority class < IMBALANCE_THRESHOLD (30%).
    If the classes are already balanced enough, SMOTE is unnecessary
    and we proceed with the original data.

    CRITICAL: SMOTE is applied ONLY to training data, NEVER test data.
    Applying SMOTE to test data would inflate performance metrics and
    give you a falsely optimistic evaluation.
    """
    print("=" * 60)
    print("  STEP 4: Checking class balance")
    print("=" * 60)

    churn_rate   = y.mean()
    active_count = (y == 0).sum()
    churn_count  = (y == 1).sum()

    print(f"  Active  (0) : {active_count:>5,}  ({(1-churn_rate)*100:.1f}%)")
    print(f"  Churned (1) : {churn_count:>5,}  ({churn_rate*100:.1f}%)")
    print(f"  Churn rate  : {churn_rate*100:.1f}%")

    if churn_rate < IMBALANCE_THRESHOLD or churn_rate > (1 - IMBALANCE_THRESHOLD):
        print(f"\n  Imbalance detected (minority class < {IMBALANCE_THRESHOLD*100:.0f}%)")
        print(f"  SMOTE will be applied to the TRAINING set only.\n")
        apply_smote = True
    else:
        print(f"\n  Classes are sufficiently balanced. "
              f"SMOTE not required.\n")
        apply_smote = False
    return X, y, apply_smote, churn_rate


# =============================================================
# STEP 5: Train-test split + optional SMOTE
# =============================================================

def split_and_resample(X: pd.DataFrame,
                       y: pd.Series,
                       apply_smote: bool) -> tuple:
    """
    Split data into train/test sets, then optionally apply SMOTE
    to the training set only.

    Why StratifiedKFold isn't used here (but is mentioned):
    For this script we use a simple stratified train/test split.
    The `stratify=y` argument ensures both splits have the same
    churn rate — important when churn is rare. In a production
    setting you'd use cross-validation (StratifiedKFold) for more
    robust evaluation, but for portfolio purposes a single split
    with a held-out test set is sufficient and more interpretable.
    """
    print("=" * 60)
    print("  STEP 5: Train-test split + SMOTE")
    print("=" * 60)

    # stratify=y ensures the churn rate is the same in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Train set : {len(X_train):,} customers  "
          f"(churn rate: {y_train.mean()*100:.1f}%)")
    print(f"  Test  set : {len(X_test):,} customers   "
          f"(churn rate: {y_test.mean()*100:.1f}%)")

    # Scale features — XGBoost doesn't strictly require this, but
    # scaling makes SHAP values more comparable across features and
    # can speed up convergence for tree-based models on very skewed data.
    # We fit the scaler on TRAIN only, then transform both train and test.
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Apply SMOTE on the scaled TRAINING data only
    if apply_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_s, y_train = smote.fit_resample(X_train_s, y_train)
        print(f"\n  After SMOTE — Train set: {len(X_train_s):,} customers  "
              f"(churn rate: {y_train.mean()*100:.1f}%)")
    else:
        print(f"\n  No SMOTE applied.")

    print()
    return X_train_s, X_test_s, y_train, y_test, scaler


# =============================================================
# STEP 6: Train XGBoost model
# =============================================================

def train_model(X_train: np.ndarray,
                y_train: pd.Series) -> XGBClassifier:
    """
    Train an XGBoost classifier.

    Why XGBoost over RandomForest or Logistic Regression?
    - Better performance on tabular data with mixed feature types
    - Native handling of class imbalance via scale_pos_weight
      (secondary to SMOTE — belt and braces approach)
    - SHAP library has native XGBoost support — the cleanest
      integration for producing feature importance plots
    - Industry standard for tabular classification problems

    Hyperparameters chosen conservatively to avoid overfitting
    on a small-to-medium dataset:
    - max_depth=4     : shallow trees → less overfitting
    - n_estimators=200: enough trees to capture patterns
    - learning_rate=0.05: slow learning → more stable
    - subsample=0.8   : use 80% of rows per tree → regularisation
    - colsample_bytree=0.8: use 80% of features per tree → regularisation
    """
    print("=" * 60)
    print("  STEP 6: Training XGBoost classifier")
    print("=" * 60)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0
    )

    model.fit(X_train, y_train)
    print(f"  XGBoost trained  ✓")
    print(f"  n_estimators : {model.n_estimators}")
    print(f"  max_depth    : {model.max_depth}")
    print(f"  learning_rate: {model.learning_rate}\n")

    return model


# =============================================================
# STEP 7: Evaluate model
# =============================================================

def evaluate_model(model: XGBClassifier,
                   X_test: np.ndarray,
                   y_test: pd.Series) -> dict:
    """
    Evaluate using multiple metrics — not just accuracy.

    Why not accuracy?
    With a 75/25 class split, a model that always predicts "active"
    achieves 75% accuracy while catching zero churners. Useless.

    Metrics we use:
    - ROC-AUC : area under the ROC curve. Measures how well the model
                separates churned from active customers across ALL
                possible classification thresholds. 0.5 = random,
                1.0 = perfect. Good for imbalanced classes.
    - F1 score : harmonic mean of precision and recall at the default
                 0.5 threshold. Good single-number summary.
    - Precision: of customers we PREDICTED as churned, what % actually
                 churned? (avoid wasting win-back campaign budget)
    - Recall   : of ALL customers who actually churned, what % did we
                 catch? (avoid missing churners who need outreach)
    - Confusion matrix: raw counts for each prediction vs. actual combination
    """
    print("=" * 60)
    print("  STEP 7: Evaluating model on test set")
    print("=" * 60)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    roc_auc   = roc_auc_score(y_test, y_pred_prob)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"  ROC-AUC   : {roc_auc:.4f}  (0.5=random, 1.0=perfect)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}  (of predicted churners, % correct)")
    print(f"  Recall    : {recall:.4f}  (of actual churners, % caught)")
    print(f"\n  Confusion matrix:")
    print(f"                 Predicted Active  Predicted Churned")
    print(f"  Actual Active  {cm[0,0]:>16,}  {cm[0,1]:>17,}")
    print(f"  Actual Churned {cm[1,0]:>16,}  {cm[1,1]:>17,}")
    print(f"\n  Full classification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Active", "Churned"]))

    metrics = {
        "roc_auc"       : round(roc_auc,   4),
        "f1_score"      : round(f1,        4),
        "precision"     : round(precision, 4),
        "recall"        : round(recall,    4),
        "tn"            : int(cm[0, 0]),
        "fp"            : int(cm[0, 1]),
        "fn"            : int(cm[1, 0]),
        "tp"            : int(cm[1, 1]),
    }
    return metrics, y_pred, y_pred_prob


# =============================================================
# STEP 8: Plot evaluation charts
# =============================================================

def plot_evaluation(model: XGBClassifier,
                    X_test: np.ndarray,
                    y_test: pd.Series,
                    y_pred: np.ndarray,
                    y_pred_prob: np.ndarray,
                    metrics: dict,
                    feature_names: list) -> None:
    print("=" * 60)
    print("  STEP 8: Plotting evaluation charts")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Churn Model Evaluation", fontsize=14)

    # ── Confusion matrix ─────────────────────────────────────
    cm_display = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ])
    sns.heatmap(
        cm_display,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Active", "Pred Churned"],
        yticklabels=["Act Active",  "Act Churned"],
        ax=axes[0],
        cbar=False,
        linewidths=1,
        linecolor="white"
    )
    axes[0].set_title(f"Confusion Matrix")

    # ── ROC curve ────────────────────────────────────────────
    RocCurveDisplay.from_predictions(
        y_test, y_pred_prob,
        ax=axes[1],
        color="#185FA5",
        name=f"XGBoost (AUC={metrics['roc_auc']:.3f})"
    )
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.5)")
    axes[1].set_title("ROC Curve")
    axes[1].legend(fontsize=9)

    # ── Feature importance (XGBoost built-in) ────────────────
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    colours = ["#993C1D" if imp == importance.max()
               else "#185FA5" for imp in importance.values]
    importance.plot(kind="barh", ax=axes[2], color=colours, edgecolor="white")
    axes[2].set_title("Feature Importance (XGBoost)")
    axes[2].set_xlabel("Importance score")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = RPT_DIR / "plot_17_churn_model_eval.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.show()
    print(f"  Saved → {out_path}")


# =============================================================
# STEP 9: SHAP feature importance
# =============================================================

def plot_shap(model: XGBClassifier,
              X_test: np.ndarray,
              feature_names: list) -> None:
    """
    SHAP (SHapley Additive exPlanations) explains WHY the model
    makes each prediction by computing each feature's contribution.

    Unlike feature importance (which only tells you WHICH features
    matter), SHAP tells you:
    - WHICH features matter most
    - In WHICH DIRECTION (high value → more/less churn risk)
    - For EACH individual customer (not just on average)

    Summary plot: each dot = one customer. Colour = feature value
    (red = high, blue = low). X-axis = SHAP value (right = pushes
    toward churn, left = pushes away from churn).

    Waterfall plot: shows the top individual prediction broken down
    contribution by contribution. This is the most explainable output
    for a business stakeholder ("why was THIS customer flagged?").
    """
    print("=" * 60)
    print("  STEP 9: Generating SHAP plots")
    print("=" * 60)

    # TreeExplainer is the fastest SHAP explainer for tree-based models
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ── SHAP summary plot ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
        plot_size=None
    )
    plt.title("SHAP Summary Plot — Feature Impact on Churn Prediction",
              fontsize=12, pad=12)
    plt.tight_layout()
    out_path = RPT_DIR / "plot_18_shap_summary.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.show()
    print(f"  Saved → {out_path}")

    # ── SHAP waterfall plot (single customer) ────────────────
    # Show the prediction explanation for the customer with the
    # HIGHEST predicted churn probability — clearest portfolio demo
    shap_exp    = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_test,
        feature_names=feature_names
    )

    # Index of customer with highest predicted churn risk
    highest_risk_idx = np.argmax(
        shap_values.sum(axis=1)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.waterfall_plot(shap_exp[highest_risk_idx], show=False, max_display=12)
    plt.title("SHAP Waterfall — Highest-Risk Customer Explained",
              fontsize=11, pad=12)
    plt.tight_layout()
    out_path_wf = RPT_DIR / "plot_19_shap_waterfall.png"
    plt.savefig(out_path_wf, bbox_inches="tight", dpi=120)
    plt.show()
    print(f"  Saved → {out_path_wf}\n")


# =============================================================
# STEP 10: Log to MLflow
# =============================================================

def log_to_mlflow(model: XGBClassifier,
                  metrics: dict,
                  churn_threshold: float,
                  churn_rate: float,
                  apply_smote: bool,
                  feature_names: list) -> None:
    print("=" * 60)
    print("  STEP 10: Logging to MLflow")
    print("=" * 60)

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("customer_segmentation")

    with mlflow.start_run(run_name="churn_model_xgboost"):

        # Params
        mlflow.log_param("model_type",       "XGBoostClassifier")
        mlflow.log_param("churn_threshold",  churn_threshold)
        mlflow.log_param("features",         str(feature_names))
        mlflow.log_param("test_size",        TEST_SIZE)
        mlflow.log_param("smote_applied",    apply_smote)
        mlflow.log_param("random_state",     RANDOM_STATE)
        mlflow.log_param("n_estimators",     model.n_estimators)
        mlflow.log_param("max_depth",        model.max_depth)
        mlflow.log_param("learning_rate",    model.learning_rate)
        mlflow.log_param("subsample",        model.subsample)
        mlflow.log_param("colsample_bytree", model.colsample_bytree)

        # Metrics
        mlflow.log_metric("roc_auc",      metrics["roc_auc"])
        mlflow.log_metric("f1_score",     metrics["f1_score"])
        mlflow.log_metric("precision",    metrics["precision"])
        mlflow.log_metric("recall",       metrics["recall"])
        mlflow.log_metric("tp",           metrics["tp"])
        mlflow.log_metric("fp",           metrics["fp"])
        mlflow.log_metric("tn",           metrics["tn"])
        mlflow.log_metric("fn",           metrics["fn"])
        mlflow.log_metric("train_churn_rate", churn_rate)

        # Artifacts
        mlflow.log_artifact(str(MODEL_PATH),  artifact_path="model")
        mlflow.log_artifact(str(SCALER_PATH), artifact_path="model")
        mlflow.log_artifact(str(OUT_LABELS),  artifact_path="data")

        for plot in ["plot_17_churn_model_eval.png",
                     "plot_18_shap_summary.png",
                     "plot_19_shap_waterfall.png"]:
            p = RPT_DIR / plot
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="charts")

        run_id = mlflow.active_run().info.run_id

    print(f"  Run logged: churn_model_xgboost")
    print(f"  Run ID    : {run_id}")
    print(f"  View at   : http://127.0.0.1:5000  "
          f"(mlflow ui --backend-store-uri sqlite:///mlflow.db)\n")


# =============================================================
# STEP 11: Save outputs
# =============================================================

def save_outputs(features: pd.DataFrame,
                 y_pred: np.ndarray,
                 y_pred_prob: np.ndarray,
                 model: XGBClassifier,
                 scaler: StandardScaler,
                 test_index: pd.Index) -> None:
    print("=" * 60)
    print("  STEP 11: Saving outputs")
    print("=" * 60)

    # Add predictions back to the full feature dataframe
    # Use index alignment — predictions are only for the test split
    features = features.copy()
    features["Churn_Pred"]    = np.nan
    features["Churn_Prob"]    = np.nan
    features.loc[test_index, "Churn_Pred"] = y_pred
    features.loc[test_index, "Churn_Prob"] = y_pred_prob.round(4)

    # Save churn labels + all features
    features.to_csv(OUT_LABELS, index=False)
    print(f"  churn_labels.csv → {OUT_LABELS}")
    print(f"  Rows    : {len(features):,}")
    print(f"  Columns : {list(features.columns)}")

    # Save model and scaler
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  churn_model.joblib  → {MODEL_PATH}")
    print(f"  churn_scaler.joblib → {SCALER_PATH}\n")


# =============================================================
# Final summary
# =============================================================

def print_final_summary(metrics: dict,
                        churn_threshold: float,
                        apply_smote: bool) -> None:
    print("=" * 60)
    print("  FINAL SUMMARY — Churn Model")
    print("=" * 60)
    print(f"  Churn threshold (data-derived) : {churn_threshold:.0f} days")
    print(f"  SMOTE applied                  : {apply_smote}")
    print(f"  ROC-AUC                        : {metrics['roc_auc']}")
    print(f"  F1 Score                       : {metrics['f1_score']}")
    print(f"  Precision                      : {metrics['precision']}")
    print(f"  Recall                         : {metrics['recall']}")
    print(f"\n  Files saved:")
    print(f"    data/churn_labels.csv")
    print(f"    data/churn_model.joblib")
    print(f"    data/churn_scaler.joblib")
    print(f"    reports/plot_17_churn_model_eval.png")
    print(f"    reports/plot_18_shap_summary.png")
    print(f"    reports/plot_19_shap_waterfall.png")
    print(f"\n  Next step: open notebooks/03_cohort_churn.ipynb\n")


# =============================================================
# Main execution
# =============================================================

if __name__ == "__main__":
    print("\n  UCI Online Retail II — Churn Prediction Pipeline")
    print("  " + "─" * 56 + "\n")

    # Load
    txn, rfm = load_data()

    # Derive threshold
    churn_threshold = derive_churn_threshold(txn)

    # Feature engineering + churn labelling
    features = engineer_features(txn, rfm, churn_threshold)

    # Save churn labels before train/test split
    # (we want all customers, not just the test split)
    features.to_csv(OUT_LABELS, index=False)

    # Separate features and target
    X = features[FEATURE_COLS].copy()
    y = features["Churned"].copy()

    # Check balance + decide on SMOTE
    X, y, apply_smote, churn_rate = check_and_balance(X, y)

    # Split + scale + SMOTE
    X_train, X_test, y_train, y_test, scaler = split_and_resample(
        X, y, apply_smote
    )

    # Store test indices before scaling changed the array to numpy
    test_index = y_test.index

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    metrics, y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)

    # Plot evaluation charts
    plot_evaluation(
        model, X_test, y_test, y_pred, y_pred_prob,
        metrics, FEATURE_COLS
    )

    # SHAP plots
    plot_shap(model, X_test, FEATURE_COLS)

    # Save model + scaler + labelled CSV
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    save_outputs(features, y_pred, y_pred_prob, model, scaler, test_index)

    # Log to MLflow
    log_to_mlflow(
        model, metrics, churn_threshold,
        churn_rate, apply_smote, FEATURE_COLS
    )

    # Predict on ALL customers (for Streamlit dashboard)
    # Scale the full feature matrix using the fitted scaler
    X_all = features[FEATURE_COLS].copy()
    X_all_scaled = scaler.transform(X_all)

    features["Churn_Pred"] = model.predict(X_all_scaled)
    features["Churn_Prob"] = model.predict_proba(X_all_scaled)[:, 1].round(4)
    features["Split"] = "train"
    features.loc[test_index, "Split"] = "test"  # keep track of who was in test

    features.to_csv(OUT_LABELS, index=False)
    print(f"  churn_labels.csv updated with predictions for all {len(features):,} customers")

    print_final_summary(metrics, churn_threshold, apply_smote)
