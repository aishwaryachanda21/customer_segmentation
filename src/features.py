# =============================================================
# src/features.py
# Week 2 — RFM Scoring & CLV Computation
# =============================================================
# What this script does:
#   1. Loads the clean transaction data (from Week 1)
#   2. Computes Recency, Frequency, Monetary per customer
#   3. Assigns quintile scores 1-5 for each dimension
#   4. Computes a combined RFM score
#   5. Estimates simple Customer Lifetime Value (CLV)
#   6. Saves everything to data/rfm_scores.csv
#
# Run from your project root:
#   python src/features.py
# =============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
IN_FILE  = ROOT / "data" / "clean_retail.csv"
OUT_FILE = ROOT / "data" / "rfm_scores.csv"


# =============================================================
# HELPER: quintile scoring via ranking
# =============================================================
# WHY RANK FIRST?
# pd.qcut needs unique cut-point values to build 5 clean buckets.
# Real data has ties — e.g. hundreds of customers with Frequency=1.
# Those ties cause duplicate cut points → broken buckets → NaN scores.
#
# The fix: convert raw values to ranks BEFORE cutting.
# method="first" assigns sequential unique integers to tied values
# by order of appearance — so ranks are ALWAYS unique. pd.qcut on
# unique ranks always produces exactly 5 clean buckets, no NaN ever.
#
# DIRECTION is controlled entirely by the label list, not by ascending:
#   labels=[1,2,3,4,5] → bucket 1 (lowest ranks) → score 1  (normal)
#   labels=[5,4,3,2,1] → bucket 1 (lowest ranks) → score 5  (inverse)
#
# We always rank ascending=True. For Recency, the label list [5,4,3,2,1]
# does the inversion: rank 1 (fewest days = best) → bucket 1 → score 5.

def quintile_score(series: pd.Series, labels: list) -> pd.Series:
    """
    Assign a 1-5 score to a numeric series using rank-based quintiles.

    Parameters:
        series : the column to score (e.g. df['Recency'])
        labels : list of 5 score values assigned low-rank to high-rank.
                 Pass [1,2,3,4,5] so higher raw values get higher scores.
                 Pass [5,4,3,2,1] so lower raw values get higher scores
                 (used for Recency — fewer days since purchase = better).

    Returns:
        A pandas Series of integer scores, no NaN values guaranteed.

    How it works — 10 customer example:
        Frequency : [1, 1, 1, 2, 3, 3, 5, 8, 10, 10]  (ties everywhere)
        Ranks     : [1, 2, 3, 4, 5, 6, 7, 8,  9, 10]  (always unique)
        qcut(q=5) : bucket1  bucket2  bucket3  bucket4  bucket5
                    ranks1-2 ranks3-4 ranks5-6 ranks7-8 ranks9-10
        labels    : [1,       2,       3,       4,       5      ]
        F_Score   : [1, 1,    2,  2,   3,  3,   4,  4,   5,  5 ] ✓

        Recency   : [5, 30, 60, 100, 150, 200, 300, 400, 600, 700]
        Ranks     : [1,  2,  3,   4,   5,   6,   7,   8,   9,  10]
        labels    : [5,       4,       3,       2,       1      ]
        R_Score   : [5, 5,    4,  4,   3,  3,   2,  2,   1,  1 ] ✓
                     ^ fewest days (best) → score 5
    """
    # Step 1: rank ascending=True always — direction set by label order
    # method="first" breaks ties by position → guaranteed unique ranks
    ranked = series.rank(method="first", ascending=True)

    # Step 2: cut ranks into 5 equal buckets, assign provided labels
    # No duplicates possible → no NaN possible → no patching needed
    scored = pd.qcut(ranked, q=5, labels=labels)

    return scored.astype(int)


# =============================================================
# STEP 1: Load clean data
# =============================================================

def load_data(filepath: Path) -> pd.DataFrame:
    print("=" * 55)
    print("  STEP 1: Loading clean data")
    print("=" * 55)

    df = pd.read_csv(filepath, parse_dates=["InvoiceDate"])
    print(f"  Loaded : {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Date range: {df['InvoiceDate'].min().date()} → "
          f"{df['InvoiceDate'].max().date()}\n")
    return df


# =============================================================
# STEP 2: Compute raw RFM values
# =============================================================

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw Recency, Frequency, and Monetary values.

    Snapshot date: the day AFTER the last transaction in the dataset.
    We use this as our "today" reference point so recency is calculated
    consistently regardless of when you run this script.

    Why one day after? So the most recent customer has Recency = 1,
    not 0. A recency of 0 would cause issues in CLV division later.
    """
    print("=" * 55)
    print("  STEP 2: Computing raw RFM values")
    print("=" * 55)

    # Snapshot date = day after the last invoice in the dataset
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"  Snapshot date (our 'today'): {snapshot_date.date()}")

    rfm = df.groupby("Customer ID").agg(
        # Recency: days between snapshot and customer's LAST purchase
        # .max() gives the most recent invoice date per customer
        Recency   = ("InvoiceDate",
                     lambda x: (snapshot_date - x.max()).days),

        # Frequency: count of UNIQUE invoices (not line items)
        # nunique() avoids counting multiple items in one order as separate orders
        Frequency = ("Invoice", "nunique"),

        # Monetary: total £ spent across all orders
        Monetary  = ("TotalPrice", "sum"),
    ).reset_index()

    print(f"\n  Customers processed: {len(rfm):,}")
    print(f"\n  RFM value ranges:")
    print(f"    Recency   — min: {rfm['Recency'].min()} days, "
          f"max: {rfm['Recency'].max()} days, "
          f"median: {rfm['Recency'].median():.0f} days")
    print(f"    Frequency — min: {rfm['Frequency'].min()}, "
          f"max: {rfm['Frequency'].max()}, "
          f"median: {rfm['Frequency'].median():.1f}")
    print(f"    Monetary  — min: £{rfm['Monetary'].min():.2f}, "
          f"max: £{rfm['Monetary'].max():,.2f}, "
          f"median: £{rfm['Monetary'].median():.2f}\n")

    return rfm, snapshot_date


# =============================================================
# STEP 3: Assign quintile scores 1-5
# =============================================================

def assign_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw R, F, M values into scores from 1 (worst) to 5 (best).

    Recency  : INVERTED — lower days = higher score (bought recently = good)
    Frequency: higher orders = higher score
    Monetary : higher spend = higher score

    Quintiles split customers into 5 equal-sized groups.
    Score 5 = top 20% of customers on that dimension.
    Score 1 = bottom 20% of customers on that dimension.
    """
    print("=" * 55)
    print("  STEP 3: Assigning quintile scores (1-5)")
    print("=" * 55)

    # Recency score: labels=[5,4,3,2,1] — lowest rank (fewest days) → score 5
    # Rank 1 = customer who bought most recently = best = should score 5
    # Reversed label list achieves this: bucket 1 (rank 1) → label[0] = 5
    rfm["R_Score"] = quintile_score(rfm["Recency"],   labels=[5, 4, 3, 2, 1])

    # Frequency score: labels=[1,2,3,4,5] — highest rank (most orders) → score 5
    rfm["F_Score"] = quintile_score(rfm["Frequency"], labels=[1, 2, 3, 4, 5])

    # Monetary score: labels=[1,2,3,4,5] — highest rank (most spend) → score 5
    rfm["M_Score"] = quintile_score(rfm["Monetary"],  labels=[1, 2, 3, 4, 5])

    print("  R_Score (recency, inverted)  ✓")
    print("  F_Score (frequency)          ✓")
    print("  M_Score (monetary)           ✓")

    # Score distribution check — each score should have ~20% of customers
    print("\n  R_Score distribution (should be roughly equal):")
    print("  ", rfm["R_Score"].value_counts().sort_index().to_dict())
    print("  F_Score distribution:")
    print("  ", rfm["F_Score"].value_counts().sort_index().to_dict())
    print("  M_Score distribution:")
    print("  ", rfm["M_Score"].value_counts().sort_index().to_dict())
    print()

    return rfm


# =============================================================
# STEP 4: Compute combined RFM score
# =============================================================

def compute_rfm_score(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Combine R, F, M scores into a single number two ways:

    1. RFM_Score (string): concatenates scores e.g. "555", "312", "111"
       Useful for rule-based segment lookup tables.

    2. RFM_Total (int): sum of R+F+M scores, range 3-15
       Useful for ranking customers overall. 15 = Champion, 3 = Lost.

    3. RFM_Weighted (float): weighted average giving more importance
       to Recency (most predictive of future behaviour) and Frequency.
       Weights: R=0.4, F=0.35, M=0.25
    """
    print("=" * 55)
    print("  STEP 4: Computing combined RFM scores")
    print("=" * 55)

    # String score e.g. "555" for a Champion
    rfm["RFM_Score"] = (rfm["R_Score"].astype(str) +
                        rfm["F_Score"].astype(str) +
                        rfm["M_Score"].astype(str))

    # Simple sum score (3 to 15)
    rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    # Weighted score (research shows R is most predictive)
    rfm["RFM_Weighted"] = (0.40 * rfm["R_Score"] +
                           0.35 * rfm["F_Score"] +
                           0.25 * rfm["M_Score"])

    print(f"  RFM_Score  (string e.g. '555') ✓")
    print(f"  RFM_Total  (sum, range 3–15)   ✓  "
          f"mean={rfm['RFM_Total'].mean():.1f}, "
          f"median={rfm['RFM_Total'].median():.1f}")
    print(f"  RFM_Weighted (weighted avg)    ✓  "
          f"mean={rfm['RFM_Weighted'].mean():.2f}\n")

    return rfm


# =============================================================
# STEP 5: Rule-based segment labels
# =============================================================

def assign_segment_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable segment names based on R and F scores.
    These are business-logic rules, NOT the K-Means clusters.

    We create them here so we have two segmentation approaches
    to compare in the notebook:
      - Rule-based (this function): interpretable, manual
      - K-Means (train.py): data-driven, automatic

    Both are useful. The rule-based labels help validate the
    K-Means clusters make intuitive sense.
    """
    print("=" * 55)
    print("  STEP 5: Assigning rule-based segment labels")
    print("=" * 55)

    def label_segment(row):
        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New Customers"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        elif r <= 2 and f >= 4:
            return "Cant Lose Them"
        elif r <= 2 and f <= 2:
            return "Lost"
        elif r >= 3 and f <= 2:
            return "Potential Loyalist"
        else:
            return "Needs Attention"

    rfm["Segment"] = rfm.apply(label_segment, axis=1)

    seg_counts = rfm["Segment"].value_counts()
    print("  Segment distribution:")
    for seg, count in seg_counts.items():
        pct = count / len(rfm) * 100
        print(f"    {seg:<22} : {count:>5,}  ({pct:.1f}%)")
    print()

    return rfm


# =============================================================
# STEP 6: Compute Customer Lifetime Value (CLV)
# =============================================================

def compute_clv(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate simple historical CLV per customer.

    Formula:
        CLV = Avg Order Value × Purchase Frequency per year

    Where:
        Avg Order Value       = total spend ÷ number of orders
        Purchase Frequency/yr = orders ÷ (customer lifespan in weeks ÷ 52)
        Customer lifespan     = days between first and last purchase

    This is a simplified model, not the probabilistic BG/NBD model
    used in industry. Choosing it here as it's appropriate for portfolio demonstration.
    Good enough for Week 2; we'll refine in the Streamlit dashboard.

    Note: customers with only 1 purchase get CLV = their order value
    (we can't estimate a repeat rate from a single data point).
    """
    print("=" * 55)
    print("  STEP 6: Computing CLV")
    print("=" * 55)

    # Compute per-customer aggregates we need for CLV
    clv_df = df.groupby("Customer ID").agg(
        first_purchase = ("InvoiceDate", "min"),
        last_purchase  = ("InvoiceDate", "max"),
        total_orders   = ("Invoice", "nunique"),
        total_revenue  = ("TotalPrice", "sum"),
    ).reset_index()

    # Average order value
    clv_df["avg_order_value"] = clv_df["total_revenue"] / clv_df["total_orders"]

    # Customer lifespan in weeks (minimum 1 week to avoid division by zero)
    clv_df["lifespan_weeks"] = (
        (clv_df["last_purchase"] - clv_df["first_purchase"]).dt.days / 7
    ).clip(lower=1)

    # Purchase frequency per year (annualised)
    # = (total orders / lifespan in weeks) × 52 weeks per year
    clv_df["purchase_freq_annual"] = (
        clv_df["total_orders"] / clv_df["lifespan_weeks"] * 52
    )

    # Simple CLV
    clv_df["CLV"] = (clv_df["avg_order_value"] *
                     clv_df["purchase_freq_annual"]).round(2)

    # Cap extreme outliers at 99th percentile (a few bulk B2B buyers
    # have astronomical CLV that would distort visualisations)
    clv_cap = clv_df["CLV"].quantile(0.99)
    clv_df["CLV_capped"] = clv_df["CLV"].clip(upper=clv_cap).round(2)

    # Merge CLV into RFM dataframe
    rfm = rfm.merge(
        clv_df[["Customer ID", "avg_order_value", "purchase_freq_annual",
                "lifespan_weeks", "CLV", "CLV_capped"]],
        on="Customer ID",
        how="left"
    )

    print(f"  CLV computed for {len(rfm):,} customers")
    print(f"  CLV range    : £{rfm['CLV'].min():.2f} → £{rfm['CLV'].max():,.2f}")
    print(f"  CLV median   : £{rfm['CLV'].median():.2f}")
    print(f"  CLV mean     : £{rfm['CLV'].mean():.2f}")
    print(f"  CLV cap (99p): £{clv_cap:,.2f}  (used for visualisations)\n")

    return rfm


# =============================================================
# STEP 7: Save and print final summary
# =============================================================

def save_and_summarise(rfm: pd.DataFrame, filepath: Path) -> None:
    print("=" * 55)
    print("  STEP 7: Saving rfm_scores.csv")
    print("=" * 55)

    rfm.to_csv(filepath, index=False)
    size_kb = filepath.stat().st_size / 1000
    print(f"  Saved → {filepath}")
    print(f"  File size : {size_kb:.1f} KB")
    print(f"  Rows      : {len(rfm):,}")
    print(f"  Columns   : {list(rfm.columns)}\n")

    print("=" * 55)
    print("  FINAL SUMMARY — RFM Scores")
    print("=" * 55)
    print(rfm[["Customer ID", "Recency", "Frequency", "Monetary",
               "R_Score", "F_Score", "M_Score",
               "RFM_Total", "Segment", "CLV_capped"]].head(10).to_string(index=False))

    print("\n  Top 5 Champions by CLV:")
    top5 = (rfm[rfm["Segment"] == "Champions"]
            .nlargest(5, "CLV_capped")
            [["Customer ID", "Recency", "Frequency", "Monetary", "CLV_capped"]])
    print(top5.to_string(index=False))

    print(f"\n  Average CLV by segment:")
    clv_by_seg = (rfm.groupby("Segment")["CLV_capped"]
                    .mean()
                    .sort_values(ascending=False)
                    .round(2))
    for seg, clv in clv_by_seg.items():
        print(f"    {seg:<22} : £{clv:,.2f}")

    print(f"\n  Next step: run src/train.py for K-Means clustering")
    print(f"  Then open notebooks/02_rfm_clustering.ipynb\n")


# =============================================================
# Main execution
# =============================================================

if __name__ == "__main__":
    print("\n  UCI Online Retail II — RFM Scoring & CLV Pipeline")
    print("  " + "─" * 51 + "\n")

    # Check input file exists
    if not IN_FILE.exists():
        raise FileNotFoundError(
            f"\n  Could not find: {IN_FILE}"
            f"\n  Make sure you have run src/ingest.py first."
        )

    df           = load_data(IN_FILE)
    rfm, snap    = compute_rfm(df)
    rfm          = assign_scores(rfm)
    rfm          = compute_rfm_score(rfm)
    rfm          = assign_segment_labels(rfm)
    rfm          = compute_clv(df, rfm)
    save_and_summarise(rfm, OUT_FILE)
