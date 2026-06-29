# =============================================================
# src/cohorts.py
# Week 3 — Cohort Retention Analysis
# =============================================================
# What this script does:
#   1. Loads clean transaction data
#   2. Assigns every customer to an acquisition cohort
#      (the month they made their FIRST purchase)
#   3. Builds a cohort retention matrix — for each cohort,
#      what % of customers came back in months 1, 2, 3 ... N?
#   4. Saves the retention matrix to data/cohort_retention.csv
#   5. Plots and saves a heatmap to reports/
#   6. Prints a business summary
#
# Run from your project root:
#   python src/cohorts.py
#
# Key concept — what is a cohort?
#   A cohort is a group of customers who all made their FIRST
#   purchase in the same calendar month. By tracking each cohort
#   separately, we can answer: "Of the customers we acquired in
#   January 2010, how many were still buying 3 months later?"
#   This is the standard way to measure retention in e-commerce.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
IN_FILE  = ROOT / "data" / "clean_retail.csv"
OUT_FILE = ROOT / "data" / "cohort_retention.csv"
RPT_DIR  = ROOT / "reports"
RPT_DIR.mkdir(exist_ok=True)


# =============================================================
# STEP 1: Load data
# =============================================================

def load_data(filepath: Path) -> pd.DataFrame:
    print("=" * 58)
    print("  STEP 1: Loading clean transaction data")
    print("=" * 58)

    if not filepath.exists():
        raise FileNotFoundError(
            f"\n  Could not find: {filepath}"
            f"\n  Run src/ingest.py first."
        )

    df = pd.read_csv(filepath, parse_dates=["InvoiceDate"])

    print(f"  Loaded : {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Date range: {df['InvoiceDate'].min().date()} → "
          f"{df['InvoiceDate'].max().date()}\n")
    return df


# =============================================================
# STEP 2: Assign acquisition cohorts
# =============================================================

def assign_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign every customer to an acquisition cohort — the month
    they made their VERY FIRST purchase.

    Why month, not day?
    Day-level cohorts would give us ~730 tiny cohorts across 2 years.
    Monthly cohorts give us ~25 meaningful groups — large enough to
    see patterns, small enough to track individually.

    Example:
        Customer 12345 first bought on 2010-03-15 → cohort "2010-03"
        Even if they later buy on 2010-06-02, they always remain in
        cohort "2010-03" — the month they were first acquired.
    """
    print("=" * 58)
    print("  STEP 2: Assigning acquisition cohorts")
    print("=" * 58)

    # Convert InvoiceDate to period (year-month) for easy grouping
    # Period is like a string "2010-03" but with time-series arithmetic built in
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")

    # For each customer, find the month of their FIRST ever purchase
    # This becomes their "cohort" — they stay in it forever
    first_purchase = (
        df.groupby("Customer ID")["InvoiceMonth"]
        .min()
        .reset_index()
        .rename(columns={"InvoiceMonth": "CohortMonth"})
    )

    # Merge the cohort label back onto every transaction row
    # Now every row knows which cohort the customer belongs to
    df = df.merge(first_purchase, on="Customer ID", how="left")

    # Cohort period index: how many months after acquisition is this purchase?
    # Month 0 = the acquisition month itself (everyone is here, so 100%)
    # Month 1 = one month after acquisition
    # Month 6 = six months after acquisition (are they still buying?)
    df["CohortIndex"] = (
        (df["InvoiceMonth"] - df["CohortMonth"])
        .apply(lambda x: x.n)  # .n extracts the integer number of periods
    )

    n_cohorts = df["CohortMonth"].nunique()
    print(f"  Total cohorts identified : {n_cohorts}")
    print(f"  Cohort range: {df['CohortMonth'].min()} → "
          f"{df['CohortMonth'].max()}")
    print(f"  Max cohort index         : {df['CohortIndex'].max()} months")
    print(f"\n  Sample — first 5 rows showing cohort assignment:")
    sample_cols = ["Customer ID", "InvoiceDate", "InvoiceMonth",
                   "CohortMonth", "CohortIndex"]
    print(df[sample_cols].head().to_string(index=False))
    print()

    return df


# =============================================================
# STEP 3: Build cohort retention matrix
# =============================================================

def build_retention_matrix(df: pd.DataFrame) -> tuple:
    """
    Build two matrices:

    1. cohort_counts : number of unique customers active in each
                       cohort × cohort_index cell
    2. retention_pct : the same numbers expressed as % of the
                       cohort's original size (Month 0 count)

    The resulting matrix has:
        Rows    = cohort months (e.g. 2010-01, 2010-02 ...)
        Columns = months since acquisition (0, 1, 2 ... N)
        Values  = % of original cohort still purchasing

    Month 0 is always 100% by definition — every customer purchases
    at least once in their acquisition month.

    Why "unique customers", not transaction count?
    We want to know IF a customer bought, not HOW MUCH. Counting
    transactions would inflate cells for heavy buyers without telling
    us about retention rate.
    """
    print("=" * 58)
    print("  STEP 3: Building cohort retention matrix")
    print("=" * 58)

    # Count unique customers per cohort × cohort_index combination
    # This gives us the raw count matrix
    cohort_data = (
        df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
        .nunique()
        .reset_index()
        .rename(columns={"Customer ID": "CustomerCount"})
    )
    print(cohort_data)
    # Pivot to matrix format:
    #   rows    = CohortMonth
    #   columns = CohortIndex (0, 1, 2 ...)
    #   values  = CustomerCount
    cohort_counts = cohort_data.pivot_table(
        index="CohortMonth",
        columns="CohortIndex",
        values="CustomerCount"
    )
    print(cohort_counts)
    # Convert cohort months from Period to string for clean display/saving
    cohort_counts.index = cohort_counts.index.astype(str)

    # Extract cohort sizes (Month 0 = everyone in that cohort)
    # This is the denominator for all retention % calculations
    cohort_sizes = cohort_counts.iloc[:, 0]
    print(cohort_sizes)
    # Retention % = each cell divided by that cohort's Month 0 count
    # .div(cohort_sizes, axis=0) divides each row by its cohort size
    retention_pct = cohort_counts.div(cohort_sizes, axis=0).round(4) * 100
    print(retention_pct)
    print(f"  Matrix shape : {retention_pct.shape}  "
          f"({retention_pct.shape[0]} cohorts × "
          f"{retention_pct.shape[1]} months)")
    print(f"\n  Cohort sizes (Month 0 — acquisition counts):")
    for cohort, size in cohort_sizes.items():
        print(f"    {cohort} : {int(size):>5,} customers")
    print()

    return cohort_counts, retention_pct


# =============================================================
# STEP 4: Compute summary statistics
# =============================================================

def compute_summary(retention_pct: pd.DataFrame) -> dict:
    """
    Extract key retention metrics that will be:
    1. Printed to console as a business summary
    2. Logged to MLflow in the notebook (Week 3 MLflow run)

    Key metrics:
    - Month 1 retention: what % of customers buy again next month?
      Industry benchmark for e-commerce: 20-40% is healthy.
    - Month 3 retention: key indicator of medium-term loyalty.
    - Month 6 retention: who are the genuinely loyal customers?
    - Average retention curve: overall trajectory across all cohorts.
    """
    print("=" * 58)
    print("  STEP 4: Computing retention summary statistics")
    print("=" * 58)

    # Average retention rate at each month index across all cohorts
    # dropna=True so we only average cohorts that have data for that month
    avg_retention = retention_pct.mean(skipna=True)

    # Month 1 retention — most important single metric
    m1  = avg_retention.get(1,  np.nan)
    m3  = avg_retention.get(3,  np.nan)
    m6  = avg_retention.get(6,  np.nan)
    m12 = avg_retention.get(12, np.nan)

    # Best and worst performing cohorts by Month 1 retention
    if 1 in retention_pct.columns:
        m1_by_cohort = retention_pct[1].dropna().sort_values(ascending=False)
        best_cohort  = m1_by_cohort.index[0]
        worst_cohort = m1_by_cohort.index[-1]
        best_m1      = m1_by_cohort.iloc[0]
        worst_m1     = m1_by_cohort.iloc[-1]
    else:
        best_cohort = worst_cohort = "N/A"
        best_m1 = worst_m1 = np.nan

    summary = {
        "avg_m1_retention_pct" : round(m1,  2) if not np.isnan(m1)  else None,
        "avg_m3_retention_pct" : round(m3,  2) if not np.isnan(m3)  else None,
        "avg_m6_retention_pct" : round(m6,  2) if not np.isnan(m6)  else None,
        "avg_m12_retention_pct": round(m12, 2) if not np.isnan(m12) else None,
        "best_cohort_m1"       : best_cohort,
        "worst_cohort_m1"      : worst_cohort,
        "best_m1_pct"          : round(best_m1,  2),
        "worst_m1_pct"         : round(worst_m1, 2),
        "n_cohorts"            : len(retention_pct),
    }

    print(f"  Average retention by month (across all cohorts):")
    print(f"    Month  1 : {m1:>6.1f}%  (bought again next month)")
    print(f"    Month  3 : {m3:>6.1f}%  (still active after 3 months)")
    print(f"    Month  6 : {m6:>6.1f}%  (medium-term loyal)")
    print(f"    Month 12 : {m12:>6.1f}%  (long-term loyal)")
    print(f"\n  Best  cohort by Month 1 retention : "
          f"{best_cohort}  ({best_m1:.1f}%)")
    print(f"  Worst cohort by Month 1 retention : "
          f"{worst_cohort}  ({worst_m1:.1f}%)")

    # Interpret the Month 1 rate
    print(f"\n  Interpretation:")
    if m1 >= 40:
        print(f"  Month 1 retention of {m1:.1f}% is strong for e-commerce.")
    elif m1 >= 20:
        print(f"  Month 1 retention of {m1:.1f}% is typical for e-commerce.")
    else:
        print(f"  Month 1 retention of {m1:.1f}% is below typical e-commerce "
              f"benchmarks (20-40%). This is common for one-time gift/occasion"
              f" buyers but warrants a win-back campaign strategy.")
    print()

    return summary


# =============================================================
# STEP 5: Plot retention heatmap
# =============================================================

def plot_heatmap(retention_pct: pd.DataFrame,
                 cohort_counts: pd.DataFrame) -> None:
    """
    Plot the retention matrix as a heatmap.

    Colour scale: dark blue = high retention, light = low retention.
    Each cell shows the retention % rounded to 1 decimal.

    Design choices:
    - Cap columns at 12 months: beyond month 12, most cohorts have
      very few data points (the dataset only spans 2 years, so early
      cohorts have the most months of data while recent cohorts have
      almost none). Showing 24 months would make most of the right
      side of the chart mostly empty/NaN and hard to read.
    - Annotate with % values: makes it immediately readable without
      needing to interpret the colour scale.
    - Separate heatmap: the retention % heatmap (annotated) is more
      useful than the raw counts heatmap for communication purposes.
      The raw counts are saved in the CSV.
    """
    print("=" * 58)
    print("  STEP 5: Plotting retention heatmap")
    print("=" * 58)

    # Cap at 12 months for readability
    max_months  = min(12, retention_pct.shape[1])
    plot_data   = retention_pct.iloc[:, :max_months].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    fig.suptitle("Cohort Retention Analysis\nUCI Online Retail II  |  "
                 "Monthly Acquisition Cohorts",
                 fontsize=14, fontweight="500", y=1.01)

    # ── Top chart: Retention % heatmap ───────────────────────
    ax = axes[0]
    mask = plot_data.isnull()   # mask NaN cells (no data yet for that cohort/month)

    sns.heatmap(
        plot_data,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        linewidths=0.4,
        linecolor="white",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Retention %", "shrink": 0.6},
        annot_kws={"size": 8},
        ax=ax
    )
    ax.set_title("Retention Rate (% of cohort still purchasing)",
                 fontsize=12, pad=10)
    ax.set_xlabel("Months since first purchase", fontsize=10)
    ax.set_ylabel("Acquisition cohort (month)", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)

    # ── Bottom chart: Retention curves (line chart) ───────────
    ax2 = axes[1]

    # Plot a line per cohort — shows the decay curve visually
    # Only plot cohorts with at least 6 months of data to avoid
    # clutter from very recent cohorts with 1-2 data points
    colours = plt.cm.Blues(
        np.linspace(0.3, 0.9, len(plot_data))
    )

    for i, (cohort, row) in enumerate(plot_data.iterrows()):
        valid = row.dropna()
        if len(valid) >= 4:     # only plot cohorts with enough data
            ax2.plot(
                valid.index,
                valid.values,
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=colours[i],
                label=str(cohort),
                alpha=0.8
            )

    # Average retention curve — the most important line
    avg_curve = plot_data.mean(skipna=True)
    ax2.plot(
        avg_curve.index,
        avg_curve.values,
        marker="o",
        markersize=6,
        linewidth=3,
        color="#993C1D",
        label="Average (all cohorts)",
        zorder=5
    )

    ax2.set_title("Retention Curves by Cohort  (red = average)",
                  fontsize=12, pad=10)
    ax2.set_xlabel("Months since first purchase", fontsize=10)
    ax2.set_ylabel("Retention %", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")
    )
    ax2.set_xticks(range(max_months))
    ax2.legend(
        bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=7, title="Cohort", title_fontsize=8,
        ncol=1
    )
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = RPT_DIR / "plot_16_cohort_retention_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.show()
    print(f"  Saved → {out_path}\n")


# =============================================================
# STEP 6: Save outputs
# =============================================================

def save_outputs(retention_pct: pd.DataFrame,
                 cohort_counts: pd.DataFrame,
                 summary: dict) -> None:
    print("=" * 58)
    print("  STEP 6: Saving outputs")
    print("=" * 58)

    # Save retention % matrix as CSV
    # Index = cohort months (strings), columns = 0,1,2...N
    retention_pct.to_csv(OUT_FILE)
    size_kb = OUT_FILE.stat().st_size / 1000
    print(f"  cohort_retention.csv  → {OUT_FILE}")
    print(f"  File size : {size_kb:.1f} KB")
    print(f"  Shape     : {retention_pct.shape}  "
          f"(cohorts × months)")

    # Save raw counts matrix alongside for reference
    counts_path = ROOT / "data" / "cohort_counts.csv"
    cohort_counts.to_csv(counts_path)
    print(f"  cohort_counts.csv     → {counts_path}")

    # Save summary stats as a small CSV for MLflow logging in notebook
    summary_path = ROOT / "data" / "cohort_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  cohort_summary.csv    → {summary_path}\n")


# =============================================================
# STEP 7: Final business summary
# =============================================================

def print_final_summary(retention_pct: pd.DataFrame,
                        summary: dict) -> None:
    print("=" * 58)
    print("  FINAL SUMMARY — Cohort Retention")
    print("=" * 58)
    print(f"  Cohorts analysed          : {summary['n_cohorts']}")
    print(f"  Avg Month 1  retention    : {summary['avg_m1_retention_pct']}%")
    print(f"  Avg Month 3  retention    : {summary['avg_m3_retention_pct']}%")
    print(f"  Avg Month 6  retention    : {summary['avg_m6_retention_pct']}%")
    print(f"  Avg Month 12 retention    : {summary['avg_m12_retention_pct']}%")
    print(f"\n  Best cohort (Month 1)     : {summary['best_cohort_m1']} "
          f"({summary['best_m1_pct']}%)")
    print(f"  Worst cohort (Month 1)    : {summary['worst_cohort_m1']} "
          f"({summary['worst_m1_pct']}%)")

    print(f"\n  Retention matrix preview (first 5 cohorts, months 0-6):")
    preview_cols = [c for c in range(7) if c in retention_pct.columns]
    print(retention_pct.iloc[:5, preview_cols].round(1).to_string())

    print(f"\n  Files saved:")
    print(f"    data/cohort_retention.csv")
    print(f"    data/cohort_counts.csv")
    print(f"    data/cohort_summary.csv")
    print(f"    reports/plot_16_cohort_retention_heatmap.png")
    print(f"\n  Next step: run src/churn_model.py")
    print(f"  Then open notebooks/03_cohort_churn.ipynb\n")


# =============================================================
# Main execution
# =============================================================

if __name__ == "__main__":
    print("\n  UCI Online Retail II — Cohort Retention Analysis")
    print("  " + "─" * 54 + "\n")

    df                          = load_data(IN_FILE)
    df                          = assign_cohorts(df)
    cohort_counts, retention_pct = build_retention_matrix(df)
    summary                     = compute_summary(retention_pct)
    plot_heatmap(retention_pct, cohort_counts)
    save_outputs(retention_pct, cohort_counts, summary)
    print_final_summary(retention_pct, summary)
