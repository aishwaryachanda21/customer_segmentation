# =============================================================
# src/ingest.py
# Week 1 — Data loading and cleaning pipeline
# =============================================================
# What this script does:
#   1. Loads both sheets from the raw Excel file
#   2. Cleans the data step by step (explained below)
#   3. Engineers useful new columns
#   4. Saves a clean CSV ready for all future work
#
# Run from your project root:
#   python src/ingest.py
# =============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
# Path() works on Windows, Mac, and Linux automatically.
# ".." means "go up one folder" — so from src/ we go up to the project root.
ROOT     = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "online_retail_II.xlsx"
OUT_FILE = ROOT / "data" / "clean_retail.csv"


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Load both Excel sheets and combine them into one DataFrame.
    The dataset comes with two sheets: 2009-2010 and 2010-2011.
    We want all the data in one place.
    """
    print("=" * 55)
    print("  STEP 1: Loading raw data")
    print("=" * 55)
    print(f"  File: {filepath}")
    print("  Loading sheet 1 (2009-2010) ... this takes ~30 seconds")

    df1 = pd.read_excel(filepath, sheet_name="Year 2009-2010", engine="openpyxl")
    print(f"  Sheet 1 loaded: {len(df1):,} rows")

    print("  Loading sheet 2 (2010-2011) ...")
    df2 = pd.read_excel(filepath, sheet_name="Year 2010-2011", engine="openpyxl")
    print(f"  Sheet 2 loaded: {len(df2):,} rows")

    df = pd.concat([df1, df2], ignore_index=True)
    print(f"\n  Combined total: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}\n")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a quick overview so we understand what we're working with.
    Always inspect before cleaning — never clean blindly.
    """
    print("=" * 55)
    print("  STEP 2: Inspecting raw data")
    print("=" * 55)

    print("\n  Data types:")
    print(df.dtypes.to_string())

    print("\n  Null value counts (and % of total rows):")
    nulls    = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(2)
    null_df  = pd.DataFrame({"null_count": nulls, "null_%": null_pct})
    print(null_df[null_df["null_count"] > 0].to_string())

    print("\n  Sample of first 3 rows:")
    print(df.head(3).to_string())
    print()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data. Each step removes a specific problem.
    We print the row count after each step so you can see the impact.
    """
    print("=" * 55)
    print("  STEP 3: Cleaning data")
    print("=" * 55)
    starting_rows = len(df)
    print(f"  Starting rows: {starting_rows:,}")

    # --- 3a. Drop rows with no Customer ID ---
    # These are anonymous/guest sessions. We can't track behaviour
    # over time without knowing WHO the customer is. They're useless
    # for segmentation, so we drop them.
    df = df.dropna(subset=["Customer ID"])
    print(f"\n  After dropping null Customer IDs : {len(df):,} rows"
          f"  (removed {starting_rows - len(df):,})")

    # --- 3b. Remove cancellations and returns ---
    # Cancelled invoices start with the letter 'C' (e.g. C536379).
    # These are refunds / reversals — not real purchases.
    before = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    print(f"  After removing cancellations     : {len(df):,} rows"
          f"  (removed {before - len(df):,})")

    # --- 3c. Remove rows with nonsensical quantities or prices ---
    # Quantity should be > 0 (negative = return we missed above).
    # Price should be > 0 (free or mispriced items distort analysis).
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    print(f"  After removing bad qty / price   : {len(df):,} rows"
          f"  (removed {before - len(df):,})")

    # --- 3d. Remove duplicate rows (safety check) ---
    before = len(df)
    df = df.drop_duplicates()
    print(f"  After dropping duplicates        : {len(df):,} rows"
          f"  (removed {before - len(df):,})")

    total_removed = starting_rows - len(df)
    pct_removed   = total_removed / starting_rows * 100
    print(f"\n  Total rows removed : {total_removed:,}  ({pct_removed:.1f}% of raw data)")
    print(f"  Clean rows kept    : {len(df):,}\n")

    return df.reset_index(drop=True)


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix column data types.
    Pandas sometimes reads numeric IDs as floats (e.g. 12345.0).
    Dates need to be datetime objects, not plain strings.
    """
    print("=" * 55)
    print("  STEP 4: Fixing data types")
    print("=" * 55)

    # Customer ID: convert float → int → string
    # We use string so IDs like "12345" don't accidentally get
    # used in arithmetic calculations.
    df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
    print("  Customer ID → string  ✓")

    # InvoiceDate: parse to proper datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    print("  InvoiceDate → datetime ✓")

    # Invoice: keep as string (some have letters)
    df["Invoice"] = df["Invoice"].astype(str)
    print("  Invoice → string ✓\n")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns that we'll need for RFM, cohort analysis,
    and the Streamlit dashboard.
    We do this once here so every downstream script gets them for free.
    """
    print("=" * 55)
    print("  STEP 5: Engineering features")
    print("=" * 55)

    # TotalPrice = how much was spent on each line item
    # This is the "Monetary" component of RFM
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    print("  TotalPrice = Quantity × Price  ✓")

    # YearMonth: e.g. "2010-12" — used for monthly aggregations
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    print("  YearMonth (YYYY-MM)  ✓")

    # Date parts for the order heatmap
    df["Year"]      = df["InvoiceDate"].dt.year
    df["Month"]     = df["InvoiceDate"].dt.month
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"]      = df["InvoiceDate"].dt.hour
    df["Date"]      = df["InvoiceDate"].dt.date
    print("  Year, Month, DayOfWeek, Hour, Date  ✓\n")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a business-level summary so you can do a sanity check
    before saving. These numbers should make intuitive sense.
    """
    print("=" * 55)
    print("  STEP 6: Final summary")
    print("=" * 55)

    invoice_rev = df.groupby("Invoice")["TotalPrice"].sum()

    print(f"  Rows (line items)    : {len(df):,}")
    print(f"  Unique customers     : {df['Customer ID'].nunique():,}")
    print(f"  Unique invoices      : {df['Invoice'].nunique():,}")
    print(f"  Unique products      : {df['StockCode'].nunique():,}")
    print(f"  Countries            : {df['Country'].nunique():,}")
    print(f"  Total revenue        : £{df['TotalPrice'].sum():,.0f}")
    print(f"  Median order value   : £{invoice_rev.median():.2f}")
    print(f"  Date range           : {df['InvoiceDate'].min().date()} → "
          f"{df['InvoiceDate'].max().date()}")
    print()


def save_clean_data(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save the clean DataFrame as a CSV.
    CSV loads ~20× faster than Excel, and works with every tool.
    We never modify this file — it is our single source of truth.
    """
    print("=" * 55)
    print("  STEP 7: Saving clean data")
    print("=" * 55)
    df.to_csv(filepath, index=False)
    size_mb = filepath.stat().st_size / 1_000_000
    print(f"  Saved → {filepath}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Rows: {len(df):,}\n")


# ── Main execution ────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  UCI Online Retail II — Data Ingestion Pipeline")
    print("  " + "─" * 51 + "\n")

    # Check the raw file exists before doing anything
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"\n  Could not find: {RAW_FILE}"
            f"\n  Make sure online_retail_II.xlsx is inside the data/ folder."
        )

    df = load_raw_data(RAW_FILE)
    inspect_data(df)
    df = clean_data(df)
    df = fix_types(df)
    df = engineer_features(df)
    print_summary(df)
    save_clean_data(df, OUT_FILE)

    print("  All done! Next step: open notebooks/01_eda.ipynb")
    print("  Run: jupyter lab\n")