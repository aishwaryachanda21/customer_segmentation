# Customer Segmentation & Retention Analysis

An end-to-end data science project analysing customer purchasing behaviour using RFM analysis and K-Means clustering, built on real UK e-commerce transaction data. This project demonstrates the complete lifecycle from raw data to a deployed, monitored model — covering data engineering, unsupervised learning, experiment tracking, and (in later phases) churn prediction, dashboarding, and MLOps.

## Project Status

```
✓  Week 1 — Data pipeline, EDA, MLflow setup
✓  Week 2 — RFM scoring, K-Means clustering, segment profiling
☐  Week 3 — Cohort retention analysis, churn prediction model
☐  Week 4 — Streamlit dashboard, model serving
☐  Week 5 — Monitoring, drift detection, retraining pipeline
```

## Dataset

**[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)** — 1.05M+ transaction line items from a UK-based online retailer, spanning December 2009 to December 2011. Contains invoice numbers, product descriptions, quantities, unit prices, customer IDs, and countries.

This dataset was selected over alternatives (Olist Brazilian E-Commerce, Telco Churn, Instacart) because it has a persistent Customer ID across orders, a genuine multi-year date range needed for cohort analysis, and realistic data quality issues (returns, nulls, cancellations) that demonstrate real-world cleaning work.

## Tech Stack

| Category | Tools |
|---|---|
| Data processing | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine learning | scikit-learn, scikit-learn-extra |
| Experiment tracking | MLflow (SQLite backend) |
| Model persistence | joblib |
| Environment | Python 3.11, venv |
| Planned (Week 3+) | XGBoost, imbalanced-learn, SHAP, Streamlit, Evidently AI |

## Project Structure

```
customer_segmentation/
├── data/                     # raw + processed data (gitignored)
│   ├── clean_retail.csv      # cleaned transaction data (Week 1)
│   ├── rfm_scores.csv        # RFM scores per customer (Week 2)
│   ├── rfm_segments.csv      # customers + cluster labels (Week 2)
│   ├── rfm_full.csv          # full dataset incl. outliers, scaled-ready
│   ├── vip_outliers.csv      # bulk B2B accounts separated pre-clustering
│   ├── scaler.joblib         # fitted StandardScaler
│   └── kmeans_model.joblib   # final K-Means model
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory data analysis
│   └── 02_rfm_clustering.ipynb  # RFM + cluster visualisation
├── src/
│   ├── ingest.py              # data loading & cleaning pipeline
│   ├── features.py            # RFM scoring & CLV computation
│   └── train.py                # K-Means clustering + MLflow tracking
├── reports/                   # saved plots and charts
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# Clone and enter the project
git clone https://github.com/<your-username>/customer_segmentation.git
cd customer_segmentation

# Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Download the dataset from [UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place `online_retail_II.xlsx` inside the `data/` folder.

## How to Run

```bash
# Week 1 — clean the raw data
python src/ingest.py

# Week 2 — compute RFM scores and CLV
python src/features.py

# Week 2 — run K-Means clustering with MLflow tracking
python src/train.py

# View experiment tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then open http://127.0.0.1:5000

# Explore results visually
jupyter lab
# open notebooks/01_eda.ipynb and notebooks/02_rfm_clustering.ipynb
```

---

## Week 1 — Data Pipeline & Exploratory Analysis

**Goal:** Load, clean, and understand the raw transaction data before any modelling.

### Cleaning steps (`src/ingest.py`)
- Dropped rows with missing Customer ID (anonymous sessions can't be tracked over time)
- Removed cancelled invoices (Invoice numbers starting with "C")
- Filtered out zero/negative quantities and prices
- Removed duplicate rows
- Engineered `TotalPrice`, `YearMonth`, `DayOfWeek`, `Hour` for downstream analysis

**Result:** ~1.05M raw rows reduced to a clean transaction table after removing roughly 20–25% of rows that were invalid for analysis purposes.

### Key EDA findings
- Revenue shows clear seasonality, peaking around November–December (holiday season)
- A small share of customers account for a disproportionate share of total orders (Pareto distribution)
- Ordering activity concentrates Tuesday–Thursday, 10am–3pm — consistent with a B2B/wholesale customer base
- The majority of customers are based in the UK; remaining international orders are concentrated in a handful of European countries

### MLflow setup
Switched from the deprecated file-based tracking backend to a SQLite backend (`sqlite:///mlflow.db`) following MLflow's recommended migration path. The first experiment run logged dataset-level statistics (row counts, customer counts, revenue, date range) as a baseline snapshot.

---

## Week 2 — RFM Scoring & Customer Segmentation

**Goal:** Score every customer on Recency, Frequency, and Monetary value, then use unsupervised clustering to discover natural customer segments.

### RFM computation (`src/features.py`)
- **Recency** — days since each customer's last purchase, relative to a fixed snapshot date (one day after the last transaction in the dataset)
- **Frequency** — count of unique invoices per customer
- **Monetary** — total amount spent per customer
- Each dimension converted to a 1–5 quintile score using **rank-based bucketing** (`series.rank(method="first")` before `pd.qcut`) to guarantee exactly 5 clean buckets with no NaN values, regardless of how many tied values exist in the raw data
- Combined scores computed three ways: concatenated string (e.g. `"554"`), simple sum (3–15), and a weighted score (Recency weighted highest at 40%, reflecting its strength as a predictor of future behaviour)
- Simple CLV estimated as `avg order value × annualised purchase frequency`, capped at the 99th percentile to control for extreme outliers

### Handling outliers
A small number of customers (~39, representing roughly the top 1% by spend and frequency) are bulk B2B wholesale accounts with order values and frequencies an order of magnitude above typical retail customers. These were identified and separated into `data/vip_outliers.csv` **before** clustering, because including them caused K-Means to dedicate entire clusters to absorbing their extreme values rather than meaningfully segmenting the remaining ~98% of customers.

### K-Means clustering (`src/train.py`)
- Features scaled using `StandardScaler` (required — Monetary values in the thousands would otherwise dominate distance calculations over Recency/Frequency)
- Tested k=2 through k=8, logging inertia, silhouette score, and cluster sizes for every run to MLflow
- **Model selection — domain override applied.** The silhouette score alone selected k=2 (0.576), but this reflects a trivial split between bulk buyers and retail customers with limited business utility. The elbow curve inflects at k=3. **k=4 was selected** to provide four interpretable, actionable customer segments, maintaining a respectable silhouette score of 0.504.
- Final segments named using scaled cluster centre thresholds (not raw medians, which were found to be unreliable for distinguishing moderately above-average clusters from extreme ones)

### Two segmentation methods, cross-validated

`rfm_segments.csv` contains two independent segment labels for every customer:

- **`Segment`** — rule-based, computed from each customer's individual 1–5 RFM quintile scores against fixed thresholds (e.g. R≥4 and F≥4 and M≥4 → Champions)
- **`KMeans_Segment`** — model-based, derived from K-Means clustering on continuous scaled RFM values, with each cluster named from its average centre

These two methods agree for the majority of customers, which supports the robustness of the segmentation. Where they disagree, it is typically for customers near a quintile boundary or near a cluster edge — these are genuinely ambiguous cases rather than errors in either method. A crosstab comparing the two (`reports/plot_15_segment_crosstab.png`) is included in the analysis notebook. **`KMeans_Segment` is treated as the primary segment** for downstream use (the Streamlit dashboard in Week 4), since it reflects overall behavioural similarity rather than independently-thresholded scores.

### Final segments

| Segment | Description |
|---|---|
| **Loyal VIP** | Very recent, very frequent, very high spend — top-tier customers |
| **Champions** | Recent, frequent, above-average spend — strong active customers |
| **New / Promising** | Recent but low frequency — recently acquired, not yet loyal |
| **Lost / Inactive** | High recency (long time since purchase), low frequency and spend |

### Validation — K-Medoids comparison
As a cross-check, K-Medoids clustering was run on the full dataset (including the separated outliers). The bulk B2B accounts consistently formed isolated, extreme clusters even under this different algorithm — confirming they represent a structurally distinct customer type rather than an artefact of K-Means' sensitivity to outliers. This validated the decision to separate them before primary segmentation.

### Outputs
- `data/rfm_segments.csv` — every customer with RFM values, scores, cluster assignment, segment name, and CLV
- `data/scaler.joblib`, `data/kmeans_model.joblib` — persisted for reuse in the Streamlit dashboard (Week 4)
- 7 experiment runs (k=2–8) plus one labelled final model run, all tracked in MLflow with logged parameters, metrics, and artifacts (evaluation charts, model files)

---

## Key Engineering Decisions

**Why rank-based quintile scoring instead of plain `pd.qcut`:** real RFM data has heavy ties (many customers with Frequency=1), which causes `pd.qcut` to produce duplicate bucket edges and NaN scores. Ranking values first (breaking ties by order of appearance) guarantees unique, evenly-distributed buckets every time.

**Why separate outliers rather than delete them:** the ~39 bulk accounts are legitimate, valuable customers — just a different customer type (wholesale vs retail). Deleting them would lose business value; clustering them together with retail customers distorts the segmentation. Separating them preserves both analyses.

**Why override the metric-optimal k:** optimising purely for silhouette score selected a mathematically clean but commercially uninformative 2-cluster split. Choosing k based on both statistical evidence (elbow inflection) and business interpretability (actionable segment count) reflects how segmentation decisions are made in practice.

## Next Steps (Week 3)

- Cohort retention analysis — track monthly retention by acquisition cohort
- Churn prediction model — XGBoost classifier with SHAP explainability
- Establish a data-driven churn threshold rather than an assumed default
