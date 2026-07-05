# Customer Segmentation & Retention Analysis

An end-to-end data science project analysing customer purchasing behaviour using RFM analysis and K-Means clustering, built on real UK e-commerce transaction data. This project demonstrates the complete lifecycle from raw data to a deployed, monitored model — covering data engineering, unsupervised learning, experiment tracking, and (in later phases) churn prediction, dashboarding, and MLOps.

## Project Status

```
✓  Week 1 — Data pipeline, EDA, MLflow setup
✓  Week 2 — RFM scoring, K-Means clustering, segment profiling
✓  Week 3 — Cohort retention analysis, churn prediction model
✓  Week 4 — Streamlit dashboard, model serving
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
| Machine learning | scikit-learn, scikit-learn-extra, XGBoost, imbalanced-learn, SHAP |
| Experiment tracking | MLflow (SQLite backend) |
| Model persistence | joblib |
| Dashboard | Streamlit (tabs, plotly charts, SHAP waterfall integration) |
| Environment | Python 3.11, venv |
| Planned (Week 5) | Evidently AI (drift detection and HTML reports) |

## Project Structure

```
customer_segmentation/
├── data/                     # raw + processed data (gitignored)
│   ├── clean_retail.csv      # cleaned transaction data (Week 1)
│   ├── rfm_scores.csv        # RFM scores per customer (Week 2)
│   ├── rfm_segments.csv      # customers + cluster labels (Week 2)
│   ├── rfm_full.csv          # full dataset incl. outliers, scaled-ready
│   ├── vip_outliers.csv      # bulk B2B accounts separated pre-clustering
│   ├── scaler.joblib         # fitted StandardScaler (RFM)
│   ├── kmeans_model.joblib   # final K-Means model
│   ├── cohort_retention.csv  # monthly retention % matrix (Week 3)
│   ├── cohort_counts.csv     # raw cohort counts matrix (Week 3)
│   ├── cohort_summary.csv    # key retention metrics (Week 3)
│   ├── churn_labels.csv      # all customers + churn label + predictions (Week 3)
│   ├── churn_model.joblib    # trained XGBoost churn model (Week 3)
│   └── churn_scaler.joblib   # fitted StandardScaler (churn features)
├── notebooks/
│   ├── 01_eda.ipynb              # exploratory data analysis
│   ├── 02_rfm_clustering.ipynb   # RFM + cluster visualisation
│   └── 03_cohort_churn.ipynb     # cohort retention + churn model evaluation
├── src/
│   ├── ingest.py              # data loading & cleaning pipeline
│   ├── features.py            # RFM scoring & CLV computation
│   ├── train.py               # K-Means clustering + MLflow tracking
│   ├── cohorts.py             # cohort retention matrix
│   └── churn_model.py         # churn feature engineering + XGBoost + SHAP
├── app/
│   ├── __init__.py            # makes app/ a Python package
│   ├── dashboard_data.py      # cached data/model loading module
│   └── streamlit_app.py       # main Streamlit dashboard (3 tabs)
├── reports/                   # saved plots and charts
├── requirements.txt
├── .gitignore
├── README.md
├── PROJECT_LOG.md
└── DESIGN_DOC.md
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

# Week 3 — cohort retention analysis
python src/cohorts.py

# Week 3 — churn prediction model (XGBoost + SHAP)
python src/churn_model.py

# Week 4 — launch Streamlit dashboard
streamlit run app/streamlit_app.py
# then open http://localhost:8501

# View experiment tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then open http://127.0.0.1:5000

# Explore results visually
jupyter lab
# open notebooks/01_eda.ipynb, 02_rfm_clustering.ipynb, 03_cohort_churn.ipynb
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

## Week 3 — Cohort Retention Analysis & Churn Prediction

**Goal:** Measure how well the business retains customers over time, and build a predictive model that flags customers likely to churn before they go silent.

### Cohort retention analysis (`src/cohorts.py`)
- Every customer assigned to an acquisition cohort based on the calendar month of their first purchase
- A retention matrix built showing what % of each cohort was still purchasing in months 1, 2, 3 ... 12 after acquisition
- **Average retention:** Month 1 = 21.2%, Month 3 = 21.6%, Month 6 = 17.8%, Month 12 = 18.2%

The retention curve is notably **flat** rather than steeply declining — consistent with a B2B/wholesale customer base where repurchase cycles are long but regular, rather than impulse retail behaviour. The best-performing cohort (2009-12, 35.3% Month 1 retention) consists of customers acquired at the very start of the tracked period — likely already-established accounts. The worst-performing cohort (2010-12, 9.2%) reflects seasonal one-time gift purchasers, a typical December acquisition pattern with low follow-through.

### Churn threshold — derived from data, not assumed

Rather than using a standard 90-day retail default, the churn threshold was computed directly from the dataset's inter-purchase gap distribution:

```
threshold = median(inter-purchase gaps) + 1.5 × IQR(inter-purchase gaps)
          = 106 days
```

This is the same statistical convention used in box-plot outlier detection (Tukey's fence), applied here to define "abnormally long silence" rather than an arbitrary fixed window. The flat retention curve directly motivated this choice — a fixed 90-day window would have misclassified many naturally long-cycle B2B customers as churned.

### Feature engineering and target leakage

Initial feature set included `Recency`, `R_Score`, and `recency_trend` alongside engineered behavioural features (`avg_days_between_orders`, `order_gap_std`, `product_diversity`). The first model trained on this set achieved a perfect ROC-AUC of 1.0000 — an immediate red flag for target leakage, since the churn label is itself derived from Recency (`Churned = 1 if Recency > 106 days`). All Recency-derived features were removed from the training set.

**Final feature set:** `Frequency`, `Monetary`, `F_Score`, `M_Score`, `avg_days_between_orders`, `order_gap_std`, `product_diversity`, `CLV_capped` — eight behavioural features with no direct encoding of the target.

### Model — XGBoost with SHAP explainability

- Class balance checked empirically: 50.7% active / 49.3% churned — a near-even split that emerged naturally from the threshold derivation. SMOTE was evaluated but not applied, since the imbalance threshold (30%) was not crossed.
- XGBoost classifier trained with conservative hyperparameters (`max_depth=4`, `n_estimators=200`, `learning_rate=0.05`) to reduce overfitting risk
- **Test set performance:** ROC-AUC = 0.764, F1 = 0.723, Precision = 0.684, Recall = 0.766

Recall was prioritised over Precision by design — a missed churner (false negative) costs more than an unnecessary win-back email (false positive). The model catches 76.6% of all customers who actually churn.

### SHAP findings

- **Frequency** is the strongest predictor: high-frequency buyers are substantially less likely to churn regardless of spend level
- **Product diversity** is the third-strongest signal: customers engaging with a broad range of products are more invested in the retailer and harder to replace with a single competing supplier
- **Monetary has a mixed directional signal** — high spend does not reliably predict retention on its own. High-spend customers include both loyal repeat accounts and one-time bulk buyers; the model correctly distinguishes between them using the other behavioural features
- A SHAP waterfall analysis of the highest-risk customer (96.3% predicted churn probability) revealed a one-time large bulk-buyer profile: high CLV and Monetary, but low Frequency and narrow product diversity — exactly the pattern a domain expert would expect to flag

### Cross-validation against Week 2 segments

Churn rate was computed per `KMeans_Segment` to check whether the unsupervised segmentation and the supervised churn model tell a consistent story. They do: Loyal VIP and Champions show the lowest churn rates, while Lost/Inactive shows the highest — independent confirmation that both analyses are capturing the same underlying customer behaviour from different angles.

### Outputs
- `data/cohort_retention.csv`, `data/cohort_counts.csv`, `data/cohort_summary.csv`
- `data/churn_labels.csv` — every customer with engineered features, churn label, and model prediction (with a `Split` column marking train vs. test)
- `data/churn_model.joblib`, `data/churn_scaler.joblib`
- SHAP summary and waterfall plots, ROC curve, confusion matrix, and churn-by-segment chart, all in `reports/`

---

## Week 4 — Streamlit Dashboard & Model Serving

**Goal:** Surface all analysis from Weeks 1–3 in an interactive dashboard — making the segmentation, retention data, and churn model accessible without running a single script or opening a notebook.

### App structure (`app/`)

Two files — a data-loading module and the main app — kept deliberately separate so each does one thing:

- `app/dashboard_data.py` — loads all CSVs and model files using Streamlit's caching system (`@st.cache_data` for DataFrames, `@st.cache_resource` for model objects). All four data files and four model/scaler files are loaded once per session; every subsequent slider move or button click uses the in-memory cache rather than re-reading from disk.
- `app/streamlit_app.py` — the three-tab dashboard. Imports everything from `dashboard_data.py`.

### Tab 1 — Segment Explorer

A `st.multiselect` widget drives all three components simultaneously — a Plotly scatter plot (Recency vs Frequency, coloured by segment), a radar chart showing average R/F/M quintile scores per segment, and a sortable customer detail table with currency-formatted columns. Scatter values are capped at the 97th percentile to prevent the bulk-buyer outliers from compressing the main cluster into a tiny corner of the chart.

### Tab 2 — Retention Viewer

An interactive Plotly heatmap of the cohort retention matrix with hover tooltips showing exact percentages. A cohort selector dropdown highlights one cohort's retention curve against the grey-background average — best and worst cohort curves are immediately comparable. Three metric cards below the chart update dynamically when the selected cohort changes.

### Tab 3 — Churn Risk Predictor

Two modes toggled by `st.radio`:
- **Existing customer lookup** — a segment pre-filter narrows the dropdown to a manageable subset before the Customer ID selector, making it usable with 5,000+ customers. Displays the stored prediction from `churn_labels.csv` alongside a live re-prediction from the model.
- **Manual slider entry** — eight sliders (one per churn feature) with ranges derived from the actual data distribution (min to 99th percentile). Integer sliders for count-based features (Frequency, scores, product diversity), float sliders for continuous features.

Both modes feed into the same live prediction block: a Plotly gauge chart showing the churn probability with a colour-coded risk label (Very Low / Low / Medium / High Risk), a feature summary table, and an on-demand SHAP waterfall explanation. SHAP computation is placed behind a button rather than running on every slider move — each computation takes ~1–2 seconds and running it reactively would make the sliders feel laggy.

### Key design decisions made during Week 4

**joblib over MLflow Model Registry:** the Model Registry adds value when multiple model versions compete for a "Production" slot across a team. For a solo project with one chosen model per task, the registry adds ceremony without solving a real problem. Direct joblib loading is simpler, faster, and more honest — an interviewer who's used MLflow in production will immediately challenge unnecessary registry usage.

**Dual-scaler safety:** two separate StandardScaler objects exist — `scaler.joblib` fitted on RFM features (for K-Means) and `churn_scaler.joblib` fitted on the 8 churn features (for XGBoost). A type mismatch between them produces silently wrong predictions with no error raised. The `load_models()` function in `dashboard_data.py` names them explicitly (`models["rfm_scaler"]` vs `models["churn_scaler"]`) and `predict_churn_single()` uses `models["churn_scaler"]` by name — making misuse visible rather than hidden.

**`KMeans_Segment` availability:** the column is already present in `churn_labels.csv` (written by `src/churn_model.py` during the full-dataset prediction pass). `get_all_data()` includes a fallback merge with explicit `astype(str)` casting on both join keys in case the column is ever missing — preventing the silent all-NaN join that caused a `KeyError` during initial development.

**Tab label length:** Streamlit truncates tab labels that overflow the available width. Labels shortened to "Segments", "Retention", "Churn Risk" with a `white-space: nowrap` CSS rule to prevent future clipping.

### Outputs
- `app/__init__.py` — marks `app/` as a Python package for clean imports
- `app/dashboard_data.py` — cached loading module with helper functions
- `app/streamlit_app.py` — full three-tab dashboard

---

## Key Engineering Decisions

**Why rank-based quintile scoring instead of plain `pd.qcut`:** real RFM data has heavy ties (many customers with Frequency=1), which causes `pd.qcut` to produce duplicate bucket edges and NaN scores. Ranking values first (breaking ties by order of appearance) guarantees unique, evenly-distributed buckets every time.

**Why separate outliers rather than delete them:** the ~39 bulk accounts are legitimate, valuable customers — just a different customer type (wholesale vs retail). Deleting them would lose business value; clustering them together with retail customers distorts the segmentation. Separating them preserves both analyses.

**Why override the metric-optimal k:** optimising purely for silhouette score selected a mathematically clean but commercially uninformative 2-cluster split. Choosing k based on both statistical evidence (elbow inflection) and business interpretability (actionable segment count) reflects how segmentation decisions are made in practice.

**Why the churn threshold was derived empirically rather than assumed:** a standard 90-day retail default would have misclassified many naturally long-cycle B2B customers as churned, given this dataset's flat retention curve. Deriving the threshold from the actual inter-purchase gap distribution (106 days) keeps the definition grounded in observed behaviour rather than a generic industry assumption.

**Why Recency was excluded from the churn model despite being available:** an initial model trained with Recency-derived features achieved a perfect ROC-AUC of 1.0 — a clear sign of target leakage, since the churn label is itself defined by Recency. Removing it produced a more honest, generalisable model (ROC-AUC 0.764) that predicts churn from purchasing behaviour rather than directly reading the answer.

**Why joblib over MLflow Model Registry for serving:** the registry is valuable when multiple model versions compete for a production slot across a team. For a solo project with one chosen model per task, it adds setup ceremony without solving a real problem — and an interviewer who uses MLflow in production would immediately ask why it was needed. Direct joblib loading is simpler, transparent, and honest.

## Next Steps (Week 5)

- `src/monitor.py` — Evidently AI drift reports comparing 2009–2010 (reference) vs 2010–2011 (current) data as a simulated drift scenario
- `src/retrain.py` — automated retraining trigger when drift exceeds threshold, logging a new MLflow run
- Add a "Monitoring" tab to the Streamlit dashboard displaying the latest Evidently HTML report via `st.components.v1.html()`
