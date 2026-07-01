# Design Document — Customer Segmentation & Retention Analysis

**Document purpose:** Technical reference covering architecture, data flow,
design decisions, and rationale for this project. Updated at the end of each
week as the build progresses. Distinct from `README.md` (portfolio-facing
narrative) and `PROJECT_LOG.md` (session-continuity notes for resuming work in
a new chat).

**Last updated:** End of Week 3 (Weeks 4-5 documented as planned architecture,
to be finalised as they are built)

---

## 1. Objective

Build an end-to-end data science project demonstrating customer segmentation and
retention analysis on real e-commerce transaction data, covering the full lifecycle:
data engineering → unsupervised learning → predictive modelling → dashboarding →
MLOps (experiment tracking, monitoring, retraining). Primary goal: portfolio /
resume artefact demonstrating practical, production-aware data science skills.

**Constraints:**
- 5-week timeline
- Solo build, intermediate Python/sklearn skill level
- Local development (Windows, PyCharm), Docker/cloud deployment deferred to post-Week-5
- No prior MLOps tooling experience — chosen stack must be low-friction to learn

---

## 2. Dataset Selection

**Chosen:** UCI Online Retail II (archive.ics.uci.edu/dataset/502)

**Alternatives considered and rejected:**

| Dataset | Rejected because |
|---|---|
| Olist Brazilian E-Commerce (Kaggle) | Customer IDs not reused across orders — no true repeat-customer tracking possible, breaks cohort/retention analysis |
| Telco Customer Churn (Kaggle/IBM) | Pre-labelled, no transaction history — usable for churn modelling only, not RFM/cohorts/CLV |
| Instacart Market Basket | No price data — cannot compute Monetary value or CLV |

**Decision driver:** need a single dataset supporting RFM, cohort retention, CLV,
and churn modelling without joining multiple sources. UCI Online Retail II has a
persistent Customer ID, a 2-year date range (Dec 2009 – Dec 2011), and realistic
data quality issues (cancellations, nulls, duplicates) that justify a real
cleaning pipeline rather than a toy one.

---

## 3. System Architecture

### 3.1 High-level pipeline (Weeks 1–2, current state)

```
online_retail_II.xlsx
        |
        v
  src/ingest.py        --> data/clean_retail.csv
        |
        v
  src/features.py      --> data/rfm_scores.csv
        |
        v
  src/train.py          --> data/rfm_segments.csv
        |                   data/vip_outliers.csv
        |                   data/scaler.joblib
        |                   data/kmeans_model.joblib
        |                   data/rfm_full.csv (full incl. outliers, for K-Medoids check)
        v
  MLflow (sqlite:///mlflow.db)  --> experiment runs, metrics, artifacts
        |
        v
  notebooks/01_eda.ipynb, 02_rfm_clustering.ipynb  --> reports/*.png, *.html
```

### 3.2 Pipeline extension — Week 3 (built)

```
data/clean_retail.csv --> src/cohorts.py --> data/cohort_retention.csv
                                              data/cohort_counts.csv
                                              data/cohort_summary.csv

data/clean_retail.csv -\
data/rfm_segments.csv --> src/churn_model.py --> data/churn_labels.csv
                                                   data/churn_model.joblib
                                                   data/churn_scaler.joblib
                                                   SHAP plots, ROC curve,
                                                   confusion matrix (reports/)
                          |
                          v
                    MLflow run: churn_model_xgboost
```

### 3.3 Pipeline extension — Weeks 4-5 (planned)

```
data/rfm_segments.csv ---\
data/churn_labels.csv -----> app/streamlit_app.py (Week 4)
data/cohort_retention.csv -/        |
                                     v
                          Tab 1: Segment Explorer
                            - filter by KMeans_Segment
                            - RFM scatter / radar charts
                          Tab 2: Retention Viewer
                            - interactive cohort heatmap
                          Tab 3: Churn Risk Predictor
                            - input sliders -> live XGBoost prediction
                            - SHAP waterfall explanation on demand
                                     |
                                     v
                  loads model via mlflow.pyfunc.load_model()
                  (Model Registry, stage="Production") -- planned;
                  current models loaded via joblib as an interim step

Week 5: Evidently AI --> drift reports (reports/*.html)
                          comparing 2009-2010 half (reference) vs
                          2010-2011 half (current) of the static dataset
                          as a simulated drift scenario
                     --> retrain trigger script --> re-runs train.py /
                                                      churn_model.py if
                                                      drift exceeds threshold
                     --> Streamlit "Monitoring" tab displaying the
                          latest drift report via st.components.html()
```

### 3.4 Folder structure

```
customer_segmentation/
├── data/                      # gitignored -- raw + all processed outputs
│   ├── clean_retail.csv
│   ├── rfm_scores.csv, rfm_segments.csv, rfm_full.csv, vip_outliers.csv
│   ├── scaler.joblib, kmeans_model.joblib
│   ├── cohort_retention.csv, cohort_counts.csv, cohort_summary.csv
│   ├── churn_labels.csv, churn_model.joblib, churn_scaler.joblib
│   └── (Week 5+) drift_reports/, retrain_log.csv
├── notebooks/                 # exploration, one notebook per week
│   ├── 01_eda.ipynb
│   ├── 02_rfm_clustering.ipynb
│   └── 03_cohort_churn.ipynb
├── src/                       # production-style scripts, importable modules
│   ├── ingest.py, features.py, train.py
│   ├── cohorts.py, churn_model.py
│   └── (Week 5+) monitor.py, retrain.py
├── app/                       # Streamlit dashboard (Week 4+)
├── reports/                   # all saved plots/charts (PNG + interactive HTML)
├── mlruns/ , mlflow.db        # gitignored -- MLflow tracking data
├── venv/                      # gitignored
├── requirements.txt
├── .gitignore
├── README.md                  # portfolio-facing narrative
├── PROJECT_LOG.md             # session-continuity working notes
└── DESIGN_DOC.md              # this file
```

**Design rationale:** `src/` scripts are pure Python (no notebook dependencies) so
they can be imported directly by `app/streamlit_app.py` later without duplicating
logic. Notebooks are exploration/visualisation layers on top of `src/` outputs,
never the source of truth for any data transformation.

---

## 4. Technology Stack & Rationale

| Layer | Choice | Why |
|---|---|---|
| Data processing | pandas, numpy | Standard, sufficient for ~1M row dataset |
| Visualisation | matplotlib, seaborn, plotly | matplotlib/seaborn for static report PNGs; plotly for the interactive 3D RFM chart (portfolio showpiece) |
| Clustering | scikit-learn (KMeans), scikit-learn-extra (KMedoids, validation only) | KMeans is fast and standard; KMedoids used only as a cross-check against outlier sensitivity, not primary method |
| Experiment tracking | MLflow, SQLite backend | Chosen over Weights & Biases (paid tiers, less beginner-friendly) and bare logging (no UI, no comparison tooling). SQLite backend specifically because MLflow's default file-store backend is deprecated and throws `MlflowException` as of the version installed |
| Model persistence | joblib | Standard for sklearn objects; faster than pickle for large numpy arrays (cluster centres) |
| Planned: Drift detection | Evidently AI | Chosen over building custom statistical tests; produces portfolio-ready HTML reports with minimal code |
| Planned: Churn model | XGBoost + imbalanced-learn (SMOTE) + SHAP | XGBoost for performance + SHAP compatibility; SMOTE because churn classes are expected to be imbalanced; SHAP for explainability |
| Planned: Dashboard | Streamlit | User's existing familiarity; fastest path to an interactive deployed demo |
| Explicitly deferred | Airflow / Prefect orchestration | Not needed until live/scheduled data ingestion exists; adds setup complexity disproportionate to a static historical dataset. May be added post-Week-5 for scheduling drift checks |

---

## 5. Data Pipeline Detail (Week 1)

### 5.1 Cleaning rules applied, in order, in `src/ingest.py`

1. Drop rows with null `Customer ID` -- anonymous transactions, untrackable over time, unusable for any per-customer analysis
2. Remove cancelled invoices (`Invoice` starting with `"C"`) -- refunds/reversals, not real purchases
3. Filter `Quantity <= 0` or `Price <= 0` -- data entry errors or returns not caught by the cancellation filter
4. Drop exact duplicate rows
5. Type fixes: `Customer ID` float to int to string (avoid accidental arithmetic on IDs); `InvoiceDate` to datetime
6. Feature engineering: `TotalPrice`, `YearMonth`, `Year`, `Month`, `DayOfWeek`, `Hour`, `Date`

**Result:** ~20-25% of raw rows removed; remainder is the single source of truth
for every downstream script (`data/clean_retail.csv`).

### 5.2 Why CSV, not re-reading Excel every time

Excel load takes 30-90 seconds for both sheets; CSV loads in ~2 seconds. Every
script after `ingest.py` reads only the clean CSV, never the raw Excel file.

---

## 6. RFM & Segmentation Detail (Week 2)

### 6.1 RFM computation (`src/features.py`)

- **Snapshot date** = one day after the last transaction in the dataset (not
  the actual current date) -- keeps recency calculations reproducible regardless
  of when the script is run
- **Recency** = days between snapshot date and each customer's most recent invoice
- **Frequency** = count of unique invoices per customer (not line items)
- **Monetary** = sum of `TotalPrice` per customer

### 6.2 Quintile scoring -- design evolution

**Problem encountered:** `pd.qcut` on raw RFM values failed/produced NaN scores
for a meaningful fraction of customers, due to heavy ties (many customers with
identical Frequency=1, for example) causing duplicate bucket edges.

**Rejected fix:** `duplicates="drop"` + NaN patching via
`fillna(3)`/`add_categories`. Works but is fragile, produces inconsistent bucket
counts depending on tie density, and obscures *why* NaNs occur.

**Adopted fix:** rank values first (`series.rank(method="first", ascending=True)`),
then `pd.qcut` the ranks. Since ranks are always unique integers, this guarantees
exactly 5 clean buckets with zero NaN values, regardless of how many ties exist
in the raw data. Direction (normal vs. inverted scoring for Recency) is controlled
purely by the **label list order** passed to `pd.qcut` (`[1,2,3,4,5]` vs.
`[5,4,3,2,1]`), not by a separate boolean flag -- this was a deliberate
simplification after an earlier version had a redundant/non-functional `inverse`
parameter that didn't actually change behaviour between branches.

```python
def quintile_score(series: pd.Series, labels: list) -> pd.Series:
    ranked = series.rank(method="first", ascending=True)
    scored = pd.qcut(ranked, q=5, labels=labels)
    return scored.astype(int)
```

### 6.3 CLV estimation

Simple historical model, not a probabilistic model (e.g. BG/NBD):
```
CLV = avg_order_value x annualised_purchase_frequency
avg_order_value = total_revenue / total_orders
annualised_purchase_frequency = (total_orders / lifespan_in_weeks) x 52
```
Capped at the 99th percentile to control extreme outlier distortion in
visualisations. Acknowledged limitation: this is a simplified, descriptive CLV,
appropriate for portfolio scope; a production system would use a probabilistic
lifetime-value model.

### 6.4 Outlier handling -- design evolution

**Problem encountered:** ~39 customers (top ~1% by Monetary/Frequency) are bulk
B2B wholesale accounts with values an order of magnitude above typical retail
customers (e.g. one cluster centre showed Monetary=£428,612 vs. dataset median
~£600). Including them in K-Means caused 2 of 4 clusters to be consumed entirely
by absorbing these extreme values, leaving only 2 clusters to represent the
remaining ~98% of customers -- defeating the purpose of segmentation.

**Adopted fix:** separate outliers (99th percentile cap on Monetary AND
Frequency) into `data/vip_outliers.csv` *before* scaling/clustering. They are
preserved as a distinct business-relevant table, not deleted.

**Validation step:** ran K-Medoids (more outlier-robust than K-Means in theory)
on the *full* dataset including outliers, to test whether the outliers would
still isolate themselves under a different algorithm. They did -- confirming
these customers represent a structurally distinct customer type, not merely an
artefact of K-Means' sensitivity to squared distances. This is documented as a
deliberate cross-validation step, not redundant analysis.

*(Implementation note: `scikit-learn-extra`'s `KMedoids` hit a numpy binary
incompatibility on the dev machine. Fallback: plain sklearn `KMeans` + manual
medoid identification via `pairwise_distances` to find the real customer closest
to each cluster centre -- achieves the same diagnostic purpose.)*

### 6.5 Choosing k -- design evolution

**Metric-only result:** silhouette score peaked at k=2 (0.927 initially; re-measured
differently after outlier removal changed the dataset composition).

**Why this was rejected despite being the "best" metric score:** inspecting the
cluster centres showed k=2 simply separates bulk-buyer-type customers from
everyone else -- a split that is mathematically clean but commercially
uninformative (two segments give no actionable retention strategy).

**Process used to select final k:**
1. Elbow curve (inertia vs. k) -- visually inflects at k=3
2. Silhouette scores for k>=3 (post-outlier-removal): k=3 -> 0.512, k=4 -> 0.504, k=5 -> 0.470
3. Business judgement: k=4 chosen for four cleanly interpretable, actionable
   segments (Loyal VIP, Champions, New/Promising, Lost/Inactive)

**Implementation:** the override is explicit and self-documenting --
`select_best_k(results, override_k=4)` -- printing both the metric-optimal k and
the override rationale at runtime, rather than a silent hardcoded reassignment.
This was a deliberate refactor after an earlier version used a bare
`best_k = 4` line with no audit trail.

**Considered but not implemented (documented for future reference):**
- `kneed` library for objective elbow-point detection (KneeLocator)
- Weighted combination scoring (normalised silhouette + normalised inertia drop)
- Gap statistic (compare real clustering vs. random-data baseline)

These were discussed as more rigorous alternatives to manual override but
judged unnecessary additional complexity for the current project stage. May be
revisited if the manual-override approach is challenged in an interview context
as insufficiently rigorous.

### 6.6 Cluster naming -- design evolution

**First attempt (rejected):** name clusters by comparing each cluster's *raw*
average R/F/M against *dataset-wide medians*. Failed because, after outlier
removal, multiple clusters were simultaneously above-median on all three
dimensions -- the binary above/below-median check couldn't distinguish a
"moderately good" cluster from an "extremely good" one. Result: 3 of 4 clusters
incorrectly named "Champions."

**Adopted fix:** name clusters using their **scaled cluster centre values**
directly (the same standardised space K-Means operates in), with thresholds at
roughly +/-0.3 standard deviations for "notably above/below average" and a
secondary check (`f > 1.5 or m > 1.5`) to separate extreme clusters ("Loyal VIP")
from solidly-positive ones ("Champions"). This correctly produced four distinct,
non-overlapping segment names from the actual k=4 model output.

### 6.7 Two parallel segmentation outputs

`rfm_segments.csv` retains both:
- `Segment` -- rule-based, computed independently per customer from quintile
  scores (`src/features.py`)
- `KMeans_Segment` -- model-based, computed from the K-Means clustering
  (`src/train.py`)

**Rationale:** having two independently-derived segmentations allows
cross-validation. High agreement between them is evidence the segmentation is
robust; disagreement identifies genuinely ambiguous boundary customers rather
than indicating an error in either method. A crosstab + heatmap comparison is
implemented in `notebooks/02_rfm_clustering.ipynb` (Cell 10).

**Decision:** `KMeans_Segment` is designated the primary segment for all
downstream use (Streamlit dashboard, Week 4+), since it reflects holistic
behavioural similarity in continuous space rather than independently-thresholded
discrete scores.

---

## 7. Cohort Retention & Churn Prediction Detail (Week 3)

### 7.1 Cohort assignment and retention matrix (`src/cohorts.py`)

- Every customer assigned to an acquisition cohort = the calendar month of
  their first-ever purchase (using pandas `Period("M")` arithmetic, which
  gives integer month-distance directly via subtraction -- no manual datetime
  delta conversion needed)
- Retention matrix built on **unique customer counts per cohort x month-index**,
  not transaction counts -- measures whether a customer purchased at all that
  month, not how much, which is the correct definition of retention
- Month 0 is always 100% by definition (every customer purchases in their own
  acquisition month)
- Heatmap capped at 12 months for readability, since the dataset only spans
  2 years and later cohorts have very few months of trailing data, making
  columns beyond month 12 mostly empty/NaN

**Result:** average retention of 21.2% (Month 1), 21.6% (Month 3), 17.8%
(Month 6), 18.2% (Month 12). The curve is notably **flat** rather than
steeply declining -- this single observation became the primary justification
for deriving the churn threshold empirically rather than assuming a retail
default (see 7.2). Best cohort: 2009-12 (35.3% M1) -- likely already-established
accounts captured at the start of the tracked window. Worst cohort: 2010-12
(9.2% M1) -- consistent with one-time December gift/seasonal buyers.

### 7.2 Churn threshold -- derived from data

**Rejected approach:** assume a standard 90-day retail churn window.

**Adopted approach:**
```
threshold = median(inter-purchase gap) + 1.5 x IQR(inter-purchase gap)
          = 106 days
```
Computed from the actual distribution of gaps between consecutive invoices
per customer (using the same Tukey's-fence convention as box-plot outlier
detection, applied here to define "abnormal silence" rather than to flag
outlier values). Directly motivated by the flat retention curve in 7.1 --
a fixed 90-day window would have misclassified many naturally long-cycle
B2B customers as churned.

### 7.3 Feature engineering

Features engineered beyond basic RFM, computed in `src/churn_model.py`:
- `avg_days_between_orders` -- mean gap between a customer's consecutive
  invoices; missing for single-purchase customers, filled with the churn
  threshold value as a conservative default (acknowledged limitation, see 7.6)
- `order_gap_std` -- standard deviation of those gaps; filled with 0 for
  single-purchase customers (no variation observed, even though this is a
  single data point, not genuine consistency)
- `product_diversity` -- count of unique `StockCode` values purchased
- `recency_trend` -- ratio of current silence to total customer lifespan;
  **excluded from the final model** (see 7.4)
- All gap-based features capped at the 99th percentile, consistent with the
  Week 2 outlier-handling convention

### 7.4 Target leakage -- detected and corrected

**Problem encountered:** first model trained with `Recency`, `R_Score`, and
`recency_trend` in the feature set achieved a perfect ROC-AUC of 1.0000.
Immediate red flag, since `Churned` is itself defined as `Recency > threshold`
-- the model had direct access to the variable used to construct its own
target, so it learned nothing beyond a lookup.

**Adopted fix:** removed all Recency-derived features (`Recency`, `R_Score`,
`recency_trend`) and `RFM_Total` (which embeds `R_Score`) from the training
feature set entirely.

**Considered but not adopted:** re-deriving a "safe" recency feature as
`silence_ratio = Recency / avg_days_between_orders` (a relative measure rather
than an absolute one). Tested empirically -- produced no change in ROC-AUC
(still 0.7637), because XGBoost as a tree-based model can already discover
this ratio internally from the two component features it already has access
to. Not included in the final feature set, since it added complexity without
measurable benefit.

**Final feature set (8 features, no leakage):** `Frequency`, `Monetary`,
`F_Score`, `M_Score`, `avg_days_between_orders`, `order_gap_std`,
`product_diversity`, `CLV_capped`.

### 7.5 Model training and evaluation

- Class balance checked empirically before deciding on SMOTE: 50.7% active /
  49.3% churned -- a near-even split that emerged naturally from the threshold
  derivation in 7.2 (placing the threshold at a statistical upper bound tends
  to split the population close to evenly). SMOTE was implemented but not
  triggered, since the imbalance threshold (minority class < 30%) was not
  crossed -- this was a deliberate empirical check, not a default-on choice.
- XGBoost classifier, hyperparameters chosen conservatively to limit
  overfitting on a small-to-medium dataset: `max_depth=4`, `n_estimators=200`,
  `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`
- Stratified train/test split (80/20) preserving the churn rate in both splits
- **Test set results:** ROC-AUC = 0.7637, F1 = 0.7228, Precision = 0.6844,
  Recall = 0.7657

**Why Recall was treated as more important than Precision:** a missed
churner (false negative) receives no win-back outreach and is lost
permanently; a false positive only costs one unnecessary marketing email.
The achieved Recall of 0.766 means roughly three in four actual churners are
correctly flagged.

### 7.6 SHAP explainability

- `TreeExplainer` used (fastest, most accurate option for tree-based models)
- **Summary plot:** Frequency is the strongest predictor (high frequency
  pushes away from churn); product_diversity is third-strongest (broad
  catalogue engagement pushes away from churn); Monetary shows a **mixed**
  directional signal -- high-spend customers split between loyal repeat
  accounts and one-time bulk buyers, and the model correctly treats them
  differently based on other features rather than spend alone
- **Waterfall plot** on the highest-risk customer (96.3% predicted probability,
  raw model output 3.272 in log-odds space) revealed a one-time large
  bulk-buyer profile: high CLV and Monetary, but low Frequency and narrow
  product diversity -- consistent with the business interpretation a domain
  expert would independently reach
- **Acknowledged limitation:** for single-purchase customers,
  `avg_days_between_orders` is filled with the churn threshold value (106),
  which can itself become a detectable pattern the model partially keys on.
  A cleaner future iteration would add an explicit `is_single_purchase`
  binary flag rather than imputing with the threshold.
- `F_Score` and `M_Score` showed near-zero SHAP contribution, since the model
  already has the more granular raw `Frequency` and `Monetary` values --
  flagged as candidates for removal in a future feature-set cleanup, not
  removed in the current version to preserve the rule-based segment
  comparison logic in `src/features.py` that also depends on these scores.

### 7.7 Full-dataset prediction fix

**Problem encountered:** initial `save_outputs()` implementation only wrote
predictions for the 20% test split into `churn_labels.csv`, leaving the 80%
training customers with null `Churn_Pred`/`Churn_Prob`. This is correct for
unbiased *evaluation* (training-set predictions are optimistic, since the
model has seen those customers) but insufficient for *downstream use* -- the
Week 4 Streamlit dashboard needs a prediction for every customer, not just
the test split.

**Adopted fix:** after evaluation, the trained model is re-applied to the
full feature matrix (`X_all`) to populate `Churn_Pred`/`Churn_Prob` for all
customers. A `Split` column (`"train"` / `"test"`) is added so that any
downstream analysis -- including the analysis notebook -- can still isolate
the held-out test set when computing honest performance metrics, while the
dashboard can use predictions for every customer.

### 7.8 Cross-validation against Week 2 segmentation

Churn rate computed per `KMeans_Segment` as an independent check that the
Week 2 unsupervised segmentation and the Week 3 supervised churn model are
internally consistent. Result: Loyal VIP and Champions show the lowest churn
rates; Lost/Inactive shows the highest -- the two analyses, built from
different methods and different target definitions, agree on the relative
risk ordering of customer segments. This cross-validation is treated as
meaningful evidence that both analyses capture genuine underlying customer
behaviour rather than artefacts of either method individually.

---

## 8. Experiment Tracking Detail

- **Backend:** `sqlite:///mlflow.db` at project root. Switched from the default
  file-store backend after hitting `MlflowException` (file store deprecated as
  of the installed MLflow version). `MLFLOW_ALLOW_FILE_STORE` env var was tested
  as a stopgap but SQLite was adopted as the permanent solution since it's the
  officially recommended migration path and adds no real overhead for a local
  single-user project.
- **Experiment name:** `"customer-segmentation"` (single experiment, multiple
  runs, used across Week 1 EDA logging, Week 2 clustering, and Week 3 churn
  modelling -- one experiment keeps the full project history comparable in one
  place rather than fragmenting it across multiple experiments)
- **Run naming convention:**
  - `kmeans_k{k}` for each k in the elbow sweep (Week 2)
  - `FINAL_MODEL_k{best_k}` for the selected production clustering model (Week 2)
  - `churn_model_xgboost` for the churn classifier (Week 3)
  - `Week3_Notebook_Summary` for notebook-level retention + evaluation metrics
    logged separately from the script-level run, so the notebook's view of the
    results is auditable independently of the script's
- **Logged per clustering run (Week 2):** params (`k`, `random_state`, `n_init`,
  `features`), metrics (`inertia`, `silhouette_score`, `n_customers`, per-cluster
  size), artifacts (final run only: model file, scaler, output CSV, chart)
- **Logged for churn model (Week 3):** params (`model_type`, `churn_threshold`,
  `features`, `test_size`, `smote_applied`, all XGBoost hyperparameters),
  metrics (`roc_auc`, `f1_score`, `precision`, `recall`, confusion matrix
  components `tp`/`fp`/`tn`/`fn`, `train_churn_rate`), artifacts (model file,
  scaler, labelled CSV, evaluation chart, both SHAP plots)
- **Known gotcha:** running a notebook from inside `notebooks/` can create a
  duplicate `mlruns/` folder relative to that working directory if the tracking
  URI isn't anchored to project ROOT -- must always resolve the SQLite path via
  `Path("..").resolve() / "mlflow.db"` or equivalent, not a relative path.
- **Planned for Week 5:** drift check runs and retraining-trigger runs will log
  to the same experiment, with run names following the pattern
  `drift_check_{date}` and `retrain_{date}`, so the full lifecycle of a model
  version -- training, evaluation, drift monitoring, retraining -- is traceable
  in one MLflow experiment.

---

## 9. Decision Log (chronological)

| # | Decision | Reason | Status |
|---|---|---|---|
| 1 | UCI Online Retail II over 3 alternatives | Only dataset supporting RFM + cohorts + CLV + churn from one source | Final |
| 2 | MLflow + Evidently over Airflow/full MLOps stack | Matches "new to MLOps" starting point; orchestration unnecessary for static dataset | Final for Weeks 1-5 |
| 3 | SQLite MLflow backend over file-store | File-store deprecated, throws exception on current MLflow version | Final |
| 4 | Rank-then-qcut over qcut+duplicates=drop | Eliminates NaN scores entirely, more robust to tie density | Final |
| 5 | Separate outliers before clustering, don't delete | Preserves business value of VIP accounts while enabling meaningful retail segmentation | Final |
| 6 | k=4 override over metric-optimal k=2 | k=2 is a trivial, commercially uninformative split | Final, documented with explicit override mechanism |
| 7 | Name clusters from scaled centres, not raw medians | Raw medians couldn't distinguish moderate vs. extreme clusters post-outlier-removal | Final |
| 8 | Keep both rule-based `Segment` and model-based `KMeans_Segment` | Enables cross-validation of segmentation robustness | Final; `KMeans_Segment` is primary |
| 9 | K-Medoids run as validation only, not primary clustering method | Confirms outlier separation was justified; not adopted as main method due to numpy compatibility issues + no clear superiority for this use case | Final |
| 10 | MCP integration deferred to optional post-Week-5 phase | Adds real value (conversational dashboard queries, retrain triggers) but risks scope creep on the 5-week core timeline; should not be added without a genuine use case | Deferred |
| 11 | Churn threshold derived empirically (106 days) over assumed 90-day default | Flat cohort retention curve indicated long natural repurchase cycles; a fixed retail default would misclassify normal B2B behaviour as churn | Final |
| 12 | Removed Recency, R_Score, recency_trend, RFM_Total from churn features | Initial model with these features scored a perfect ROC-AUC of 1.0 -- confirmed target leakage, since churn label is defined from Recency | Final |
| 13 | Did not add `silence_ratio` feature despite testing it | Tested empirically; produced no change in ROC-AUC (tree model already discovers the ratio internally from existing features) | Tested, not adopted |
| 14 | SMOTE implemented but not triggered | Churn classes emerged naturally balanced (50.7%/49.3%) from the threshold derivation; imbalance check is empirical, not assumed | Final |
| 15 | Full-dataset prediction pass added after initial test-only predictions | Streamlit dashboard (Week 4) needs predictions for all customers, not just the 20% test split; `Split` column preserves ability to isolate test set for honest metrics | Final |

---

## 10. Open Questions / Risks

### Resolved during Week 3 (previously open)
- ~~Churn threshold derivation~~ -- resolved, see 7.2 (106 days, data-derived)
- ~~Outlier risk recurrence in churn features~~ -- did not resurface as a
  blocking issue; gap-based features were capped at the 99th percentile as a
  precaution, consistent with the Week 2 approach, but no separate VIP
  exclusion was needed for the churn model specifically
- ~~Class imbalance~~ -- resolved empirically; classes emerged balanced,
  SMOTE implemented but not triggered (see 9, decision #14)

### Open going into Week 4
- **Model loading pattern:** current plan is `mlflow.pyfunc.load_model()`
  from the Model Registry with stage="Production", but models are currently
  only saved via `joblib.dump()`. Registering them formally in MLflow's Model
  Registry (via `mlflow.sklearn.log_model()` / `mlflow.xgboost.log_model()`
  with a registered model name) is a Week 4 prerequisite, not yet done.
- **Single-purchase customer imputation:** `avg_days_between_orders` filled
  with the churn threshold value for single-purchase customers is a known
  limitation (see 7.6). Whether to fix this with an explicit
  `is_single_purchase` flag before building the Streamlit predictor, or to
  accept it as a documented limitation, is undecided.
- **Feature redundancy:** `F_Score`, `M_Score` (near-zero SHAP contribution)
  and `CLV_capped` (highly correlated with Frequency x Monetary) are
  candidates for removal from the churn feature set in a cleanup pass. Not
  done yet because `F_Score`/`M_Score` are still used by the rule-based
  `Segment` column logic in `src/features.py` -- removing them from the churn
  model only (not from `features.py`) is the likely path if pursued.

### Open going into Week 5
- **Drift simulation strategy:** since the dataset is static (no live data
  feed), drift detection will be demonstrated by splitting the historical
  data temporally -- treating the 2009-2010 half as the "reference" window
  and the 2010-2011 half as "current" -- rather than monitoring genuinely
  live data. This is a deliberate simplification appropriate for portfolio
  scope; it should be described as a simulated drift scenario, not
  live production monitoring, in any documentation or interview discussion.
- **Retraining trigger granularity:** undecided whether the retrain trigger
  will re-run the full pipeline (ingest -> features -> train -> churn_model)
  or only the affected stage. Re-running the full pipeline is simpler to
  implement correctly; partial retraining is more realistic but adds
  complexity disproportionate to the portfolio scope.

---

## 11. Planned Design — Weeks 4-5

*This section documents the intended architecture before implementation, to
be revised with "as-built" detail (in the style of sections 5-7) once each
week is actually completed.*

### 11.1 Week 4 — Streamlit Dashboard & Model Serving

**Planned structure:** three tabs in a single `app/streamlit_app.py`.

- **Tab 1, Segment Explorer:** sidebar filter by `KMeans_Segment`; displays
  RFM scatter plot and radar chart for the selected segment(s); customer table
  with sortable columns. Reuses the plotting logic already built in
  `notebooks/02_rfm_clustering.ipynb`, refactored into reusable functions
  rather than duplicated.
- **Tab 2, Retention Viewer:** interactive cohort heatmap with a date-range
  selector; reuses `data/cohort_retention.csv` directly, no new computation.
- **Tab 3, Churn Risk Predictor:** input sliders for the 8 non-leaking churn
  features; on submission, loads the trained XGBoost model and displays the
  predicted churn probability plus a live SHAP waterfall explanation for that
  specific input -- directly extending the SHAP work from Week 3 into an
  interactive tool rather than a static notebook plot.

**Model loading approach (planned, not yet implemented):** register both the
K-Means and XGBoost models in MLflow's Model Registry
(`mlflow.sklearn.log_model(..., registered_model_name=...)` /
`mlflow.xgboost.log_model(...)`), promote the chosen versions to the
"Production" stage, and load them in the Streamlit app via
`mlflow.pyfunc.load_model("models:/{name}/Production")`. This mirrors a real
production serving pattern (swap models by changing the registry stage, not
by editing application code) rather than hardcoding a joblib file path.

**Risk to monitor:** the StandardScaler used for clustering (`scaler.joblib`)
and the one used for churn (`churn_scaler.joblib`) are different objects
fitted on different feature sets. The dashboard must apply the correct
scaler to the correct model's inputs -- a likely source of bugs if not kept
explicit in the app code.

### 11.2 Week 5 — Monitoring, Drift Detection, Retraining

**Planned structure:**
- `src/monitor.py` -- uses Evidently AI's `ColumnDriftReport` /
  `ClassificationPreset` to compare a reference window against a current
  window of the data (see 10, drift simulation strategy)
- `src/retrain.py` -- checks the latest drift report's summary metrics
  against a defined threshold; if exceeded, re-runs the relevant pipeline
  stage(s) and logs a new MLflow run tagged with the trigger reason
- A fourth Streamlit tab, "Monitoring," displaying the latest HTML drift
  report via `st.components.v1.html()`

**Explicitly out of scope for Week 5,** deferred to the post-Week-5 phase:
Docker containerisation, cloud deployment (Streamlit Cloud / AWS / GCP),
Prefect/Airflow scheduling of the retrain check, a FastAPI serving endpoint,
and MCP integration (see decision #10 in section 9).

### 11.3 Post-Week-5 (optional, deferred)

| Extension | Trigger to revisit |
|---|---|
| Docker + docker-compose | Once Streamlit + MLflow are both stable locally |
| Cloud deployment (Streamlit Cloud / Render) | After Docker, if a public demo link is wanted for the portfolio |
| Prefect/Airflow scheduling | Only if moving beyond a static historical dataset to genuinely live data |
| FastAPI serving endpoint | If programmatic (non-UI) access to predictions becomes a stated requirement |
| MCP server for conversational queries | Only with a genuine use case (e.g. "ask Claude which segment has highest churn risk this month") -- not for resume-listing purposes alone |

---

## 12. Changelog

- **End of Week 2:** Document created. Covered dataset selection, architecture
  through Week 2, full Week 1-2 design rationale and decision log.
- **End of Week 3:** Added section 7 (Cohort Retention & Churn Prediction
  Detail) covering cohort assignment, the empirically-derived 106-day churn
  threshold, feature engineering, the target-leakage incident and its fix,
  model training/evaluation results (ROC-AUC 0.7637), SHAP explainability
  findings, the full-dataset prediction fix, and cross-validation against
  Week 2 segments. Expanded section 8 (Experiment Tracking) to cover the
  churn model's MLflow logging. Added decisions #11-15 to the decision log.
  Resolved the three open risks carried over from Week 2. Added section 11
  documenting planned (not yet built) architecture for Weeks 4-5, to be
  converted to as-built detail as each week is completed. Renumbered all
  subsequent sections accordingly.
