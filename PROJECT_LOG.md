# Project Log — Customer Segmentation & Retention Analysis

> Paste this entire file at the start of a new Claude chat to resume work
> with full context. Update it at the end of every session.

## How to resume in a new chat
1. Open a new conversation
2. Paste this whole file
3. Say what you want to do next (e.g. "Continue with Week 3, src/cohorts.py")

---

## Project Setup
- **OS:** Windows, PyCharm + Python 3.11, venv at `venv\`
- **Dataset:** UCI Online Retail II (`data/online_retail_II.xlsx`, gitignored)
- **Repo:** github.com/aishwaryachanda21/customer_segmentation
- **Tracking:** MLflow with SQLite backend — `mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")`. NOT the default file store (deprecated, throws `MlflowException` in newer MLflow).
- **Folder structure:** `data/`, `notebooks/`, `src/`, `app/` (Week 4+), `reports/`, all per the README.

## Current Status: Week 2 complete, starting Week 3

## Files that exist and work
| File | Produces | Status |
|---|---|---|
| `src/ingest.py` | `data/clean_retail.csv` | ✓ done |
| `src/features.py` | `data/rfm_scores.csv` | ✓ done |
| `src/train.py` | `data/rfm_segments.csv`, `scaler.joblib`, `kmeans_model.joblib`, `vip_outliers.csv` | ✓ done |
| `notebooks/01_eda.ipynb` | 7 EDA plots in `reports/` | ✓ done |
| `notebooks/02_rfm_clustering.ipynb` | plots 09–15 in `reports/`, includes K-Medoids validation + segment crosstab | ✓ done |
| `README.md` | — | ✓ covers Week 1–2 |

## Key decisions made (don't re-litigate these without reason)
1. **Quintile scoring uses `.rank(method="first")` before `pd.qcut`**, NOT `duplicates="drop"`. Reason: ties in raw Frequency/Monetary caused duplicate bucket edges → NaN scores. Ranking first guarantees unique values → always exactly 5 clean buckets. Direction (normal vs inverted for Recency) is controlled purely by label order (`[1,2,3,4,5]` vs `[5,4,3,2,1]`), not by an `ascending`/`inverse` parameter.
2. **~39 outlier customers (bulk B2B, top ~1% by spend/frequency) are separated BEFORE K-Means**, saved to `data/vip_outliers.csv`. Reason: including them caused K-Means to dedicate entire clusters to absorbing their extreme values instead of segmenting the other ~98%. Validated this was the right call by running K-Medoids on the full dataset (including outliers) — it also isolated them into their own clusters.
3. **k=4 was chosen for K-Means, overriding the metric-optimal k=2** (silhouette 0.927 but a trivial bulk-vs-retail split with no business value). Elbow inflects at k=3. k=4 gives 4 interpretable segments at silhouette=0.504. This override is implemented explicitly inside `select_best_k(results, override_k=4)` with a printed justification — not a silent hardcoded value.
4. **Cluster naming uses SCALED cluster centre values, not raw dataset medians.** Raw medians were too blunt to distinguish "moderately above average" from "extremely above average" clusters (this caused 3 of 4 clusters to all get named "Champions" in an earlier bug).
5. **Two segment columns exist in `rfm_segments.csv` on purpose:** `Segment` (rule-based, per-customer quintile thresholds) and `KMeans_Segment` (model-based, per-cluster). They can disagree for boundary customers — this is expected, not a bug. `KMeans_Segment` is the primary one used going forward (Streamlit, Week 4+). Crosstab comparison lives in notebook Cell 10.
6. **Final 4 segments:** Loyal VIP, Champions, New / Promising, Lost / Inactive.

## Known environment gotchas (don't relitigate)
- Windows Git: `core.autocrlf true` needed once globally to stop CRLF/LF warning loops. They are harmless even when they appear, but commits still go through.
- `.gitignore` must be staged (`git add .gitignore`) BEFORE running `git add .` for the first time, or it has no effect on that commit.
- `scikit-learn-extra` (for KMedoids) has had numpy binary-incompatibility issues on this machine — fallback used: manual medoid-finding via `pairwise_distances` on top of plain sklearn `KMeans`, see Cell 10/11 of notebook 02.
- MLflow must use `sqlite:///mlflow.db` backend, not file store. Also: when running notebooks from inside `notebooks/`, make sure the tracking URI path resolves to project ROOT `mlflow.db`, not a duplicate one inside `notebooks/`.

## Next planned step
**Week 3 — Cohort retention analysis + churn prediction model**
- `src/cohorts.py` — acquisition cohorts, monthly retention matrix, heatmap
- `src/churn_model.py` — churn threshold derived from data (not assumed 90 days), feature engineering, XGBoost + SMOTE, MLflow logging, SHAP plots
- `notebooks/03_cohort_churn.ipynb`
- Watch for: the same outlier issue may resurface in churn features (e.g. avg days between orders for bulk accounts) — handle the same way (separate, don't delete) if it appears.

## Overall roadmap (5 weeks total)
```
✓ Week 1 — Data pipeline, EDA, MLflow setup
✓ Week 2 — RFM scoring, K-Means clustering, segment profiling
☐ Week 3 — Cohort retention analysis, churn prediction model
☐ Week 4 — Streamlit dashboard, model serving (load model from MLflow registry)
☐ Week 5 — Monitoring (Evidently AI drift reports), retraining trigger script
Post-Week 5 (optional): Docker, Streamlit Cloud deploy, Prefect scheduling, FastAPI endpoint
```
