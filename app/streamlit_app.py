# =============================================================
# app/streamlit_app.py
# Week 4 — Customer Segmentation & Retention Dashboard
# =============================================================
# Run from your project root:
#   streamlit run app/streamlit_app.py
#
# Tabs:
#   1. Segment Explorer  — filter by segment, RFM scatter,
#                          radar chart, customer table
#   2. Retention Viewer  — interactive cohort heatmap,
#                          per-cohort curve selector
#   3. Churn Predictor   — existing customer lookup OR
#                          manual slider entry, live prediction,
#                          SHAP waterfall explanation
# =============================================================

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Add project root to sys.path ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dashboard_data import (
    CHURN_FEATURE_COLS,
    FEATURE_DISPLAY,
    SEGMENT_COLOURS,
    get_churn_risk_label,
    get_feature_ranges,
    load_churn_data,
    load_cohort_retention,
    load_models,
    load_segments,
    predict_churn_single,
)

# =============================================================
# PAGE CONFIG — must be the first Streamlit call
# =============================================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# CUSTOM CSS
# =============================================================
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    [data-testid="metric-container"] {
        background: rgba(24, 95, 165, 0.06);
        border: 0.5px solid rgba(24, 95, 165, 0.15);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] > div { white-space: nowrap; }
    section[data-testid="stSidebar"] { font-size: 0.9rem; }
    hr { border-color: rgba(128, 128, 128, 0.15); }
</style>
""", unsafe_allow_html=True)


# =============================================================
# DATA LOADING
# =============================================================
# @st.cache_data: loads DataFrames once per session, returns
#   cached copy on every subsequent Streamlit re-run.
# @st.cache_resource: loads model objects once, shares the
#   same instance across all re-runs (safe for read-only inference).

@st.cache_data
def get_all_data():
    """Load all CSVs. KMeans_Segment is already in churn_labels.csv."""
    seg_df       = load_segments()
    retention_df = load_cohort_retention()
    churn_df     = load_churn_data()

    # KMeans_Segment is saved into churn_labels.csv by src/churn_model.py.
    # If it's missing, the user needs to re-run that script.
    if "KMeans_Segment" not in churn_df.columns:
        # Try a fallback merge before giving up
        seg_df["Customer ID"]   = seg_df["Customer ID"].astype(str)
        churn_df["Customer ID"] = churn_df["Customer ID"].astype(str)
        churn_df = churn_df.merge(
            seg_df[["Customer ID", "KMeans_Segment"]],
            on="Customer ID",
            how="left"
        )

    return seg_df, retention_df, churn_df


# Catch ALL exceptions so DATA_OK is always defined
# regardless of what goes wrong during loading.
DATA_OK    = False
DATA_ERROR = ""
seg_df = retention_df = churn_df = models = feat_ranges = None

try:
    seg_df, retention_df, churn_df = get_all_data()
    models      = load_models()
    feat_ranges = get_feature_ranges(churn_df)
    DATA_OK     = True
except Exception as e:
    DATA_ERROR = str(e)


# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.title("📊 Customer Analytics")
    st.caption("UCI Online Retail II · UK E-Commerce · 2009–2011")
    st.divider()

    if DATA_OK:
        total_customers = len(seg_df)
        n_segments      = seg_df["KMeans_Segment"].nunique()
        churn_rate      = churn_df["Churned"].mean() * 100
        avg_clv         = seg_df["CLV_capped"].mean()

        st.markdown("**Dataset overview**")
        col1, col2 = st.columns(2)
        col1.metric("Customers",  f"{total_customers:,}")
        col2.metric("Segments",   n_segments)
        col1.metric("Churn rate", f"{churn_rate:.1f}%")
        col2.metric("Avg CLV",    f"£{avg_clv:,.0f}")

        st.divider()
        st.markdown("**Customers per segment**")
        seg_counts = (
            seg_df["KMeans_Segment"]
            .value_counts()
            .reset_index()
            .rename(columns={"KMeans_Segment": "Segment",
                             "count": "Customers"})
        )
        for _, row in seg_counts.iterrows():
            colour = SEGMENT_COLOURS.get(row["Segment"], "#888882")
            pct    = row["Customers"] / total_customers * 100
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'margin-bottom:4px;">'
                f'<span style="color:{colour};font-weight:500;">'
                f'{row["Segment"]}</span>'
                f'<span style="color:grey;">{row["Customers"]:,} '
                f'({pct:.1f}%)</span></div>',
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown("**Model performance**")
        st.markdown("XGBoost churn classifier")
        st.markdown("ROC-AUC **0.764** · Recall **0.766**")
        st.markdown("Threshold: **106 days** (data-derived)")

    else:
        st.error(f"Data load error:\n\n{DATA_ERROR}")

    st.divider()
    st.caption("github.com/aishwaryachanda21/customer_segmentation")


# =============================================================
# STOP if data failed to load
# =============================================================
if not DATA_OK:
    st.error(
        "Could not load data. Please run the pipeline scripts first:\n\n"
        "```\n"
        "python src/ingest.py\n"
        "python src/features.py\n"
        "python src/train.py\n"
        "python src/cohorts.py\n"
        "python src/churn_model.py\n"
        "```\n\n"
        f"Error detail: {DATA_ERROR}"
    )
    st.stop()


# =============================================================
# TABS
# =============================================================
tab1, tab2, tab3 = st.tabs([
    "🗂️ Segments",
    "📈 Retention",
    "⚠️ Churn Risk",
])


# =============================================================
# TAB 1 — SEGMENT EXPLORER
# =============================================================
with tab1:
    st.header("Customer Segment Explorer")
    st.caption(
        "K-Means clustering on RFM features (k=4, outliers separated). "
        "Filter by segment to explore customer profiles."
    )

    all_segments  = sorted(seg_df["KMeans_Segment"].dropna().unique())
    selected_segs = st.multiselect(
        "Select segments to display",
        options=all_segments,
        default=all_segments,
        help="Choose one or more segments. All shown by default."
    )

    if not selected_segs:
        st.warning("Select at least one segment to see charts.")
        st.stop()

    filtered = seg_df[seg_df["KMeans_Segment"].isin(selected_segs)].copy()
    st.caption(f"Showing **{len(filtered):,}** of **{len(seg_df):,}** customers")
    st.divider()

    # ── Scatter + Radar ───────────────────────────────────────
    col_scatter, col_radar = st.columns([3, 2], gap="large")

    with col_scatter:
        st.subheader("Recency vs Frequency")
        st.caption(
            "Each dot is a customer. Colour = segment. "
            "Well-separated clusters confirm meaningful segmentation."
        )
        plot_s = filtered.copy()
        plot_s["Recency"]   = plot_s["Recency"].clip(
            upper=seg_df["Recency"].quantile(0.97))
        plot_s["Frequency"] = plot_s["Frequency"].clip(
            upper=seg_df["Frequency"].quantile(0.97))

        fig_scatter = px.scatter(
            plot_s,
            x="Recency",
            y="Frequency",
            color="KMeans_Segment",
            color_discrete_map=SEGMENT_COLOURS,
            hover_data={
                "Customer ID"    : True,
                "Monetary"       : ":,.0f",
                "CLV_capped"     : ":,.0f",
                "RFM_Total"      : True,
                "KMeans_Segment" : True,
            },
            labels={
                "Recency"        : "Recency (days since last purchase)",
                "Frequency"      : "Frequency (number of orders)",
                "KMeans_Segment" : "Segment",
            },
            opacity=0.55,
            height=400,
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(
            legend_title_text="Segment",
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_scatter)

    with col_radar:
        st.subheader("Segment RFM Fingerprints")
        st.caption(
            "Average quintile scores (1–5) per segment. "
            "Larger area = stronger across all RFM dimensions."
        )
        radar_cols   = ["R_Score", "F_Score", "M_Score"]
        radar_labels = ["Recency Score", "Frequency Score", "Monetary Score"]
        seg_means    = (
            filtered.groupby("KMeans_Segment")[radar_cols]
            .mean()
            .reset_index()
        )

        fig_radar = go.Figure()
        for _, row in seg_means.iterrows():
            seg  = row["KMeans_Segment"]
            vals = [row[c] for c in radar_cols] + [row[radar_cols[0]]]
            lbls = radar_labels + [radar_labels[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=lbls,
                fill="toself",
                name=seg,
                line_color=SEGMENT_COLOURS.get(seg, "#888882"),
                opacity=0.65,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5],
                                       tickvals=[1,2,3,4,5])),
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.1),
            margin=dict(l=30, r=30, t=10, b=40),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar)

    st.divider()

    # ── Segment profile summary ───────────────────────────────
    st.subheader("Segment Profiles")
    profile = (
        filtered.groupby("KMeans_Segment")
        .agg(
            Customers    = ("Customer ID", "count"),
            Avg_Recency  = ("Recency",     "mean"),
            Avg_Frequency= ("Frequency",   "mean"),
            Avg_Monetary = ("Monetary",    "mean"),
            Avg_RFM_Total= ("RFM_Total",   "mean"),
            Avg_CLV      = ("CLV_capped",  "mean"),
        )
        .round(1)
        .reset_index()
        .sort_values("Customers", ascending=False)
    )
    profile.columns = [
        "Segment", "Customers", "Avg Recency (days)",
        "Avg Frequency", "Avg Monetary (£)",
        "Avg RFM Total", "Avg CLV (£)"
    ]
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.divider()

    # ── Customer detail table ─────────────────────────────────
    st.subheader("Customer Detail Table")
    st.caption("Click any column header to sort.")
    display_cols = [c for c in [
        "Customer ID", "KMeans_Segment", "Recency", "Frequency",
        "Monetary", "R_Score", "F_Score", "M_Score",
        "RFM_Total", "CLV_capped"
    ] if c in filtered.columns]

    st.dataframe(
        filtered[display_cols]
        .sort_values("Monetary", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=350,
        column_config={
            "Customer ID"    : st.column_config.TextColumn("Customer ID"),
            "KMeans_Segment" : st.column_config.TextColumn("Segment"),
            "Recency"        : st.column_config.NumberColumn("Recency (days)", format="%d"),
            "Frequency"      : st.column_config.NumberColumn("Frequency", format="%d"),
            "Monetary"       : st.column_config.NumberColumn("Monetary (£)", format="£%.0f"),
            "CLV_capped"     : st.column_config.NumberColumn("CLV (£)", format="£%.0f"),
        },
    )


# =============================================================
# TAB 2 — RETENTION VIEWER
# =============================================================
with tab2:
    st.header("Cohort Retention Viewer")
    st.caption(
        "Each row = customers who first purchased in that month. "
        "Each cell = % of that cohort still purchasing N months later."
    )

    max_months  = min(12, retention_df.shape[1])
    plot_ret    = retention_df.iloc[:, :max_months].copy()
    all_cohorts = plot_ret.index.tolist()

    # ── Heatmap ───────────────────────────────────────────────
    st.subheader("Retention Heatmap")
    hover_text = []
    for cohort in plot_ret.index:
        row_text = []
        for month in plot_ret.columns:
            val = plot_ret.loc[cohort, month]
            row_text.append(
                "No data yet" if pd.isna(val)
                else f"Cohort: {cohort}<br>Month {month}: {val:.1f}% retained"
            )
        hover_text.append(row_text)

    fig_heat = go.Figure(data=go.Heatmap(
        z=plot_ret.values,
        x=[f"Month {c}" for c in plot_ret.columns],
        y=plot_ret.index.tolist(),
        text=[[f"{v:.1f}%" if not pd.isna(v) else ""
               for v in row] for row in plot_ret.values],
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        colorscale="Blues",
        zmin=0, zmax=100,
        colorbar=dict(title="Retention %", ticksuffix="%"),
    ))
    fig_heat.update_layout(
        xaxis_title="Months since first purchase",
        yaxis_title="Acquisition cohort",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
    )
    st.plotly_chart(fig_heat)

    st.divider()

    # ── Cohort selector + curves ──────────────────────────────
    st.subheader("Cohort Retention Curves")
    selected_cohort = st.selectbox(
        "Select acquisition cohort to highlight",
        options=all_cohorts,
        index=0,
    )

    fig_curve = go.Figure()

    # All cohorts — light grey background lines
    for cohort in all_cohorts:
        row = plot_ret.loc[cohort].dropna()
        if len(row) < 2:
            continue
        fig_curve.add_trace(go.Scatter(
            x=row.index.tolist(), y=row.values.tolist(),
            mode="lines", name=cohort,
            line=dict(color="rgba(150,150,150,0.25)", width=1),
            showlegend=False,
            hovertemplate=f"{cohort}: %{{y:.1f}}%<extra></extra>",
        ))

    # Average curve
    avg_curve = plot_ret.mean(skipna=True)
    fig_curve.add_trace(go.Scatter(
        x=avg_curve.index.tolist(), y=avg_curve.values.tolist(),
        mode="lines+markers", name="Average (all cohorts)",
        line=dict(color="#185FA5", width=2.5),
        marker=dict(size=6),
    ))

    # Selected cohort — highlighted
    sel_row = plot_ret.loc[selected_cohort].dropna()
    fig_curve.add_trace(go.Scatter(
        x=sel_row.index.tolist(), y=sel_row.values.tolist(),
        mode="lines+markers", name=f"Selected: {selected_cohort}",
        line=dict(color="#993C1D", width=3, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))

    fig_curve.update_layout(
        xaxis_title="Months since first purchase",
        yaxis_title="Retention %",
        yaxis=dict(ticksuffix="%", range=[0, 105]),
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=0, r=0, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    st.plotly_chart(fig_curve)

    # Cohort metrics
    if 1 in sel_row.index:
        avg_m1 = avg_curve.get(1, float("nan"))
        delta  = sel_row[1] - avg_m1
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{selected_cohort} — Month 1",
                  f"{sel_row[1]:.1f}%",
                  delta=f"{delta:+.1f}% vs avg")
        if 3 in sel_row.index:
            c2.metric("Month 3", f"{sel_row[3]:.1f}%")
        if 6 in sel_row.index:
            c3.metric("Month 6", f"{sel_row[6]:.1f}%")


# =============================================================
# TAB 3 — CHURN PREDICTOR
# =============================================================
with tab3:
    st.header("Churn Risk Predictor")
    st.caption(
        "Predict whether a customer is likely to churn. "
        "Look up an existing customer or enter values manually."
    )

    mode = st.radio(
        "Prediction mode",
        options=["🔍  Existing customer lookup", "🎛️  Manual entry"],
        horizontal=True,
    )

    st.divider()
    feature_values = {}

    # ==========================================================
    # MODE A — Existing customer lookup
    # ==========================================================
    if "Existing" in mode:
        st.subheader("Look up an existing customer")

        # churn_df already has KMeans_Segment (from churn_labels.csv)
        display_df = churn_df.copy()

        fc1, fc2 = st.columns([2, 3])
        with fc1:
            seg_filter = st.selectbox(
                "Filter by segment (optional)",
                options=["All segments"] + sorted(
                    display_df["KMeans_Segment"].dropna().unique().tolist()
                ),
                index=0,
            )

        filtered_customers = (
            display_df if seg_filter == "All segments"
            else display_df[display_df["KMeans_Segment"] == seg_filter]
        )

        with fc2:
            customer_options = sorted(
                filtered_customers["Customer ID"].astype(str).tolist()
            )
            selected_id = st.selectbox(
                f"Select Customer ID  ({len(customer_options):,} available)",
                options=customer_options,
            )

        cust_row = display_df[
            display_df["Customer ID"].astype(str) == str(selected_id)
        ].iloc[0]

        # Customer profile metrics
        st.markdown("**Customer profile**")
        ic = st.columns(5)
        ic[0].metric("Segment",   cust_row.get("KMeans_Segment", "—"))
        ic[1].metric("Recency",   f"{cust_row['Recency']:.0f} days")
        ic[2].metric("Frequency", f"{cust_row['Frequency']:.0f} orders")
        ic[3].metric("Monetary",  f"£{cust_row['Monetary']:,.0f}")
        ic[4].metric("CLV",       f"£{cust_row['CLV_capped']:,.0f}")

        # Extract feature values from the customer row
        for col in CHURN_FEATURE_COLS:
            feature_values[col] = float(cust_row[col])

        # Show stored prediction if available
        if "Churn_Pred" in cust_row and pd.notna(cust_row.get("Churn_Pred")):
            stored_prob  = cust_row.get("Churn_Prob", None)
            stored_label = int(cust_row["Churn_Pred"])
            split_tag    = cust_row.get("Split", "")
            if pd.notna(stored_prob):
                rl, rc, re = get_churn_risk_label(float(stored_prob))
                st.info(
                    f"**Stored prediction** ({split_tag} split):  "
                    f"{'Churned' if stored_label == 1 else 'Active'}  "
                    f"· probability {float(stored_prob):.1%}  · {re} {rl}"
                )

    # ==========================================================
    # MODE B — Manual slider entry
    # ==========================================================
    else:
        st.subheader("Enter customer features manually")
        st.caption(
            "Adjust sliders to explore how different profiles "
            "affect predicted churn risk."
        )

        left_col, right_col = st.columns(2)
        cols_cycle = [left_col, right_col]

        for i, feat in enumerate(CHURN_FEATURE_COLS):
            r               = feat_ranges.get(feat, {})
            label, desc, unit = FEATURE_DISPLAY.get(feat, (feat, "", ""))
            suffix   = f" {unit}" if unit else ""
            f_min    = r.get("min",    0.0)
            f_max    = r.get("max",  100.0)
            f_median = r.get("median", (f_min + f_max) / 2)
            use_int  = feat in ["Frequency", "F_Score", "M_Score",
                                 "product_diversity"]

            with cols_cycle[i % 2]:
                if use_int:
                    val = st.slider(
                        f"{label}{suffix}",
                        min_value=int(f_min),
                        max_value=int(f_max),
                        value=int(f_median),
                        step=1,
                        help=desc,
                    )
                else:
                    val = st.slider(
                        f"{label}{suffix}",
                        min_value=float(f_min),
                        max_value=float(f_max),
                        value=float(f_median),
                        step=float(max((f_max - f_min) / 100, 0.1)),
                        format="%.1f",
                        help=desc,
                    )
                feature_values[feat] = float(val)

    # ==========================================================
    # LIVE PREDICTION
    # ==========================================================
    st.divider()
    st.subheader("Prediction")

    churn_prob, churn_pred, X_scaled = predict_churn_single(
        feature_values, models
    )
    risk_label, risk_colour, risk_emoji = get_churn_risk_label(churn_prob)

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=churn_prob * 100,
        number=dict(suffix="%", font=dict(size=36)),
        delta=dict(reference=50, suffix="%",
                   increasing=dict(color="#993C1D"),
                   decreasing=dict(color="#0F6E56")),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%"),
            bar=dict(color=risk_colour, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  25], color="rgba(15, 110, 86, 0.15)"),
                dict(range=[25, 50], color="rgba(133, 79, 11, 0.10)"),
                dict(range=[50, 75], color="rgba(153, 60, 29, 0.12)"),
                dict(range=[75,100], color="rgba(153, 60, 29, 0.22)"),
            ],
            threshold=dict(
                line=dict(color="black", width=2),
                thickness=0.75, value=50
            ),
        ),
        title=dict(
            text=f"{risk_emoji}  {risk_label}",
            font=dict(size=18, color=risk_colour)
        ),
    ))
    fig_gauge.update_layout(
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
    )

    gauge_col, result_col = st.columns([2, 3], gap="large")

    with gauge_col:
        st.plotly_chart(fig_gauge)

    with result_col:
        st.markdown(f"#### {risk_emoji} {risk_label}")
        st.markdown(
            f"**Predicted:** "
            f"{'🔴 Likely to churn' if churn_pred == 1 else '🟢 Likely to stay active'}"
        )
        st.markdown(f"**Churn probability:** {churn_prob:.1%}")
        st.markdown("**Classification threshold:** 50%")
        st.divider()

        feat_summary = pd.DataFrame({
            "Feature" : [FEATURE_DISPLAY.get(k, (k,))[0]
                         for k in CHURN_FEATURE_COLS],
            "Value"   : [f"{feature_values[k]:.1f}"
                         for k in CHURN_FEATURE_COLS],
        })
        st.markdown("**Input features used:**")
        st.dataframe(
            feat_summary,
            use_container_width=True,
            hide_index=True,
            height=230,
        )

    # ==========================================================
    # SHAP WATERFALL — on demand
    # ==========================================================
    st.divider()
    st.subheader("Why this prediction? — SHAP Explanation")
    st.caption(
        "SHAP values show which features pushed this prediction toward "
        "or away from churn. Red = pushes toward churn. Blue = pushes away."
    )

    if st.button("🔍  Generate SHAP explanation", type="primary"):
        with st.spinner("Computing SHAP values ..."):
            try:
                explainer   = shap.TreeExplainer(models["churn_model"])
                shap_values = explainer.shap_values(X_scaled)

                shap_exp = shap.Explanation(
                    values=shap_values,
                    base_values=explainer.expected_value,
                    data=X_scaled,
                    feature_names=CHURN_FEATURE_COLS
                )

                # SHAP uses matplotlib; render via st.pyplot
                matplotlib.use("Agg")
                fig_shap, ax = plt.subplots(figsize=(9, 5))
                plt.sca(ax)
                shap.waterfall_plot(
                    shap_exp[0],
                    show=False,
                    max_display=8,
                )
                plt.title(
                    "SHAP Waterfall — Feature contributions to this prediction",
                    fontsize=11, pad=10
                )
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close(fig_shap)

                # Top driver explanation
                top_idx   = np.argmax(np.abs(shap_values[0]))
                top_feat  = CHURN_FEATURE_COLS[top_idx]
                top_shap  = shap_values[0][top_idx]
                top_label = FEATURE_DISPLAY.get(top_feat, (top_feat,))[0]
                direction = ("toward churn ↑" if top_shap > 0
                             else "away from churn ↓")
                st.info(
                    f"**Top driver:** {top_label}  "
                    f"(SHAP value: {top_shap:+.3f} — pushes {direction})"
                )

            except Exception as e:
                st.error(f"SHAP computation failed: {e}")

    else:
        st.markdown(
            '<div style="background:rgba(24,95,165,0.05);border-radius:8px;'
            'padding:1.5rem;text-align:center;color:grey;">'
            'Click the button above to generate the SHAP explanation.'
            '</div>',
            unsafe_allow_html=True
        )