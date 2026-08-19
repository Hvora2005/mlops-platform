import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MLOps Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #0f1420 0%, #131a2b 100%);
    }
    [data-testid="stSidebar"] {
        background: #0b0f19;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    h1, h2, h3 { color: #f5f7fa !important; }
    p, label, span, .stMarkdown { color: #c7cddb; }

    .hero {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(99,102,241,0.25);
    }
    .hero h1 { color: white !important; margin: 0; font-size: 2rem; }
    .hero p { color: rgba(255,255,255,0.9); margin: 0.35rem 0 0 0; font-size: 1rem; }

    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }

    .step-badge {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .step-done { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .step-pending { background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid rgba(148,163,184,0.25); }

    .best-model-banner {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.08));
        border: 1px solid rgba(74,222,128,0.35);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .best-model-banner .label { color: #86efac; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
    .best-model-banner .name { color: #f0fdf4; font-size: 1.6rem; font-weight: 700; margin-top: 0.15rem; }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(99,102,241,0.35);
    }

    [data-testid="stMetricValue"] { color: #f5f7fa; }
    [data-testid="stMetricLabel"] { color: #94a3b8; }

    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 0.6rem 1.2rem;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.18) !important;
        color: #e0e7ff !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>🧠 MLOps Platform</h1>
        <p>Upload a dataset, auto-clean it, train &amp; tune multiple models, and serve the best one — end to end.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar: pipeline status ----
with st.sidebar:
    st.markdown("### Pipeline status")

    def step_badge(label: str, done: bool):
        cls = "step-done" if done else "step-pending"
        icon = "✓" if done else "○"
        st.markdown(f'<span class="step-badge {cls}">{icon} {label}</span>', unsafe_allow_html=True)

    has_dataset = "dataset" in st.session_state
    has_run = "run" in st.session_state and st.session_state["run"].get("status") == "completed"

    step_badge("1. Dataset uploaded", has_dataset)
    st.write("")
    step_badge("2. Models trained", has_run)
    st.write("")
    step_badge("3. Ready to predict", has_run)

    st.markdown("---")
    if has_dataset:
        ds = st.session_state["dataset"]
        st.caption("Current dataset")
        st.write(f"**{ds['filename']}**")
        st.write(f"{ds['n_rows']} rows · {ds['n_cols']} cols")
    if has_run:
        run = st.session_state["run"]
        st.caption("Best model")
        st.write(f"🏆 **{run['best_model_name']}**")

    st.markdown("---")
    with st.expander("⚙️ Settings"):
        st.caption("Backend connection")
        st.code(API_URL, language=None)

tab_upload, tab_train, tab_results, tab_predict = st.tabs(
    ["📁 Upload", "⚙️ Train", "📊 Results", "🔮 Predict"]
)

with tab_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload a CSV dataset")
    uploaded = st.file_uploader("Drag and drop or browse", type="csv", label_visibility="collapsed")
    if uploaded is not None:
        if st.button("Upload dataset", use_container_width=False):
            with st.spinner("Uploading and profiling..."):
                resp = requests.post(
                    f"{API_URL}/datasets/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                )
            if resp.ok:
                st.session_state["dataset"] = resp.json()
                st.session_state.pop("run", None)
                st.success(f"Uploaded: {resp.json()['filename']}")
            else:
                st.error(resp.text)
    st.markdown("</div>", unsafe_allow_html=True)

    if "dataset" in st.session_state:
        ds = st.session_state["dataset"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", ds["n_rows"])
        col2.metric("Columns", ds["n_cols"])
        missing_cols = sum(1 for c in ds["column_summary"].values() if c["missing_pct"] > 0)
        col3.metric("Columns with missing data", missing_cols)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Column summary")
        st.dataframe(pd.DataFrame(ds["column_summary"]).T, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_train:
    if "dataset" not in st.session_state:
        st.info("Upload a dataset first in the **Upload** tab.")
    else:
        ds = st.session_state["dataset"]
        columns = list(ds["column_summary"].keys())

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Configure training")
        c1, c2 = st.columns(2)
        with c1:
            target_column = st.selectbox("Target column", columns)
        with c2:
            task_type = st.radio("Task type", ["classification", "regression"], horizontal=True)

        st.caption("Trains Logistic/Linear Regression, Decision Tree, Random Forest, and XGBoost, each with hyperparameter tuning, and picks the best by validation score.")

        if st.button("🚀 Train & tune models"):
            with st.spinner("Training and tuning models... this can take a minute"):
                resp = requests.post(
                    f"{API_URL}/training/run",
                    json={
                        "dataset_id": ds["id"],
                        "target_column": target_column,
                        "task_type": task_type,
                    },
                )
            if resp.ok:
                st.session_state["run"] = resp.json()
                st.success("Training complete — check the Results tab.")
                st.balloons()
            else:
                st.error(resp.text)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_results:
    if "run" not in st.session_state:
        st.info("Run training first in the **Train** tab.")
    else:
        run = st.session_state["run"]

        st.markdown(
            f"""
            <div class="best-model-banner">
                <div class="label">🏆 Best model</div>
                <div class="name">{run['best_model_name']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics_df = pd.DataFrame(run["metrics"]).T
        metric_cols = list(metrics_df.columns)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model comparison")
        chart_metric = metric_cols[0] if "f1" not in metric_cols else "f1"
        chart_metric = st.selectbox("Compare by metric", metric_cols, index=metric_cols.index(chart_metric))
        st.bar_chart(metrics_df[chart_metric])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Full metrics table")
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color="rgba(74,222,128,0.25)"),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab_predict:
    if "run" not in st.session_state:
        st.info("Run training first in the **Train** tab.")
    else:
        run = st.session_state["run"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"Predict with {run['best_model_name']}")
        st.caption("Enter feature values for a single row to get a prediction.")

        inputs = {}
        cols = st.columns(2)
        for i, col in enumerate(run["feature_columns"]):
            with cols[i % 2]:
                inputs[col] = st.text_input(col, key=f"predict_{col}")

        if st.button("🔮 Predict"):
            resp = requests.post(
                f"{API_URL}/training/predict",
                json={"training_run_id": run["id"], "records": [inputs]},
            )
            if resp.ok:
                prediction = resp.json()["predictions"][0]
                st.markdown(
                    f"""
                    <div class="best-model-banner">
                        <div class="label">Prediction</div>
                        <div class="name">{prediction}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(resp.text)
        st.markdown("</div>", unsafe_allow_html=True)
