import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
GITHUB_URL = "https://github.com/Hvora2005/mlops-platform"

MODEL_INFO = {
    "logistic_regression": ("📈", "Logistic Regression", "A fast, interpretable linear baseline for classification."),
    "linear_regression": ("📈", "Linear Regression", "A fast, interpretable linear baseline for regression."),
    "decision_tree": ("🌳", "Decision Tree", "A single tree of if/else rules — easy to visualize, prone to overfitting alone."),
    "random_forest": ("🌲", "Random Forest", "An ensemble of many decision trees — robust and usually a strong default."),
    "xgboost": ("⚡", "XGBoost", "A gradient-boosted tree ensemble — often the top performer on tabular data."),
}

st.set_page_config(
    page_title="MLOps Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

DARK = {
    "bg": "#0f1420", "bg2": "#131a2b", "sidebar": "#0b0f19",
    "text": "#f5f7fa", "text2": "#c7cddb", "muted": "#94a3b8",
    "card": "rgba(255,255,255,0.04)", "border": "rgba(255,255,255,0.08)",
    "tab_bg": "rgba(255,255,255,0.04)",
}
LIGHT = {
    "bg": "#f7f8fc", "bg2": "#ffffff", "sidebar": "#ffffff",
    "text": "#161a25", "text2": "#3b4254", "muted": "#6b7280",
    "card": "rgba(15,20,32,0.03)", "border": "rgba(15,20,32,0.08)",
    "tab_bg": "rgba(15,20,32,0.04)",
}
T = LIGHT if st.session_state["theme"] == "light" else DARK

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    .stApp {{ background: linear-gradient(180deg, {T['bg']} 0%, {T['bg2']} 100%); }}
    [data-testid="stSidebar"] {{ background: {T['sidebar']}; border-right: 1px solid {T['border']}; }}
    h1, h2, h3 {{ color: {T['text']} !important; font-weight: 700 !important; }}
    p, label, span, .stMarkdown {{ color: {T['text2']}; }}

    .navbar {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.9rem 1.5rem; border-radius: 14px;
        background: {T['card']}; border: 1px solid {T['border']};
        margin-bottom: 1.25rem;
    }}
    .navbar .brand {{ font-size: 1.15rem; font-weight: 800; color: {T['text']}; }}
    .navbar .links a {{
        color: {T['muted']}; text-decoration: none; margin-left: 1.4rem; font-size: 0.9rem; font-weight: 600;
    }}
    .navbar .links a:hover {{ color: #8b5cf6; }}

    .hero {{
        padding: 2rem 2.25rem; border-radius: 18px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        margin-bottom: 1.5rem; box-shadow: 0 8px 30px rgba(99,102,241,0.25);
    }}
    .hero h1 {{ color: white !important; margin: 0; font-size: 2.1rem; }}
    .hero p {{ color: rgba(255,255,255,0.92); margin: 0.5rem 0 0 0; font-size: 1.05rem; max-width: 640px; }}

    .how-step {{
        background: {T['card']}; border: 1px solid {T['border']}; border-radius: 14px;
        padding: 1.1rem; text-align: center; height: 100%;
    }}
    .how-step .num {{
        width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg,#6366f1,#8b5cf6);
        color: white; display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.85rem; margin: 0 auto 0.6rem auto;
    }}
    .how-step .icon {{ font-size: 1.6rem; margin-bottom: 0.3rem; }}
    .how-step .title {{ font-weight: 700; color: {T['text']}; font-size: 0.95rem; }}
    .how-step .desc {{ color: {T['muted']}; font-size: 0.8rem; margin-top: 0.2rem; }}

    .card {{
        background: {T['card']}; border: 1px solid {T['border']}; border-radius: 14px;
        padding: 1.25rem 1.5rem; margin-bottom: 1rem;
    }}

    .stat-card {{
        background: {T['card']}; border: 1px solid {T['border']}; border-radius: 14px;
        padding: 1rem 1.25rem; display: flex; align-items: center; gap: 0.9rem;
    }}
    .stat-card .icon {{
        font-size: 1.4rem; width: 42px; height: 42px; border-radius: 10px;
        background: rgba(139,92,246,0.15); display: flex; align-items: center; justify-content: center;
    }}
    .stat-card .value {{ font-size: 1.4rem; font-weight: 800; color: {T['text']}; line-height: 1.1; }}
    .stat-card .label {{ font-size: 0.78rem; color: {T['muted']}; }}

    .model-card {{
        background: {T['card']}; border: 1px solid {T['border']}; border-radius: 12px;
        padding: 0.9rem 1rem; height: 100%;
    }}
    .model-card .name {{ font-weight: 700; color: {T['text']}; font-size: 0.9rem; }}
    .model-card .desc {{ color: {T['muted']}; font-size: 0.78rem; margin-top: 0.25rem; }}

    .step-badge {{
        display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 0.4rem;
    }}
    .step-done {{ background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }}
    .step-pending {{ background: rgba(148,163,184,0.14); color: {T['muted']}; border: 1px solid rgba(148,163,184,0.25); }}

    .best-model-banner {{
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.08));
        border: 1px solid rgba(34,197,94,0.35); border-radius: 14px;
        padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
    }}
    .best-model-banner .label {{ color: #22c55e; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }}
    .best-model-banner .name {{ color: {T['text']}; font-size: 1.6rem; font-weight: 800; margin-top: 0.15rem; }}

    .empty-state {{
        text-align: center; padding: 2.5rem 1.5rem; background: {T['card']};
        border: 1px dashed {T['border']}; border-radius: 14px;
    }}
    .empty-state .icon {{ font-size: 2.2rem; margin-bottom: 0.5rem; }}
    .empty-state .title {{ font-weight: 700; color: {T['text']}; font-size: 1.05rem; }}
    .empty-state .desc {{ color: {T['muted']}; font-size: 0.88rem; margin-top: 0.3rem; }}

    .footer {{
        margin-top: 2.5rem; padding-top: 1.5rem; border-top: 1px solid {T['border']};
        text-align: center; color: {T['muted']}; font-size: 0.82rem;
    }}
    .footer .badges {{ margin-bottom: 0.6rem; }}
    .footer .badge {{
        display: inline-block; background: {T['card']}; border: 1px solid {T['border']};
        border-radius: 999px; padding: 0.2rem 0.75rem; margin: 0.15rem; font-size: 0.75rem; color: {T['text2']};
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; border: none;
        border-radius: 10px; padding: 0.55rem 1.4rem; font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 6px 18px rgba(99,102,241,0.35); }}

    [data-testid="stMetricValue"] {{ color: {T['text']}; }}
    [data-testid="stMetricLabel"] {{ color: {T['muted']}; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 0.5rem; }}
    .stTabs [data-baseweb="tab"] {{
        background: {T['tab_bg']}; border-radius: 10px 10px 0 0; padding: 0.6rem 1.2rem; color: {T['muted']};
    }}
    .stTabs [aria-selected="true"] {{ background: rgba(99,102,241,0.18) !important; color: #8b5cf6 !important; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="navbar">
        <div class="brand">🧠 MLOps Platform</div>
        <div class="links">
            <a href="{API_URL}/docs" target="_blank">API Docs</a>
            <a href="{GITHUB_URL}" target="_blank">GitHub</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Train and ship ML models without the boilerplate</h1>
        <p>Upload a CSV, let the platform clean it, train and tune several models, compare them side by side, and predict from the best one — all tracked in MLflow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How it works", expanded=("dataset" not in st.session_state)):
    steps = [
        ("1", "📁", "Upload", "Drop in any CSV. We profile every column automatically."),
        ("2", "🧹", "Clean & Configure", "Missing values, encoding, and scaling are handled for you."),
        ("3", "⚙️", "Train & Tune", "Four model families are trained and hyperparameter-tuned."),
        ("4", "🔮", "Predict", "The best model is auto-selected and ready to serve predictions."),
    ]
    cols = st.columns(4)
    for col, (num, icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="how-step">
                    <div class="num">{num}</div>
                    <div class="icon">{icon}</div>
                    <div class="title">{title}</div>
                    <div class="desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---- Sidebar ----
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

    if has_dataset:
        st.markdown("---")
        st.caption("Recent runs")
        try:
            resp = requests.get(f"{API_URL}/training/runs", params={"dataset_id": st.session_state["dataset"]["id"]}, timeout=5)
            runs = resp.json() if resp.ok else []
        except requests.exceptions.RequestException:
            runs = []
        if not runs:
            st.caption("No runs yet.")
        else:
            for r in runs[:5]:
                status_icon = {"completed": "✅", "running": "⏳", "failed": "❌"}.get(r["status"], "•")
                label = r["best_model_name"] or r["status"]
                st.caption(f"{status_icon} {label} · {r['created_at'][:10]}")

    st.markdown("---")
    theme_choice = st.radio("Theme", ["Dark", "Light"], index=0 if st.session_state["theme"] == "dark" else 1, horizontal=True)
    new_theme = theme_choice.lower()
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

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
        if st.button("Upload dataset"):
            if "run" in st.session_state:
                st.toast("Previous training results were cleared for the new dataset.", icon="⚠️")
            with st.spinner("Uploading and profiling..."):
                resp = requests.post(
                    f"{API_URL}/datasets/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                )
            if resp.ok:
                st.session_state["dataset"] = resp.json()
                st.session_state.pop("run", None)
                st.toast(f"Uploaded {resp.json()['filename']}", icon="✅")
                st.rerun()
            else:
                st.error(resp.text)
    st.markdown("</div>", unsafe_allow_html=True)

    if "dataset" in st.session_state:
        ds = st.session_state["dataset"]

        stat_cols = st.columns(3)
        missing_cols = sum(1 for c in ds["column_summary"].values() if c["missing_pct"] > 0)
        stats = [("📊", str(ds["n_rows"]), "Rows"), ("📋", str(ds["n_cols"]), "Columns"), ("⚠️", str(missing_cols), "Columns w/ missing data")]
        for col, (icon, value, label) in zip(stat_cols, stats):
            with col:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="icon">{icon}</div>
                        <div>
                            <div class="value">{value}</div>
                            <div class="label">{label}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data preview")
        try:
            preview_resp = requests.get(f"{API_URL}/datasets/{ds['id']}/preview", timeout=10)
            if preview_resp.ok:
                preview = preview_resp.json()
                st.dataframe(pd.DataFrame(preview["rows"], columns=preview["columns"]), use_container_width=True)
            else:
                st.caption("Preview unavailable.")
        except requests.exceptions.RequestException:
            st.caption("Preview unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Column summary")
        st.dataframe(pd.DataFrame(ds["column_summary"]).T, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">📁</div>
                <div class="title">No dataset yet</div>
                <div class="desc">Upload a CSV above to see row/column stats, a data preview, and column profiling here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_train:
    if "dataset" not in st.session_state:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">⚙️</div>
                <div class="title">Nothing to train yet</div>
                <div class="desc">Upload a dataset in the Upload tab first.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Models that will be trained")
        model_keys = (
            ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
            if task_type == "classification"
            else ["linear_regression", "decision_tree", "random_forest", "xgboost"]
        )
        mcols = st.columns(4)
        for col, key in zip(mcols, model_keys):
            icon, name, desc = MODEL_INFO[key]
            with col:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="name">{icon} {name}</div>
                        <div class="desc">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("Each model is hyperparameter-tuned with RandomizedSearchCV, then ranked by validation score.")

        if st.button("🚀 Train & tune models"):
            with st.spinner("Training and tuning models... this can take a minute"):
                resp = requests.post(
                    f"{API_URL}/training/run",
                    json={"dataset_id": ds["id"], "target_column": target_column, "task_type": task_type},
                )
            if resp.ok:
                st.session_state["run"] = resp.json()
                st.toast("Training complete!", icon="🎉")
                st.balloons()
            else:
                st.error(resp.text)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_results:
    if "run" not in st.session_state:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">📊</div>
                <div class="title">No results yet</div>
                <div class="desc">Train models in the Train tab to see comparisons here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        run = st.session_state["run"]
        best_icon, best_name, best_desc = MODEL_INFO.get(run["best_model_name"], ("🏆", run["best_model_name"], ""))

        st.markdown(
            f"""
            <div class="best-model-banner">
                <div class="label">🏆 Best model</div>
                <div class="name">{best_icon} {best_name}</div>
                <div style="color:{T['muted']}; font-size:0.85rem; margin-top:0.3rem;">{best_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics_df = pd.DataFrame(run["metrics"]).T
        metric_cols = list(metrics_df.columns)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model comparison")
        chart_metric = "f1" if "f1" in metric_cols else metric_cols[0]
        chart_metric = st.selectbox("Compare by metric", metric_cols, index=metric_cols.index(chart_metric))
        st.bar_chart(metrics_df[chart_metric])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Full metrics table")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="rgba(34,197,94,0.25)"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_predict:
    if "run" not in st.session_state:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">🔮</div>
                <div class="title">No model ready</div>
                <div class="desc">Train models first, then come back here to predict.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
                st.toast("Prediction ready", icon="🔮")
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

st.markdown(
    f"""
    <div class="footer">
        <div class="badges">
            <span class="badge">FastAPI</span>
            <span class="badge">scikit-learn</span>
            <span class="badge">XGBoost</span>
            <span class="badge">MLflow</span>
            <span class="badge">PostgreSQL</span>
            <span class="badge">Streamlit</span>
            <span class="badge">Docker</span>
        </div>
        MLOps Platform · <a href="{GITHUB_URL}" target="_blank" style="color:{T['muted']};">View source on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
