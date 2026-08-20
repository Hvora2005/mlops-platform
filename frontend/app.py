import os
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "changeme")
GITHUB_URL = "https://github.com/Hvora2005/mlops-platform"

MODEL_INFO = {
    "logistic_regression": ("trending-up", "Logistic Regression", "A fast, interpretable linear baseline for classification."),
    "linear_regression": ("trending-up", "Linear Regression", "A fast, interpretable linear baseline for regression."),
    "decision_tree": ("git-branch", "Decision Tree", "A single tree of if/else rules — easy to visualize, prone to overfitting alone."),
    "random_forest": ("layers", "Random Forest", "An ensemble of many decision trees — robust and usually a strong default."),
    "xgboost": ("zap", "XGBoost", "A gradient-boosted tree ensemble — often the top performer on tabular data."),
}

# ---- Minimal inline icon set (Feather-style, single stroke, no emoji) ----
_ICON_PATHS = {
    "upload": '<path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
    "filter": '<path d="M4 5h16M7 12h10M10 19h4"/>',
    "sliders": '<line x1="4" y1="6" x2="20" y2="6"/><circle cx="9" cy="6" r="2"/><line x1="4" y1="12" x2="20" y2="12"/><circle cx="15" cy="12" r="2"/><line x1="4" y1="18" x2="20" y2="18"/><circle cx="7" cy="18" r="2"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="0.8" fill="currentColor"/>',
    "bar-chart": '<line x1="4" y1="20" x2="4" y2="10"/><line x1="10" y1="20" x2="10" y2="4"/><line x1="16" y1="20" x2="16" y2="14"/>',
    "check-circle": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9"/>',
    "circle": '<circle cx="12" cy="12" r="9"/>',
    "alert-triangle": '<path d="M12 3.5l9.5 16.5H2.5L12 3.5z"/><line x1="12" y1="9.5" x2="12" y2="14"/><circle cx="12" cy="16.8" r="0.75" fill="currentColor"/>',
    "grid": '<rect x="3.5" y="3.5" width="6.5" height="6.5" rx="1"/><rect x="14" y="3.5" width="6.5" height="6.5" rx="1"/><rect x="3.5" y="14" width="6.5" height="6.5" rx="1"/><rect x="14" y="14" width="6.5" height="6.5" rx="1"/>',
    "list": '<line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/>',
    "download": '<path d="M12 3v12M12 15l-4-4M12 15l4-4"/><path d="M4 19h16"/>',
    "star": '<path d="M12 2.5l2.9 6.6 7.1.6-5.4 4.7 1.7 7-6.3-3.9-6.3 3.9 1.7-7-5.4-4.7 7.1-.6L12 2.5z"/>',
    "trending-up": '<path d="M3 16l6-6 4 4 8-8"/><path d="M15 6h6v6"/>',
    "git-branch": '<circle cx="6" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="9" r="2"/><path d="M6 7v10"/><path d="M6 11a6 6 0 0 0 6 6h4"/>',
    "layers": '<path d="M12 2.5l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5"/><path d="M3 16.5l9 5 9-5"/>',
    "zap": '<path d="M13 2L4 14h6l-1 8 9-12h-6l1-8z"/>',
    "database": '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v13c0 1.66 3.58 3 8 3s8-1.34 8-3v-13"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/>',
    "cube": '<path d="M12 2.5l8.5 4.75v9.5L12 21.5l-8.5-4.75v-9.5L12 2.5z"/><path d="M3.5 7.25L12 12l8.5-4.75"/><path d="M12 12v9.5"/>',
    "arrow-right": '<line x1="4" y1="12" x2="19" y2="12"/><path d="M13 6l6 6-6 6"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke_width: float = 1.8) -> str:
    body = _ICON_PATHS.get(name, _ICON_PATHS["circle"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:block">{body}</svg>'
    )


st.set_page_config(
    page_title="MLOps Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

ACCENT = "#4f6ef7"
ACCENT_STRONG = "#3d59e0"
SUCCESS = "#1a9e6b"
WARNING = "#b7791f"

DARK = {
    "bg": "#0d1117", "bg2": "#0d1117", "sidebar": "#0a0e14",
    "text": "#eef1f6", "text2": "#b7bfcc", "muted": "#7c8698",
    "card": "#12161f", "border": "#232a37", "border_soft": "#1a2029",
    "chip": "#1a2029", "track": "#1e2530",
}
LIGHT = {
    "bg": "#f7f8fa", "bg2": "#f7f8fa", "sidebar": "#ffffff",
    "text": "#12151c", "text2": "#3f4757", "muted": "#6b7385",
    "card": "#ffffff", "border": "#e4e7ed", "border_soft": "#edeff3",
    "chip": "#f1f2f6", "track": "#e9ebf0",
}
T = LIGHT if st.session_state["theme"] == "light" else DARK
IS_DARK = st.session_state["theme"] == "dark"

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    .stApp {{ background: {T['bg']}; }}
    [data-testid="stSidebar"] {{ background: {T['sidebar']}; border-right: 1px solid {T['border']}; }}
    [data-testid="stSidebar"] > div {{ padding-top: 1.25rem; }}
    h1, h2, h3 {{ color: {T['text']} !important; font-weight: 700 !important; letter-spacing: -0.01em; }}
    p, label, span, .stMarkdown {{ color: {T['text2']}; }}
    [data-testid="stAppViewBlockContainer"] {{ padding-top: 1.75rem; max-width: 1180px; }}

    /* ---- Header ---- */
    .app-header {{
        display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 1.1rem; margin-bottom: 1.5rem; border-bottom: 1px solid {T['border']};
    }}
    .app-header .brand {{ display: flex; align-items: center; gap: 0.75rem; }}
    .logo-mark {{
        width: 36px; height: 36px; border-radius: 9px; background: {ACCENT};
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 800; font-size: 1.05rem; flex-shrink: 0;
    }}
    .brand-name {{ font-size: 1.08rem; font-weight: 700; color: {T['text']}; line-height: 1.25; }}
    .brand-tag {{ font-size: 0.8rem; color: {T['muted']}; line-height: 1.25; }}
    .app-header .links {{ display: flex; align-items: center; gap: 1.5rem; }}
    .app-header .links a {{
        color: {T['text2']}; text-decoration: none; font-size: 0.85rem; font-weight: 600;
    }}
    .app-header .links a:hover {{ color: {ACCENT}; }}

    /* ---- Pipeline stepper (top) ---- */
    .pipeline {{ display: flex; align-items: center; margin-bottom: 1.75rem; }}
    .pipeline-step {{ display: flex; align-items: center; gap: 0.55rem; flex: 1; }}
    .pipeline-step .dot {{
        width: 26px; height: 26px; border-radius: 50%; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
    }}
    .pipeline-step.done .dot {{ background: {ACCENT}; color: white; }}
    .pipeline-step.active .dot {{ background: {T['card']}; border: 1.8px solid {ACCENT}; color: {ACCENT}; }}
    .pipeline-step.upcoming .dot {{ background: {T['card']}; border: 1.8px solid {T['border']}; color: {T['muted']}; }}
    .pipeline-step .label {{ font-size: 0.83rem; font-weight: 600; color: {T['muted']}; white-space: nowrap; }}
    .pipeline-step.done .label, .pipeline-step.active .label {{ color: {T['text']}; }}
    .pipeline-line {{ flex: 1; height: 1.5px; background: {T['border']}; margin: 0 0.75rem; }}
    .pipeline-line.done {{ background: {ACCENT}; }}

    /* ---- How it works ---- */
    .how-step {{
        background: {T['card']}; border: 1px solid {T['border']}; border-radius: 12px;
        padding: 1.1rem; height: 100%;
    }}
    .how-step .icon-row {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }}
    .how-step .icon-box {{
        width: 30px; height: 30px; border-radius: 8px; background: {T['chip']};
        display: flex; align-items: center; justify-content: center; color: {ACCENT}; flex-shrink: 0;
    }}
    .how-step .step-no {{ font-size: 0.72rem; font-weight: 700; color: {T['muted']}; letter-spacing: 0.04em; }}
    .how-step .title {{ font-weight: 700; color: {T['text']}; font-size: 0.92rem; }}
    .how-step .desc {{ color: {T['muted']}; font-size: 0.8rem; margin-top: 0.25rem; line-height: 1.4; }}

    .card {{
        background: {T['card']}; border: 1px solid {T['border']}; border-radius: 12px;
        padding: 1.25rem 1.5rem; margin-bottom: 1rem;
    }}
    .card .subhead {{ font-size: 0.98rem; font-weight: 700; color: {T['text']}; margin-bottom: 0.15rem; }}
    .card .subcaption {{ font-size: 0.82rem; color: {T['muted']}; margin-bottom: 0.9rem; }}

    .stat-card {{
        background: {T['card']}; border: 1px solid {T['border']}; border-radius: 12px;
        padding: 1rem 1.15rem; display: flex; align-items: center; gap: 0.85rem;
    }}
    .stat-card .icon-box {{
        width: 38px; height: 38px; border-radius: 9px; flex-shrink: 0;
        background: {T['chip']}; color: {ACCENT}; display: flex; align-items: center; justify-content: center;
    }}
    .stat-card .value {{ font-size: 1.3rem; font-weight: 800; color: {T['text']}; line-height: 1.15; }}
    .stat-card .label {{ font-size: 0.76rem; color: {T['muted']}; }}

    .model-card {{
        background: {T['card']}; border: 1px solid {T['border']}; border-radius: 12px;
        padding: 0.95rem 1.05rem; height: 100%;
    }}
    .model-card .icon-row {{ display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.4rem; color: {ACCENT}; }}
    .model-card .name {{ font-weight: 700; color: {T['text']}; font-size: 0.88rem; }}
    .model-card .desc {{ color: {T['muted']}; font-size: 0.77rem; margin-top: 0.15rem; line-height: 1.4; }}

    .best-model-banner {{
        background: {T['card']}; border: 1px solid {T['border']}; border-left: 3px solid {SUCCESS};
        border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1.25rem;
        display: flex; align-items: center; gap: 0.9rem;
    }}
    .best-model-banner .icon-box {{
        width: 40px; height: 40px; border-radius: 9px; background: rgba(26,158,107,0.12);
        color: {SUCCESS}; display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }}
    .best-model-banner .label {{ color: {SUCCESS}; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }}
    .best-model-banner .name {{ color: {T['text']}; font-size: 1.3rem; font-weight: 800; line-height: 1.2; }}
    .best-model-banner .desc {{ color: {T['muted']}; font-size: 0.82rem; margin-top: 0.1rem; }}

    .empty-state {{
        text-align: center; padding: 2.75rem 1.5rem; background: {T['card']};
        border: 1px dashed {T['border']}; border-radius: 12px;
    }}
    .empty-state .icon-box {{
        width: 46px; height: 46px; border-radius: 11px; background: {T['chip']}; color: {T['muted']};
        display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem auto;
    }}
    .empty-state .title {{ font-weight: 700; color: {T['text']}; font-size: 1rem; }}
    .empty-state .desc {{ color: {T['muted']}; font-size: 0.85rem; margin-top: 0.3rem; }}

    .footer {{
        margin-top: 2.5rem; padding-top: 1.5rem; border-top: 1px solid {T['border']};
        text-align: center; color: {T['muted']}; font-size: 0.8rem;
    }}
    .footer .badges {{ margin-bottom: 0.6rem; }}
    .footer .badge {{
        display: inline-block; background: {T['chip']}; border: 1px solid {T['border']};
        border-radius: 6px; padding: 0.2rem 0.65rem; margin: 0.15rem; font-size: 0.73rem; color: {T['text2']}; font-weight: 500;
    }}

    .stButton > button {{
        background: {ACCENT}; color: white; border: none;
        border-radius: 8px; padding: 0.55rem 1.4rem; font-weight: 600; font-size: 0.88rem;
        transition: background 0.15s ease, box-shadow 0.15s ease;
        box-shadow: none;
    }}
    .stButton > button:hover {{ background: {ACCENT_STRONG}; box-shadow: 0 2px 10px rgba(79,110,247,0.25); }}
    .stDownloadButton > button {{
        border-radius: 8px; font-weight: 600; font-size: 0.85rem;
        border: 1px solid {T['border']}; background: {T['card']}; color: {T['text']};
    }}

    [data-testid="stMetricValue"] {{ color: {T['text']}; }}
    [data-testid="stMetricLabel"] {{ color: {T['muted']}; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; border-bottom: 1px solid {T['border']}; }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; border-radius: 0; padding: 0.6rem 1.1rem; color: {T['muted']};
        font-weight: 600; font-size: 0.88rem; border-bottom: 2px solid transparent;
    }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; }}

    .status-chip {{
        display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.2rem 0.6rem;
        border-radius: 999px; font-size: 0.74rem; font-weight: 600;
    }}
    .status-done {{ background: rgba(26,158,107,0.12); color: {SUCCESS}; }}
    .status-pending {{ background: {T['chip']}; color: {T['muted']}; }}
    .status-running {{ background: rgba(183,121,31,0.12); color: {WARNING}; }}
    .status-failed {{ background: rgba(220,60,60,0.12); color: #dc3c3c; }}

    .side-section-title {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
        color: {T['muted']}; margin-bottom: 0.6rem; margin-top: 0.4rem;
    }}
    .side-info-row {{ font-size: 0.84rem; color: {T['text2']}; margin-bottom: 0.15rem; }}
    .side-info-row b {{ color: {T['text']}; }}
    .run-row {{ display: flex; align-items: center; gap: 0.5rem; font-size: 0.79rem; color: {T['text2']}; padding: 0.2rem 0; }}
    .run-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CHART_TEMPLATE = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=T["text2"], size=12),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(showgrid=True, gridcolor=T["border_soft"], zeroline=False, color=T["muted"]),
    yaxis=dict(showgrid=False, zeroline=False, color=T["muted"]),
    hoverlabel=dict(bgcolor=T["card"], font_size=12, font_family="Inter, sans-serif", bordercolor=T["border"]),
)


def model_display_name(key: str) -> str:
    return MODEL_INFO.get(key, (None, key.replace("_", " ").title(), None))[1]


# ---- Header ----
st.markdown(
    f"""
    <div class="app-header">
        <div class="brand">
            <div class="logo-mark">M</div>
            <div>
                <div class="brand-name">MLOps Platform</div>
                <div class="brand-tag">Automated training &amp; serving for tabular datasets</div>
            </div>
        </div>
        <div class="links">
            <a href="{API_URL}/docs" target="_blank">API Reference</a>
            <a href="{GITHUB_URL}" target="_blank">Source</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Pipeline stepper ----
_has_dataset = "dataset" in st.session_state
_has_run = "run" in st.session_state and st.session_state["run"].get("status") == "completed"
_pipeline_states = ["done", "done" if _has_dataset else "active", "done" if _has_run else ("active" if _has_dataset else "upcoming"), "active" if _has_run else "upcoming"]
_pipeline_labels = ["Upload dataset", "Configure target", "Train & tune", "Predict"]
_step_html = []
for i, (state, label) in enumerate(zip(_pipeline_states, _pipeline_labels)):
    dot_content = icon("check-circle", 14) if state == "done" else str(i + 1)
    _step_html.append(f'<div class="pipeline-step {state}"><div class="dot">{dot_content}</div><div class="label">{label}</div></div>')
    if i < len(_pipeline_labels) - 1:
        line_cls = "done" if state == "done" else ""
        _step_html.append(f'<div class="pipeline-line {line_cls}"></div>')
st.markdown(f'<div class="pipeline">{"".join(_step_html)}</div>', unsafe_allow_html=True)

with st.expander("How it works", expanded=("dataset" not in st.session_state)):
    steps = [
        ("upload", "Upload", "Drop in any CSV. Every column is profiled automatically."),
        ("filter", "Clean & configure", "Missing values, encoding, and scaling are handled for you."),
        ("sliders", "Train & tune", "Four model families are trained and hyperparameter-tuned."),
        ("target", "Predict", "The best model is auto-selected and ready to serve predictions."),
    ]
    cols = st.columns(4)
    for i, (col, (icon_name, title, desc)) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(
                f"""
                <div class="how-step">
                    <div class="icon-row">
                        <div class="icon-box">{icon(icon_name, 16, ACCENT)}</div>
                        <div class="step-no">STEP {i + 1}</div>
                    </div>
                    <div class="title">{title}</div>
                    <div class="desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<div class="side-section-title">Pipeline status</div>', unsafe_allow_html=True)

    has_dataset = "dataset" in st.session_state
    has_run = "run" in st.session_state and st.session_state["run"].get("status") == "completed"

    def status_row(label: str, done: bool):
        i = icon("check-circle", 14, ACCENT if done else T["muted"])
        weight = "600" if done else "500"
        color = T["text"] if done else T["muted"]
        st.markdown(
            f'<div class="side-info-row" style="display:flex;align-items:center;gap:0.5rem;font-weight:{weight};color:{color};">{i}<span>{label}</span></div>',
            unsafe_allow_html=True,
        )

    status_row("Dataset uploaded", has_dataset)
    status_row("Models trained", has_run)
    status_row("Ready to predict", has_run)

    st.markdown("---")
    if has_dataset:
        ds = st.session_state["dataset"]
        st.markdown('<div class="side-section-title">Current dataset</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="side-info-row"><b>{ds["filename"]}</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="side-info-row">{ds["n_rows"]} rows · {ds["n_cols"]} columns</div>', unsafe_allow_html=True)
    if has_run:
        run = st.session_state["run"]
        st.markdown('<div class="side-section-title" style="margin-top:0.9rem;">Best model</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="side-info-row"><b>{model_display_name(run["best_model_name"])}</b></div>', unsafe_allow_html=True)

    if has_dataset:
        st.markdown("---")
        st.markdown('<div class="side-section-title">Recent runs</div>', unsafe_allow_html=True)
        try:
            resp = requests.get(f"{API_URL}/training/runs", params={"dataset_id": st.session_state["dataset"]["id"]}, timeout=5)
            runs = resp.json() if resp.ok else []
        except requests.exceptions.RequestException:
            runs = []
        if not runs:
            st.caption("No runs yet.")
        else:
            dot_colors = {"completed": SUCCESS, "running": WARNING, "failed": "#dc3c3c", "pending": T["muted"]}
            for r in runs[:5]:
                dot = dot_colors.get(r["status"], T["muted"])
                label = model_display_name(r["best_model_name"]) if r["best_model_name"] else r["status"].capitalize()
                st.markdown(
                    f'<div class="run-row"><span class="run-dot" style="background:{dot};"></span>'
                    f'<span>{label} · {r["created_at"][:10]}</span></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    theme_choice = st.radio("Theme", ["Dark", "Light"], index=0 if st.session_state["theme"] == "dark" else 1, horizontal=True)
    new_theme = theme_choice.lower()
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    with st.expander("Settings"):
        st.caption("Backend connection")
        st.code(API_URL, language=None)

tab_upload, tab_train, tab_results, tab_predict = st.tabs(["Upload", "Train", "Results", "Predict"])

with tab_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="subhead">Upload a CSV dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="subcaption">The platform infers column types and profiles data quality automatically.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag and drop or browse", type="csv", label_visibility="collapsed")
    if uploaded is not None:
        if st.button("Upload dataset"):
            if "run" in st.session_state:
                st.toast("Previous training results were cleared for the new dataset.")
            with st.spinner("Uploading and profiling..."):
                resp = requests.post(
                    f"{API_URL}/datasets/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                )
            if resp.ok:
                st.session_state["dataset"] = resp.json()
                st.session_state.pop("run", None)
                st.session_state.pop("active_run_id", None)
                st.toast(f"Uploaded {resp.json()['filename']}")
                st.rerun()
            else:
                st.error(resp.text)
    st.markdown("</div>", unsafe_allow_html=True)

    if "dataset" in st.session_state:
        ds = st.session_state["dataset"]

        stat_cols = st.columns(3)
        missing_cols = sum(1 for c in ds["column_summary"].values() if c["missing_pct"] > 0)
        stats = [("list", str(ds["n_rows"]), "Rows"), ("grid", str(ds["n_cols"]), "Columns"), ("alert-triangle", str(missing_cols), "Columns w/ missing data")]
        for col, (icon_name, value, label) in zip(stat_cols, stats):
            with col:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="icon-box">{icon(icon_name, 18)}</div>
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
        st.markdown('<div class="subhead">Data preview</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="subhead">Column summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(ds["column_summary"]).T, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div class="empty-state">
                <div class="icon-box">{icon("upload", 20)}</div>
                <div class="title">No dataset yet</div>
                <div class="desc">Upload a CSV above to see row/column stats, a data preview, and column profiling here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_train:
    if "dataset" not in st.session_state:
        st.markdown(
            f"""
            <div class="empty-state">
                <div class="icon-box">{icon("sliders", 20)}</div>
                <div class="title">Nothing to train yet</div>
                <div class="desc">Upload a dataset in the Upload tab first.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif "active_run_id" in st.session_state:
        # A run is in flight. Hiding the form (rather than just disabling the button)
        # is what actually prevents a duplicate submission from launching a second,
        # competing training job on the same machine.
        run_id = st.session_state["active_run_id"]
        try:
            poll = requests.get(f"{API_URL}/training/{run_id}", timeout=10)
            poll_data = poll.json()
        except requests.exceptions.RequestException:
            poll_data = {"status": "running"}

        status = poll_data.get("status", "running")
        if status == "completed":
            st.session_state["run"] = poll_data
            del st.session_state["active_run_id"]
            st.toast("Training complete")
            st.rerun()
        elif status == "failed":
            del st.session_state["active_run_id"]
            st.error(poll_data.get("error_message") or "Training failed")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="subhead">Training in progress</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="subcaption">Status: {status}. Four models are being trained and '
                "hyperparameter-tuned — this can take a few minutes depending on dataset size. "
                "You can switch tabs freely; this will keep tracking progress.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            time.sleep(2)
            st.rerun()
    else:
        ds = st.session_state["dataset"]
        columns = list(ds["column_summary"].keys())

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subhead">Configure training</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            target_column = st.selectbox("Target column", columns)
        with c2:
            task_type = st.radio("Task type", ["classification", "regression"], horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subhead">Models that will be trained</div>', unsafe_allow_html=True)
        model_keys = (
            ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
            if task_type == "classification"
            else ["linear_regression", "decision_tree", "random_forest", "xgboost"]
        )
        mcols = st.columns(4)
        for col, key in zip(mcols, model_keys):
            icon_name, name, desc = MODEL_INFO[key]
            with col:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="icon-row">{icon(icon_name, 17)}<span class="name">{name}</span></div>
                        <div class="desc">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("Each model is hyperparameter-tuned with RandomizedSearchCV, then ranked by validation score.")
        st.write("")

        if st.button("Train & tune models"):
            resp = requests.post(
                f"{API_URL}/training/run",
                json={"dataset_id": ds["id"], "target_column": target_column, "task_type": task_type},
            )
            if resp.ok:
                st.session_state["active_run_id"] = resp.json()["id"]
                st.rerun()
            else:
                st.error(resp.text)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_results:
    if "run" not in st.session_state:
        st.markdown(
            f"""
            <div class="empty-state">
                <div class="icon-box">{icon("bar-chart", 20)}</div>
                <div class="title">No results yet</div>
                <div class="desc">Train models in the Train tab to see comparisons here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        run = st.session_state["run"]
        best_key = run["best_model_name"]
        best_icon_name, best_name, best_desc = MODEL_INFO.get(best_key, ("star", best_key, ""))

        st.markdown(
            f"""
            <div class="best-model-banner">
                <div class="icon-box">{icon("star", 18)}</div>
                <div>
                    <div class="label">Best model</div>
                    <div class="name">{best_name}</div>
                    <div class="desc">{best_desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics_df = pd.DataFrame(run["metrics"]).T
        metric_cols = list(metrics_df.columns)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subhead">Model comparison</div>', unsafe_allow_html=True)
        chart_metric = "f1" if "f1" in metric_cols else metric_cols[0]
        chart_metric = st.selectbox("Compare by metric", metric_cols, index=metric_cols.index(chart_metric))

        model_names = [model_display_name(n) for n in metrics_df.index]
        bar_colors = [ACCENT if n == best_key else T["track"] for n in metrics_df.index]
        fig = go.Figure(go.Bar(
            x=model_names, y=metrics_df[chart_metric], marker=dict(color=bar_colors),
            hovertemplate="%{x}<br>" + chart_metric + ": %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(**CHART_TEMPLATE, height=300)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subhead">Full metrics table</div>', unsafe_allow_html=True)
        st.dataframe(
            metrics_df.rename(index=model_display_name).style.highlight_max(axis=0, color="rgba(26,158,107,0.18)"),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if run.get("feature_importance"):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="subhead">Feature importance — {best_name}</div>', unsafe_allow_html=True)
            st.markdown('<div class="subcaption">Which columns influenced the best model\'s predictions the most.</div>', unsafe_allow_html=True)

            fi = run["feature_importance"]
            fi_labels = [k.split("__", 1)[-1] if "__" in k else k for k in fi]
            fi_values = list(fi.values())
            ordered = sorted(zip(fi_labels, fi_values), key=lambda p: p[1])
            fi_labels, fi_values = zip(*ordered)

            n = len(fi_values)
            fi_colors = [f"rgba(79,110,247,{0.4 + 0.6 * (i / max(n - 1, 1))})" for i in range(n)]
            fig2 = go.Figure(go.Bar(
                x=list(fi_values), y=list(fi_labels), orientation="h", marker=dict(color=fi_colors),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig2.update_layout(**{**CHART_TEMPLATE, "xaxis": CHART_TEMPLATE["xaxis"], "yaxis": dict(showgrid=False, zeroline=False, color=T["muted"])})
            fig2.update_layout(height=max(240, 28 * n))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

with tab_predict:
    if "run" not in st.session_state:
        st.markdown(
            f"""
            <div class="empty-state">
                <div class="icon-box">{icon("target", 20)}</div>
                <div class="title">No model ready</div>
                <div class="desc">Train models first, then come back here to predict.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        run = st.session_state["run"]
        best_label = model_display_name(run["best_model_name"])

        predict_mode = st.radio("Prediction mode", ["Single record", "Batch (CSV)"], horizontal=True)

        if predict_mode == "Single record":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="subhead">Predict with {best_label}</div>', unsafe_allow_html=True)
            st.markdown('<div class="subcaption">Enter feature values for a single row to get a prediction.</div>', unsafe_allow_html=True)

            inputs = {}
            cols = st.columns(2)
            for i, col in enumerate(run["feature_columns"]):
                with cols[i % 2]:
                    inputs[col] = st.text_input(col, key=f"predict_{col}")

            if st.button("Predict"):
                resp = requests.post(
                    f"{API_URL}/training/predict",
                    json={"training_run_id": run["id"], "records": [inputs]},
                    headers={"x-api-key": API_KEY},
                )
                if resp.ok:
                    prediction = resp.json()["predictions"][0]
                    st.toast("Prediction ready")
                    st.markdown(
                        f"""
                        <div class="best-model-banner">
                            <div class="icon-box">{icon("target", 18)}</div>
                            <div>
                                <div class="label">Prediction</div>
                                <div class="name">{prediction}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(resp.text)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="subhead">Batch predict with {best_label}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="subcaption">Upload a CSV containing at least these columns: '
                + ", ".join(run["feature_columns"]) + "</div>",
                unsafe_allow_html=True,
            )
            batch_file = st.file_uploader("CSV file", type=["csv"], key="batch_predict_file")

            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                missing_cols = set(run["feature_columns"]) - set(batch_df.columns)
                if missing_cols:
                    st.error(f"Missing required columns: {sorted(missing_cols)}")
                else:
                    st.dataframe(batch_df.head(10), use_container_width=True)
                    if st.button("Predict for all rows"):
                        records = batch_df[run["feature_columns"]].to_dict(orient="records")
                        resp = requests.post(
                            f"{API_URL}/training/predict",
                            json={"training_run_id": run["id"], "records": records},
                            headers={"x-api-key": API_KEY},
                        )
                        if resp.ok:
                            predictions = resp.json()["predictions"]
                            result_df = batch_df.copy()
                            result_df["prediction"] = predictions
                            st.toast(f"{len(predictions)} predictions ready")
                            st.dataframe(result_df, use_container_width=True)
                            st.download_button(
                                "Download predictions (CSV)",
                                data=result_df.to_csv(index=False).encode("utf-8"),
                                file_name="predictions.csv",
                                mime="text/csv",
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
