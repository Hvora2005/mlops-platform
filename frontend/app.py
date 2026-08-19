import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MLOps Platform", layout="wide")
st.title("MLOps Platform")

tab_upload, tab_train, tab_results, tab_predict = st.tabs(
    ["1. Upload", "2. Train", "3. Results", "4. Predict"]
)

with tab_upload:
    uploaded = st.file_uploader("Upload a CSV dataset", type="csv")
    if uploaded is not None:
        if st.button("Upload"):
            resp = requests.post(
                f"{API_URL}/datasets/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            )
            if resp.ok:
                st.session_state["dataset"] = resp.json()
                st.success(f"Uploaded: {resp.json()['filename']}")
            else:
                st.error(resp.text)

    if "dataset" in st.session_state:
        ds = st.session_state["dataset"]
        st.subheader("Column Summary")
        st.dataframe(pd.DataFrame(ds["column_summary"]).T)

with tab_train:
    if "dataset" not in st.session_state:
        st.info("Upload a dataset first")
    else:
        ds = st.session_state["dataset"]
        columns = list(ds["column_summary"].keys())
        target_column = st.selectbox("Target column", columns)
        task_type = st.radio("Task type", ["classification", "regression"])

        if st.button("Train models"):
            with st.spinner("Training models..."):
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
                st.success("Training complete")
            else:
                st.error(resp.text)

with tab_results:
    if "run" not in st.session_state:
        st.info("Run training first")
    else:
        run = st.session_state["run"]
        st.subheader(f"Best model: {run['best_model_name']}")
        st.dataframe(pd.DataFrame(run["metrics"]).T)

with tab_predict:
    if "run" not in st.session_state:
        st.info("Run training first")
    else:
        run = st.session_state["run"]
        st.write(f"Using best model: **{run['best_model_name']}**")
        st.write("Enter one value per feature column (comma-separated for multiple rows):")

        inputs = {}
        for col in run["feature_columns"]:
            inputs[col] = st.text_input(col)

        if st.button("Predict"):
            record = {col: val for col, val in inputs.items()}
            resp = requests.post(
                f"{API_URL}/training/predict",
                json={"training_run_id": run["id"], "records": [record]},
            )
            if resp.ok:
                st.success(f"Prediction: {resp.json()['predictions']}")
            else:
                st.error(resp.text)
