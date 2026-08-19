import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from app.ml.pipelines.preprocessing import build_preprocessor, infer_column_types

CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(eval_metric="logloss"),
}

REGRESSION_MODELS = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
    "xgboost": XGBRegressor(),
}


def train_and_compare(df: pd.DataFrame, target_column: str, task_type: str) -> dict:
    numeric_cols, categorical_cols = infer_column_types(df, target_column)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    results = {}

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        if task_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds, average="weighted"),
            }
        else:
            metrics = {
                "rmse": mean_squared_error(y_test, preds, squared=False),
                "r2": r2_score(y_test, preds),
            }

        results[name] = {"pipeline": pipeline, "metrics": metrics}

    ranking_key = "f1" if task_type == "classification" else "r2"
    best_name = max(results, key=lambda n: results[n]["metrics"][ranking_key])

    return {"results": results, "best_model": best_name}
