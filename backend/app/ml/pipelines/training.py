import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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

# Hyperparameter grids, keyed by model name. Params are prefixed with "model__"
# since each estimator sits inside a Pipeline step named "model".
CLASSIFICATION_PARAM_GRIDS = {
    "logistic_regression": {"model__C": [0.01, 0.1, 1.0, 10.0]},
    "decision_tree": {"model__max_depth": [3, 5, 10, None], "model__min_samples_split": [2, 5, 10]},
    "random_forest": {"model__n_estimators": [100, 200, 300], "model__max_depth": [5, 10, None]},
    "xgboost": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.01, 0.1, 0.3]},
}

REGRESSION_PARAM_GRIDS = {
    "linear_regression": {},
    "decision_tree": {"model__max_depth": [3, 5, 10, None], "model__min_samples_split": [2, 5, 10]},
    "random_forest": {"model__n_estimators": [100, 200, 300], "model__max_depth": [5, 10, None]},
    "xgboost": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.01, 0.1, 0.3]},
}


def train_and_compare(df: pd.DataFrame, target_column: str, task_type: str, tune: bool = True) -> dict:
    numeric_cols, categorical_cols = infer_column_types(df, target_column)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    param_grids = CLASSIFICATION_PARAM_GRIDS if task_type == "classification" else REGRESSION_PARAM_GRIDS
    results = {}

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        grid = param_grids.get(name, {})
        best_params = {}

        if tune and grid:
            n_candidates = min(10, _grid_size(grid))
            search = RandomizedSearchCV(
                pipeline, param_distributions=grid, n_iter=n_candidates,
                cv=3, random_state=42, n_jobs=-1,
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            best_params = search.best_params_
        else:
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

        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "best_params": best_params,
            "feature_importance": _extract_feature_importance(pipeline),
        }

    ranking_key = "f1" if task_type == "classification" else "r2"
    best_name = max(results, key=lambda n: results[n]["metrics"][ranking_key])

    return {"results": results, "best_model": best_name, "feature_columns": list(X.columns)}


def _grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _extract_feature_importance(pipeline: Pipeline, top_n: int = 20) -> dict | None:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = abs(coef[0]) if coef.ndim > 1 else abs(coef)
    else:
        return None

    ranked = sorted(zip(feature_names, values.tolist()), key=lambda pair: -pair[1])
    return dict(ranked[:top_n])
