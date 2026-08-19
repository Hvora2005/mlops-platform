import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def profile_dataframe(df: pd.DataFrame) -> dict:
    summary = {}
    for col in df.columns:
        summary[col] = {
            "dtype": str(df[col].dtype),
            "missing_pct": round(df[col].isna().mean() * 100, 2),
            "n_unique": int(df[col].nunique()),
        }
    return summary


def infer_column_types(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    features = df.drop(columns=[target_column])
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = features.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
