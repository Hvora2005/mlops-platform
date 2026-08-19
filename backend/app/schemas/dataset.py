from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DatasetOut(BaseModel):
    id: UUID
    filename: str
    n_rows: int
    n_cols: int
    column_summary: dict

    class Config:
        from_attributes = True


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[dict]


class TrainingRunSummary(BaseModel):
    id: UUID
    status: str
    best_model_name: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class TrainRequest(BaseModel):
    dataset_id: UUID
    target_column: str
    task_type: str  # "classification" | "regression"


class TrainingRunOut(BaseModel):
    id: UUID
    status: str
    best_model_name: str | None
    feature_columns: list[str] | None
    metrics: dict | None

    class Config:
        from_attributes = True


class PredictRequest(BaseModel):
    training_run_id: UUID
    records: list[dict]


class PredictResponse(BaseModel):
    predictions: list
