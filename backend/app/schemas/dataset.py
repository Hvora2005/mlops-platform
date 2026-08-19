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


class TrainRequest(BaseModel):
    dataset_id: UUID
    target_column: str
    task_type: str  # "classification" | "regression"


class TrainingRunOut(BaseModel):
    id: UUID
    status: str
    best_model_name: str | None
    metrics: dict | None

    class Config:
        from_attributes = True
