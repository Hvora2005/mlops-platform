import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID

from app.core.db import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    n_rows = Column(Integer)
    n_cols = Column(Integer)
    column_summary = Column(JSON)
    target_column = Column(String, nullable=True)
    task_type = Column(String, nullable=True)  # "classification" | "regression"
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    mlflow_run_id = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending | running | completed | failed
    best_model_name = Column(String, nullable=True)
    best_model_uri = Column(String, nullable=True)
    feature_columns = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
