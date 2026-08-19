import uuid

import mlflow
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import get_db
from app.ml.pipelines.training import train_and_compare
from app.models.dataset import Dataset, TrainingRun
from app.schemas.dataset import (
    PredictRequest,
    PredictResponse,
    TrainingRunOut,
    TrainingRunSummary,
    TrainRequest,
)

router = APIRouter(prefix="/training", tags=["training"])

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


@router.post("/run", response_model=TrainingRunOut)
def run_training(req: TrainRequest, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    df = pd.read_csv(dataset.file_path)
    if req.target_column not in df.columns:
        raise HTTPException(400, f"Target column '{req.target_column}' not in dataset")

    run = TrainingRun(id=uuid.uuid4(), dataset_id=dataset.id, status="running")
    db.add(run)
    db.commit()

    try:
        mlflow.set_experiment(f"dataset_{dataset.id}")
        outcome = train_and_compare(df, req.target_column, req.task_type)

        all_metrics = {}
        best_model_uri = None
        for name, res in outcome["results"].items():
            with mlflow.start_run(run_name=name) as mlflow_run:
                mlflow.log_params({"model": name, "task_type": req.task_type, **res["best_params"]})
                mlflow.log_metrics(res["metrics"])
                mlflow.sklearn.log_model(res["pipeline"], artifact_path="model")
                if name == outcome["best_model"]:
                    best_model_uri = f"runs:/{mlflow_run.info.run_id}/model"
                    run.mlflow_run_id = mlflow_run.info.run_id
            all_metrics[name] = res["metrics"]

        run.status = "completed"
        run.best_model_name = outcome["best_model"]
        run.best_model_uri = best_model_uri
        run.feature_columns = outcome["feature_columns"]
        run.metrics = all_metrics

        dataset.target_column = req.target_column
        dataset.task_type = req.task_type

        db.commit()
        db.refresh(run)
        return run
    except Exception as e:
        run.status = "failed"
        db.commit()
        raise HTTPException(500, str(e))


@router.get("/runs", response_model=list[TrainingRunSummary])
def list_training_runs(dataset_id: uuid.UUID | None = None, limit: int = 10, db: Session = Depends(get_db)):
    query = db.query(TrainingRun)
    if dataset_id is not None:
        query = query.filter(TrainingRun.dataset_id == dataset_id)
    return query.order_by(TrainingRun.created_at.desc()).limit(limit).all()


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_training_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(404, "Training run not found")
    return run


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter(TrainingRun.id == req.training_run_id).first()
    if not run or run.status != "completed":
        raise HTTPException(404, "Completed training run not found")

    model = mlflow.sklearn.load_model(run.best_model_uri)
    input_df = pd.DataFrame(req.records)

    missing = set(run.feature_columns) - set(input_df.columns)
    if missing:
        raise HTTPException(400, f"Missing feature columns: {sorted(missing)}")

    predictions = model.predict(input_df[run.feature_columns])
    return {"predictions": predictions.tolist()}
