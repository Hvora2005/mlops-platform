import uuid

import mlflow
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import SessionLocal, get_db
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


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid or missing API key")


@router.post("/run", response_model=TrainingRunOut, status_code=202)
def run_training(req: TrainRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    try:
        columns = pd.read_csv(dataset.file_path, nrows=0).columns
    except Exception as e:
        raise HTTPException(400, f"Could not read dataset: {e}")
    if req.target_column not in columns:
        raise HTTPException(400, f"Target column '{req.target_column}' not in dataset")

    run = TrainingRun(id=uuid.uuid4(), dataset_id=dataset.id, status="pending")
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(
        _execute_training, run.id, dataset.id, dataset.file_path, req.target_column, req.task_type
    )
    return run


def _execute_training(run_id: uuid.UUID, dataset_id: uuid.UUID, file_path: str, target_column: str, task_type: str):
    """Runs in-process after the HTTP response is sent, so the /run request returns immediately.
    Uses its own DB session since the request-scoped one is closed by then."""
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        run.status = "running"
        db.commit()

        df = pd.read_csv(file_path)
        mlflow.set_experiment(f"dataset_{dataset_id}")
        outcome = train_and_compare(df, target_column, task_type)

        all_metrics = {}
        best_model_uri = None
        best_feature_importance = None
        for name, res in outcome["results"].items():
            with mlflow.start_run(run_name=name) as mlflow_run:
                mlflow.log_params({"model": name, "task_type": task_type, **res["best_params"]})
                mlflow.log_metrics(res["metrics"])
                mlflow.sklearn.log_model(res["pipeline"], artifact_path="model")
                if res["feature_importance"]:
                    mlflow.log_dict(res["feature_importance"], "feature_importance.json")
                if name == outcome["best_model"]:
                    best_model_uri = f"runs:/{mlflow_run.info.run_id}/model"
                    run.mlflow_run_id = mlflow_run.info.run_id
                    best_feature_importance = res["feature_importance"]
            all_metrics[name] = res["metrics"]

        run.status = "completed"
        run.best_model_name = outcome["best_model"]
        run.best_model_uri = best_model_uri
        run.feature_columns = outcome["feature_columns"]
        run.metrics = all_metrics
        run.feature_importance = best_feature_importance

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        dataset.target_column = target_column
        dataset.task_type = task_type

        db.commit()
    except Exception as e:
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = "failed"
            run.error_message = str(e)
            db.commit()
    finally:
        db.close()


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


@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter(TrainingRun.id == req.training_run_id).first()
    if not run or run.status != "completed":
        raise HTTPException(404, "Completed training run not found")

    model = mlflow.sklearn.load_model(run.best_model_uri)
    input_df = pd.DataFrame(req.records)

    missing = set(run.feature_columns) - set(input_df.columns)
    if missing:
        raise HTTPException(400, f"Missing feature columns: {sorted(missing)}")

    try:
        predictions = model.predict(input_df[run.feature_columns])
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"Invalid input data: {e}")
    return {"predictions": predictions.tolist()}
