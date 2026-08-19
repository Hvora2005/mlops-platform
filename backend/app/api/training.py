import uuid

import mlflow
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import get_db
from app.ml.pipelines.training import train_and_compare
from app.models.dataset import Dataset, TrainingRun
from app.schemas.dataset import TrainingRunOut, TrainRequest

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
        for name, res in outcome["results"].items():
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "task_type": req.task_type})
                mlflow.log_metrics(res["metrics"])
                mlflow.sklearn.log_model(res["pipeline"], artifact_path="model")
            all_metrics[name] = res["metrics"]

        run.status = "completed"
        run.best_model_name = outcome["best_model"]
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


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_training_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(404, "Training run not found")
    return run
