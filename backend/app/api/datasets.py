import os
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import get_db
from app.ml.pipelines.preprocessing import profile_dataframe
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetOut, DatasetPreview

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetOut)
def upload_dataset(file: UploadFile, db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    os.makedirs(settings.upload_dir, exist_ok=True)
    dataset_id = uuid.uuid4()
    file_path = os.path.join(settings.upload_dir, f"{dataset_id}.csv")

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    df = pd.read_csv(file_path)
    summary = profile_dataframe(df)

    dataset = Dataset(
        id=dataset_id,
        filename=file.filename,
        file_path=file_path,
        n_rows=len(df),
        n_cols=len(df.columns),
        column_summary=summary,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return dataset


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(dataset_id: uuid.UUID, rows: int = 10, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    df = pd.read_csv(dataset.file_path).head(rows)
    df = df.where(pd.notnull(df), None)
    return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
