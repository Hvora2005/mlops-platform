from fastapi import FastAPI

from app.api import datasets, training
from app.core.db import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="MLOps Platform")

app.include_router(datasets.router)
app.include_router(training.router)


@app.get("/health")
def health():
    return {"status": "ok"}
