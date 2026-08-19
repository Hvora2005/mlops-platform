# MLOps Platform

A self-hosted, mini Vertex AI / Azure ML: upload a CSV, get it auto-cleaned, train and compare several models with hyperparameter tuning, and serve the best one behind a prediction API — all tracked in MLflow.

## Features

- Upload any CSV dataset via a REST API or the Streamlit UI
- Automatic profiling (dtypes, missing values, cardinality) and cleaning (imputation, encoding, scaling) via scikit-learn pipelines
- Trains and compares Logistic/Linear Regression, Decision Tree, Random Forest, and XGBoost
- Hyperparameter tuning with `RandomizedSearchCV`
- Every run and model logged to MLflow (params, metrics, artifacts)
- Best model auto-selected and served via a `/predict` endpoint
- Dockerized (Postgres + MLflow + API + Streamlit UI) with a GitHub Actions CI pipeline

## Tech stack

Python · FastAPI · scikit-learn · XGBoost · MLflow · PostgreSQL · Streamlit · Docker · GitHub Actions

## Architecture

```
frontend (Streamlit)  ──HTTP──>  backend (FastAPI)  ──>  PostgreSQL   (datasets, run metadata)
                                        │
                                        └──────────────>  MLflow      (experiments, model artifacts)
```

## Project layout

```
backend/
  app/
    api/            REST endpoints (datasets, training, predict)
    core/           config + db session
    ml/pipelines/   preprocessing + training/tuning logic
    models/         SQLAlchemy models
    schemas/        Pydantic request/response schemas
  tests/
frontend/
  app.py            Streamlit UI: upload -> train -> results -> predict
docker/             Dockerfiles for backend + frontend
docker-compose.yml  Postgres + MLflow + api + frontend
.github/workflows/  CI: tests + image builds
```

## Running locally with Docker (recommended)

```bash
docker compose up --build
```

- API: http://localhost:8000 (docs at `/docs`)
- Streamlit UI: http://localhost:8501
- MLflow UI: http://localhost:5000

## Running without Docker

Requires Python 3.11 (newer ML packages don't yet ship wheels for 3.12/3.13 on all platforms) and a running Postgres instance.

```bash
# backend
cd backend
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
copy ..\.env.example .env   # then edit DATABASE_URL / MLFLOW_TRACKING_URI
uvicorn app.main:app --reload

# frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## API overview

| Endpoint | Method | Description |
|---|---|---|
| `/datasets/upload` | POST | Upload a CSV, get back column profiling |
| `/datasets/{id}` | GET | Fetch dataset metadata |
| `/training/run` | POST | Train + tune all models, log to MLflow, pick the best |
| `/training/{run_id}` | GET | Fetch a training run's status/metrics |
| `/training/predict` | POST | Predict using a completed run's best model |

## Testing

```bash
cd backend
pytest tests -q
```

## Roadmap

- API key auth on the prediction endpoint
- Model registry promotion (staging -> production) via MLflow Model Registry
- Auto-generated PDF/HTML comparison report
- React frontend as an alternative to Streamlit

## License

MIT — see [LICENSE](LICENSE).
