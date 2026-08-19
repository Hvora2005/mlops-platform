from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://mlops:mlops@localhost:5432/mlops"
    mlflow_tracking_uri: str = "http://localhost:5000"
    upload_dir: str = "./data/uploads"

    class Config:
        env_file = ".env"


settings = Settings()
