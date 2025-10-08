from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
from pathlib import Path
import os

class Settings(BaseSettings):
    # .env
    MINIO_ENDPOINT: str = ""
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "haybici"
    MLFLOW_TRACKING_URI: str = ""
    MLFLOW_EXPERIMENT: str = "haybici-exp"
    LOCAL_TZ: str = "America/Argentina/Buenos_Aires"

    # YAML
    yaml_cfg: dict = Field(default_factory=dict)

    class Config:
        env_file = ".env"
        extra = "ignore"

def get_settings() -> Settings:
    s = Settings()
    with open(Path("config/config.yaml"), "r") as f:
        s.yaml_cfg = yaml.safe_load(f)
    return s
