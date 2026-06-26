"""Configuration management loading from environment variables and yaml configurations."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Pydantic model representing application-wide environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_dir: str = Field(default="logs", validation_alias="LOG_DIR")

    device: str = Field(default="cuda", validation_alias="DEVICE")

    wandb_api_key: str | None = Field(default=None, validation_alias="WANDB_API_KEY")
    wandb_project: str = Field(default="brain-mri-analysis", validation_alias="WANDB_PROJECT")
    wandb_entity: str | None = Field(default=None, validation_alias="WANDB_ENTITY")

    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    api_debug: bool = Field(default=False, validation_alias="API_DEBUG")

    streamlit_port: int = Field(default=8501, validation_alias="STREAMLIT_PORT")


def load_yaml_config(file_path: Path | str) -> Dict[str, Any]:
    """Helper to safely load a YAML configuration file."""
    path = Path(file_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# Global instance of settings
settings = AppSettings()
