"""Backend application settings.

Environment driven (see .env.example).  ``DATABASE_URL`` selects the
database; tests override it with SQLite.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "TuneFlow API"
    version: str = "0.1.0"
    debug: bool = False

    database_url: str = Field(
        default="postgresql+psycopg2://tuneflow:tuneflow@localhost:5432/tuneflow",
    )
    db_echo: bool = False

    # ML configuration (same files the training scripts use)
    config_files: str = "default.yaml"          # comma-separated YAML names in configs/
    artifact_name: str = "latest"

    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:4173"

    # session handling
    session_timeout_minutes: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def config_file_list(self) -> list[str]:
        return [c.strip() for c in self.config_files.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
