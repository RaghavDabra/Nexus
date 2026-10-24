from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://nexus:nexus_dev@localhost:5432/nexus"
    redis_url: str = "redis://localhost:6379"

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    log_level: str = "INFO"
    gateway_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
