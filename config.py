"""App configuration from environment."""
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    allowed_origins: str = "*"
    chroma_persist_dir: Path = Path("./chroma_db")
    port: int = 8000
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    @property
    def origins_list(self) -> list[str]:
        parts = [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        return parts if parts else ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
