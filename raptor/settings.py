"""Runtime settings read from the environment and an optional ``.env`` file.

Secrets never live in code. Set ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) in the
environment or in a ``.env`` file in the working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="RAPTOR_")

    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "RAPTOR_GEMINI_API_KEY"),
    )
    cache_dir: Path = Path(".cache/raptor")
    cache_enabled: bool = True
    # Requests per minute. Defaults are conservative so the free tier works out of the box.
    gemini_rpm: int = 15
    gemini_embed_rpm: int = 100
    ollama_host: str = "http://localhost:11434"
    log_level: str = "INFO"

    def require_gemini_key(self) -> str:
        if self.gemini_api_key is None or not self.gemini_api_key.get_secret_value().strip():
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in the environment or in a .env file "
                "(see .env.example)."
            )
        return self.gemini_api_key.get_secret_value().strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
