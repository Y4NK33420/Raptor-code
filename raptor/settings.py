"""Runtime settings read from the environment and an optional ``.env`` file.

Secrets never live in code. Set ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) in the
environment or in a ``.env`` file in the working directory.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# GEMINI_API_KEY, GEMINI_API_KEY2, GEMINI_API_KEY3, ... (GOOGLE_API_KEY* also accepted)
_KEY_NAME = re.compile(r"^(GEMINI|GOOGLE)_API_KEY(\d*)$")


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
        return self.require_gemini_keys()[0]

    def gemini_keys(self, env_file: str | Path | None = ".env") -> list[str]:
        """All configured Gemini keys, in order: ``GEMINI_API_KEY``, ``GEMINI_API_KEY2``, ...

        Values from the process environment override the ``.env`` file. Duplicates are dropped.
        """
        found: dict[str, str] = {}
        if env_file is not None and Path(env_file).is_file():
            from dotenv import dotenv_values

            found.update({k: v for k, v in dotenv_values(env_file).items() if v})
        found.update(os.environ)

        def order(name: str) -> tuple[int, int]:
            m = _KEY_NAME.match(name)
            assert m is not None
            return (0 if m.group(1) == "GEMINI" else 1, int(m.group(2) or 1))

        names = sorted((n for n in found if _KEY_NAME.match(n)), key=order)
        keys = [found[n].strip() for n in names if found[n].strip()]
        if self.gemini_api_key is not None:
            keys.insert(0, self.gemini_api_key.get_secret_value().strip())
        return [k for k in dict.fromkeys(keys) if k]

    def require_gemini_keys(self) -> list[str]:
        keys = self.gemini_keys()
        if not keys:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in the environment or in a .env file "
                "(see .env.example). Extra keys can be added as GEMINI_API_KEY2, GEMINI_API_KEY3, ..."
            )
        return keys


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
