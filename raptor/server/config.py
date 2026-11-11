"""Settings for the API server (paths, model mode, limits)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    """Where the server looks for data, and how it runs models.

    Paths are resolved against the working directory, so ``raptor serve`` is normally run
    from the repository root.
    """

    #: Read-only demo indexes shipped with the repo.
    demo_dir: Path = Path("demo/trees")
    #: Indexes built through the API are written here.
    indexes_dir: Path = Path("artifacts/indexes")
    #: Experiment runs written by ``raptor eval`` (Phase 2 artifact contract).
    runs_dirs: list[Path] = Field(
        default_factory=lambda: [Path("artifacts/runs"), Path("tests/fixtures/runs")]
    )
    #: Saved code indexes (``CodeIndex.save``).
    code_dir: Path = Path("artifacts/code")
    #: Repositories indexed on the fly by the code stub when ``raptor.code`` is not installed.
    code_stub_repos: list[Path] = Field(default_factory=lambda: [Path("tests/fixtures/code_repo")])
    quality_dir: Path = Path("data/quality")
    quality_split: str = "dev"
    #: Cached UMAP projections and other derived data.
    derived_dir: Path = Path("artifacts/.server")
    #: Built frontend; served at ``/`` when it exists.
    web_dist: Path = Path("web/dist")

    #: Use deterministic offline models (no API calls). For demos, tests and Playwright.
    fake_models: bool = False
    #: Seconds between streamed tokens / summaries in fake mode, so the UI animates.
    fake_delay: float = 0.0
    projection_method: Literal["umap", "pca"] = "umap"
    #: Allow cross-origin requests from the Vite dev server.
    cors_origins: list[str] = Field(default_factory=list)
    max_upload_bytes: int = 5_000_000
