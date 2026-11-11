"""``raptor serve``: run the API and the built web UI from one process."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer


def serve(
    port: Annotated[int, typer.Option("--port", "-p")] = 8765,
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    fake_models: Annotated[
        bool,
        typer.Option(
            "--fake-models", help="Deterministic offline models: no API calls or quota needed."
        ),
    ] = False,
    fake_delay: Annotated[
        float, typer.Option(help="Seconds per streamed token with --fake-models (animates the UI).")
    ] = 0.03,
    dev: Annotated[
        bool,
        typer.Option(
            "--dev", help="Reload on code changes and allow the Vite dev server (port 5173)."
        ),
    ] = False,
    web_dist: Annotated[Path, typer.Option(help="Built frontend directory.")] = Path("web/dist"),
) -> None:
    """Serve the RAPTOR web UI and API."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise typer.BadParameter("`raptor serve` needs `pip install raptor-code[server]`") from exc
    import logging

    from .config import ServerSettings

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "google_genai", "numba", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = ServerSettings(
        fake_models=fake_models,
        fake_delay=fake_delay if fake_models else 0.0,
        web_dist=web_dist,
        cors_origins=["http://localhost:5173", "http://127.0.0.1:5173"] if dev else [],
    )
    if not (web_dist / "index.html").is_file():
        typer.echo(f"note: {web_dist} has no build; run `npm run build` in web/ (API still works)")
    mode = "fake models (offline)" if fake_models else "Gemini"
    typer.echo(f"RAPTOR UI on http://{host}:{port}  [{mode}]")
    if dev:
        os.environ["RAPTOR_SERVER_SETTINGS"] = settings.model_dump_json()
        uvicorn.run(
            "raptor.server.app:app_from_env",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=["raptor"],
        )
    else:
        from .app import create_app

        uvicorn.run(create_app(settings), host=host, port=port)
