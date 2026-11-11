"""HTTP API for the RAPTOR web UI (FastAPI).

Install with ``pip install raptor-code[server]`` and run ``raptor serve``. The service is a
thin layer over :mod:`raptor.pipeline`, :mod:`raptor.eval` and :mod:`raptor.code`.
"""

from .config import ServerSettings

__all__ = ["ServerSettings", "create_app"]


def create_app(*args, **kwargs):
    """Lazy alias for :func:`raptor.server.app.create_app` (keeps FastAPI optional)."""
    from .app import create_app as _create

    return _create(*args, **kwargs)
