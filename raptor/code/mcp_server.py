"""MCP server exposing a code index to Claude Code, IDEs and other MCP clients.

Run it over stdio with ``raptor code serve-mcp <index_dir>`` (see docs/CODE.md for
registering it with ``claude mcp add``).

Tools: ``search_code``, ``ask_codebase``, ``get_symbol``, ``explain_module``,
``get_callers`` and ``get_callees``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .index import CodeIndex
from .types import CodeHit

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "Tools over a RAPTOR index of one code repository. Use search_code to find relevant "
    "code (exact identifiers or natural language), get_symbol to read a symbol with its "
    "callers and callees, explain_module for a summary of a file or directory, and "
    "ask_codebase for a cited answer to a broader question. Locations are path:start-end."
)
TEXT_LIMIT = 4000


def _server_class() -> Any:
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:  # mcp 1.x
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined,unused-ignore]
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError("The MCP server needs `pip install raptor-code[mcp]`") from exc
        return FastMCP


def _hit(h: CodeHit, *, text: bool = True, limit: int = TEXT_LIMIT) -> dict[str, Any]:
    d: dict[str, Any] = {
        "node": h.node_index,
        "kind": h.kind,
        "location": h.location,
        "symbol": h.symbol,
        "score": round(h.score, 5),
    }
    if text:
        d["text"] = h.text if len(h.text) <= limit else h.text[:limit] + "\n... (truncated)"
    return d


def _resolve(index: CodeIndex, symbol: str) -> tuple[int | None, list[CodeHit]]:
    """Node for ``symbol``: a node index, a (qualified) name or a ``path:line`` location."""
    s = symbol.strip()
    if s.isdigit() and int(s) in index.tree.all_nodes:
        return int(s), []
    if ":" in s and s.rsplit(":", 1)[1].split("-")[0].isdigit():
        path, line_s = s.rsplit(":", 1)
        line = int(line_s.split("-")[0])
        best: tuple[int, int] | None = None
        for n in index.tree.leaf_nodes.values():
            m = n.metadata
            if m.get("path") == path and m["start_line"] <= line <= m["end_line"]:
                span = m["end_line"] - m["start_line"]
                if best is None or span < best[0]:
                    best = (span, n.index)
        if best is not None:
            return best[1], []
    matches = index.find_symbol(s)
    if not matches:
        hits = index.search(s, k=5)
        return (hits[0].node_index if hits else None), hits[1:]
    # Prefer classes and whole symbols over parts / module code.
    matches.sort(key=lambda h: (h.kind not in ("class", "symbol"), h.kind != "class", h.layer))
    return matches[0].node_index, matches[1:]


def create_server(index: CodeIndex) -> Any:
    server = _server_class()("raptor-code", instructions=INSTRUCTIONS)

    @server.tool()
    def search_code(query: str, k: int = 10) -> dict[str, Any]:
        """Hybrid (lexical + semantic) search over the code index.

        Works for exact identifiers ("Session.request") and natural-language questions
        ("where are redirects handled"). Returns ranked nodes (symbols, classes, files,
        directories, concepts) with their location (path:start-end) and text.
        """
        hits = index.search(query, k=max(1, min(k, 50)))
        return {"query": query, "results": [_hit(h, limit=2000) for h in hits]}

    @server.tool()
    def ask_codebase(question: str, k: int = 8) -> dict[str, Any]:
        """Answer a question about the codebase, citing path:start-end locations."""
        ans = index.ask(question, k=max(1, min(k, 30)))
        return {
            "answer": ans.answer,
            "citations": [_hit(c, text=False) for c in ans.citations],
        }

    @server.tool()
    def get_symbol(symbol: str) -> dict[str, Any]:
        """Source and context of a symbol.

        ``symbol`` may be a name ("apply_discount"), a qualified name ("Cart.total"), a
        location ("shop/cart.py:32") or a node id. Returns its text, children, parents,
        callers and callees.
        """
        idx, others = _resolve(index, symbol)
        if idx is None:
            return {"error": f"no symbol matching {symbol!r}"}
        info = index.symbol(idx)
        return {
            "symbol": _hit(info["hit"]),
            "children": [_hit(h, text=False) for h in info["children"]],
            "parents": [_hit(h, text=False) for h in info["parents"]],
            "callers": [_hit(h, text=False) for h in info["callers"]],
            "callees": [_hit(h, text=False) for h in info["callees"]],
            "other_matches": [_hit(h, text=False) for h in others[:10]],
        }

    @server.tool()
    def explain_module(path: str) -> dict[str, Any]:
        """Summary of a file or directory (path like "raptor/code" or dotted module name),
        or of the whole repository for "."; lists its children."""
        idx = index.module_node(path)
        if idx is None:
            return {"error": f"no file or directory {path!r} in the index"}
        info = index.symbol(idx)
        return {
            "module": _hit(info["hit"]),
            "children": [_hit(h, text=False) for h in info["children"]],
            "concepts": [_hit(h, text=False) for h in info["parents"] if h.kind == "concept"],
        }

    @server.tool()
    def get_callers(symbol: str) -> dict[str, Any]:
        """Functions/methods that call ``symbol`` (best-effort static resolution).
        For a file, the files that import it."""
        idx, _ = _resolve(index, symbol)
        if idx is None:
            return {"error": f"no symbol matching {symbol!r}"}
        info = index.symbol(idx)
        return {
            "symbol": _hit(info["hit"], text=False),
            "callers": [_hit(h, limit=1500) for h in info["callers"]],
        }

    @server.tool()
    def get_callees(symbol: str) -> dict[str, Any]:
        """Functions/methods/classes that ``symbol`` calls (best-effort static resolution).
        For a file, the files it imports."""
        idx, _ = _resolve(index, symbol)
        if idx is None:
            return {"error": f"no symbol matching {symbol!r}"}
        info = index.symbol(idx)
        return {
            "symbol": _hit(info["hit"], text=False),
            "callees": [_hit(h, limit=1500) for h in info["callees"]],
        }

    return server


def serve(index_dir: str | Path) -> None:
    """Load the index at ``index_dir`` and serve it over stdio."""
    index = CodeIndex.load(index_dir)
    logger.info("Serving %r over MCP stdio", index)
    create_server(index).run("stdio")
