"""Smoke-test the raptor MCP server through the MCP SDK's stdio client.

    python scripts/mcp_check.py data/indexes/raptor
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
CALLS = [
    ("search_code", {"query": "resume incremental update of changed files", "k": 3}),
    ("get_symbol", {"symbol": "CodeIndex.update"}),
    ("get_callers", {"symbol": "CodeIndex.update"}),
    ("get_callees", {"symbol": "CodeIndex.update"}),
    ("explain_module", {"path": "raptor/code/search.py"}),
    ("ask_codebase", {"question": "How does graph expansion work?", "k": 5}),
]


async def main(index: str) -> None:
    exe = ROOT / ".venv" / ("Scripts/raptor.exe" if sys.platform == "win32" else "bin/raptor")
    params = StdioServerParameters(command=str(exe), args=["code", "serve-mcp", index])
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        tools = [t.name for t in (await session.list_tools()).tools]
        print("tools:", tools)
        for name, args in CALLS:
            res = await session.call_tool(name, args)
            text = res.content[0].text if res.content else ""
            print(f"- {name}: error={res.is_error} chars={len(text)}")
            print("  " + text[:200].replace("\n", " "))
            assert not res.is_error and text, name


asyncio.run(main(sys.argv[1]))
