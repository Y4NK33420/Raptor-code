"""Turn parsed files into leaf chunks (one per symbol, with splitting and merging)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from .config import ChunkConfig
from .languages.base import ImportRef, ParsedFile, Symbol

SEP = " › "
MAX_HEADER_IMPORTS = 8


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Leaf:
    """A leaf chunk before it becomes a tree node."""

    path: str
    start_line: int
    end_line: int
    symbol: str  # qualname, "a, b" for merged leaves, "<module>" for module-level code
    symbol_kind: str  # function | method | class | module | merged
    code: str
    text: str = ""  # contextual text that is embedded and shown to readers
    language: str = "python"
    signature: str = ""
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    class_name: str | None = None  # top-level class this leaf belongs to
    symbols: list[str] = field(default_factory=list)  # every qualname covered
    part: tuple[int, int] | None = None

    @property
    def content_hash(self) -> str:
        return short_hash(self.code)

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "kind": "symbol",
            "symbol_kind": self.symbol_kind,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "symbols": self.symbols,
            "signature": self.signature,
            "docstring": self.docstring,
            "decorators": self.decorators,
            "language": self.language,
            "content_hash": self.content_hash,
            "class_name": self.class_name,
        }
        if self.part is not None:
            meta["part"] = list(self.part)
        return meta


def top_class(sym: Symbol) -> str | None:
    if sym.kind == "class" or sym.parent:
        return sym.qualname.split(".")[0]
    return None


def _ntok(tok: tiktoken.Encoding, text: str) -> int:
    return len(tok.encode(text, disallowed_special=()))


def _units(parsed: ParsedFile, lines: list[str]) -> list[Leaf]:
    units: list[Leaf] = []
    for sym in parsed.symbols:
        cls = top_class(sym)
        if sym.kind == "class":
            end = sym.header_end_line or sym.start_line
            code = sym.text or "\n".join(lines[sym.start_line - 1 : end])
            units.append(
                Leaf(
                    parsed.path,
                    sym.start_line,
                    max(sym.start_line, end),
                    sym.qualname,
                    "class",
                    code,
                    language=parsed.language,
                    signature=sym.signature,
                    docstring=sym.docstring,
                    decorators=sym.decorators,
                    class_name=cls,
                    symbols=[sym.qualname],
                )
            )
        else:
            units.append(
                Leaf(
                    parsed.path,
                    sym.start_line,
                    sym.end_line,
                    sym.qualname,
                    sym.kind,
                    sym.text,
                    language=parsed.language,
                    signature=sym.signature,
                    docstring=sym.docstring,
                    decorators=sym.decorators,
                    class_name=cls,
                    symbols=[sym.qualname],
                )
            )
    for a, b in parsed.module_code:
        units.append(
            Leaf(
                parsed.path,
                a,
                b,
                "<module>",
                "module",
                "\n".join(lines[a - 1 : b]),
                language=parsed.language,
                docstring=parsed.module_docstring if a == 1 else "",
                symbols=["<module>"],
            )
        )
    units.sort(key=lambda u: (u.start_line, u.end_line))
    return units


def _split(unit: Leaf, tok: tiktoken.Encoding, max_tokens: int) -> list[Leaf]:
    if _ntok(tok, unit.code) <= max_tokens:
        return [unit]
    code_lines = unit.code.split("\n")
    parts: list[tuple[int, int]] = []  # [start, end) indices into code_lines
    start, used = 0, 0
    for i, line in enumerate(code_lines):
        n = _ntok(tok, line) + 1
        if used + n > max_tokens and i > start:
            parts.append((start, i))
            start, used = i, 0
        used += n
    parts.append((start, len(code_lines)))
    out = []
    for k, (a, b) in enumerate(parts):
        out.append(
            Leaf(
                unit.path,
                unit.start_line + a,
                unit.start_line + b - 1,
                unit.symbol,
                unit.symbol_kind,
                "\n".join(code_lines[a:b]),
                language=unit.language,
                signature=unit.signature,
                docstring=unit.docstring if k == 0 else "",
                decorators=unit.decorators,
                class_name=unit.class_name,
                symbols=list(unit.symbols),
                part=(k + 1, len(parts)),
            )
        )
    return out


def _merge(units: list[Leaf], tok: tiktoken.Encoding, cfg: ChunkConfig) -> list[Leaf]:
    """Merge runs of adjacent tiny units that share a class (or are all module level)."""
    if cfg.min_symbol_tokens <= 0:
        return units
    out: list[Leaf] = []
    run: list[Leaf] = []
    run_tokens = 0

    def flush() -> None:
        nonlocal run, run_tokens
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
        else:
            names = [s for u in run for s in u.symbols]
            out.append(
                Leaf(
                    run[0].path,
                    run[0].start_line,
                    run[-1].end_line,
                    ", ".join(dict.fromkeys(names)),
                    "merged",
                    "\n\n".join(u.code for u in run),
                    language=run[0].language,
                    signature="; ".join(u.signature for u in run if u.signature),
                    docstring="\n".join(u.docstring for u in run if u.docstring),
                    class_name=run[0].class_name,
                    symbols=list(dict.fromkeys(names)),
                )
            )
        run, run_tokens = [], 0

    for u in units:
        n = _ntok(tok, u.code)
        tiny = n < cfg.min_symbol_tokens and u.part is None
        if not tiny:
            flush()
            out.append(u)
            continue
        if run and (run[-1].class_name != u.class_name or run_tokens + n > cfg.max_symbol_tokens):
            flush()
        run.append(u)
        run_tokens += n
        if run_tokens >= cfg.min_symbol_tokens:
            flush()
    flush()
    return out


def relevant_imports(code: str, imports: list[ImportRef]) -> list[str]:
    words = set(re.findall(r"[A-Za-z_]\w*", code))
    picked: list[str] = []
    for imp in imports:
        if imp.bound_name in words and imp.text not in picked:
            picked.append(imp.text)
        if len(picked) >= MAX_HEADER_IMPORTS:
            break
    return picked


def header(leaf: Leaf, imports: list[ImportRef]) -> str:
    crumbs = [leaf.path]
    if leaf.class_name and leaf.symbol_kind != "class":
        crumbs.append(f"class {leaf.class_name}")
    if leaf.symbol_kind in ("function", "method", "class") and leaf.signature:
        crumbs.append(leaf.signature)
    elif leaf.symbol_kind == "merged":
        crumbs.append(leaf.symbol)
    elif leaf.symbol_kind == "module":
        crumbs.append("module-level code")
    first = "# " + SEP.join(crumbs)
    if leaf.part is not None:
        first += f" (part {leaf.part[0]}/{leaf.part[1]})"
    lines = [first]
    imps = relevant_imports(leaf.code, imports)
    if imps:
        lines.append("# uses: " + "; ".join(imps))
    return "\n".join(lines)


def make_leaves(
    parsed: ParsedFile, source: str, tok: tiktoken.Encoding, cfg: ChunkConfig
) -> list[Leaf]:
    """Leaf chunks for one parsed file, in source order."""
    lines = source.splitlines()
    units = _units(parsed, lines)
    split: list[Leaf] = []
    for u in units:
        split.extend(_split(u, tok, cfg.max_symbol_tokens))
    leaves = _merge(split, tok, cfg)
    for leaf in leaves:
        leaf.text = (
            f"{header(leaf, parsed.imports)}\n{leaf.code}" if cfg.contextual_header else leaf.code
        )
    return leaves
