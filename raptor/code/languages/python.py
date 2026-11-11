"""Python symbol extraction with tree-sitter."""

from __future__ import annotations

import ast
import re
import threading
from typing import Any

from .base import EXTRACTORS, ImportRef, LanguageExtractor, ParsedFile, Symbol

_SIMPLE_CALLEE = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)*$")
_SOURCE_ROOTS = ("src", "lib", "python")


def _load_language() -> Any:
    try:
        import tree_sitter_python
        from tree_sitter import Language

        return Language(tree_sitter_python.language())
    except ImportError:
        pass
    try:
        from tree_sitter_language_pack import get_language

        return get_language("python")
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError(
            "Code indexing needs tree-sitter: `pip install raptor-code[code]`"
        ) from exc


_LOCAL = threading.local()


def _parser() -> Any:
    parser = getattr(_LOCAL, "parser", None)
    if parser is None:
        from tree_sitter import Parser

        parser = _LOCAL.parser = Parser(_load_language())
    return parser


def _text(node: Any) -> str:
    return node.text.decode("utf-8", errors="replace")


def _line(node: Any) -> int:
    return int(node.start_point[0]) + 1


def _end_line(node: Any) -> int:
    row, col = node.end_point[0], node.end_point[1]
    # A node ending at column 0 ends on the previous line (trailing newline).
    return int(row) + (1 if col > 0 else 0) or 1


def _string_value(node: Any) -> str:
    raw = _text(node)
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return str(value)
    except (ValueError, SyntaxError):
        return raw.strip("rRbBuUfF").strip("'\"")


def _docstring(body: Any) -> str:
    """Docstring of a ``block``/``module`` node, if its first statement is a string."""
    if body is None:
        return ""
    for child in body.named_children:
        if child.type == "comment":
            continue
        if child.type == "expression_statement" and child.named_children:
            first = child.named_children[0]
            if first.type in ("string", "concatenated_string"):
                return _clean_doc(_string_value(first))
        return ""
    return ""


def _clean_doc(doc: str) -> str:
    import inspect

    return inspect.cleandoc(doc).strip()


def _unwrap(node: Any) -> tuple[Any, list[str], Any]:
    """Return ``(definition, decorators, outer_node)`` for a possibly decorated definition."""
    if node.type == "decorated_definition":
        decorators = [_text(c).lstrip("@").strip() for c in node.children if c.type == "decorator"]
        definition = node.child_by_field_name("definition")
        return definition, decorators, node
    return node, [], node


def _signature(definition: Any, source: bytes) -> str:
    body = definition.child_by_field_name("body")
    end = body.start_byte if body is not None else definition.end_byte
    sig = source[definition.start_byte : end].decode("utf-8", errors="replace").strip()
    sig = re.sub(r"\s+", " ", sig).rstrip(":").strip()
    sig = sig.replace("( ", "(").replace(" )", ")").replace(", )", ")")
    return sig


def _callee(func: Any) -> str | None:
    text = _text(func)
    if func.type == "attribute":
        obj = func.child_by_field_name("object")
        attr = func.child_by_field_name("attribute")
        if obj is not None and obj.type == "call" and attr is not None:
            inner = obj.child_by_field_name("function")
            if inner is not None and _text(inner) == "super":
                return f"super.{_text(attr)}"
    text = re.sub(r"\s+", "", text)
    return text if _SIMPLE_CALLEE.match(text) else None


def _collect_calls(node: Any, out: list[str], *, skip_defs: bool) -> None:
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type == "call":
            func = cur.child_by_field_name("function")
            if func is not None:
                name = _callee(func)
                if name:
                    out.append(name)
        for child in cur.children:
            if skip_defs and child.type in (
                "function_definition",
                "class_definition",
                "decorated_definition",
            ):
                continue
            stack.append(child)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


@EXTRACTORS.register("python")
class PythonExtractor(LanguageExtractor):
    name = "python"
    extensions = (".py", ".pyi")

    def module_name(self, path: str) -> str | None:
        parts = path.replace("\\", "/").split("/")
        if not parts[-1].endswith((".py", ".pyi")):
            return None
        parts[-1] = parts[-1].rsplit(".", 1)[0]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        while len(parts) > 1 and parts[0] in _SOURCE_ROOTS:
            parts = parts[1:]
        return ".".join(parts) if parts else None

    def parse(self, source: bytes, path: str) -> ParsedFile:
        tree = _parser().parse(source)
        root = tree.root_node
        lines = source.decode("utf-8", errors="replace").splitlines()
        parsed = ParsedFile(path=path, language=self.name, num_lines=len(lines))
        parsed.module_docstring = _docstring(root)

        covered: list[tuple[int, int]] = []
        for child in root.named_children:
            if child.type in ("import_statement", "import_from_statement"):
                parsed.imports.extend(self._imports(child))
            elif child.type in ("if_statement", "try_statement"):
                # Guarded imports (try/except ImportError, TYPE_CHECKING) are common.
                for sub in _walk_types(child, ("import_statement", "import_from_statement")):
                    parsed.imports.extend(self._imports(sub))
            definition, decorators, outer = _unwrap(child)
            if definition is None:
                continue
            if definition.type == "function_definition":
                parsed.symbols.append(
                    self._function(definition, decorators, outer, source, lines, parent=None)
                )
                covered.append((_line(outer), _end_line(outer)))
            elif definition.type == "class_definition":
                parsed.symbols.extend(
                    self._class(definition, decorators, outer, source, lines, prefix="")
                )
                covered.append((_line(outer), _end_line(outer)))

        parsed.module_code = _uncovered(len(lines), covered, lines)
        return parsed

    # ----------------------------------------------------------------- helpers
    def _imports(self, node: Any) -> list[ImportRef]:
        line = _line(node)
        text = re.sub(r"\s+", " ", _text(node)).strip()
        out: list[ImportRef] = []
        if node.type == "import_statement":
            for child in node.named_children:
                if child.type == "dotted_name":
                    out.append(ImportRef(module=_text(child), line=line, text=text))
                elif child.type == "aliased_import":
                    name = child.child_by_field_name("name")
                    alias = child.child_by_field_name("alias")
                    if name is not None:
                        out.append(
                            ImportRef(
                                module=_text(name),
                                alias=_text(alias) if alias is not None else None,
                                line=line,
                                text=text,
                            )
                        )
            return out
        # import_from_statement
        module_node = node.child_by_field_name("module_name")
        module, level = "", 0
        if module_node is not None:
            raw = _text(module_node)
            level = len(raw) - len(raw.lstrip("."))
            module = raw.lstrip(".")
        names = node.children_by_field_name("name")
        if not names and any(c.type == "wildcard_import" for c in node.children):
            out.append(ImportRef(module=module, name="*", level=level, line=line, text=text))
        for n in names:
            if n.type == "aliased_import":
                name = n.child_by_field_name("name")
                alias = n.child_by_field_name("alias")
                out.append(
                    ImportRef(
                        module=module,
                        name=_text(name) if name is not None else None,
                        alias=_text(alias) if alias is not None else None,
                        level=level,
                        line=line,
                        text=text,
                    )
                )
            else:
                out.append(
                    ImportRef(module=module, name=_text(n), level=level, line=line, text=text)
                )
        return out

    def _function(
        self,
        definition: Any,
        decorators: list[str],
        outer: Any,
        source: bytes,
        lines: list[str],
        *,
        parent: str | None,
    ) -> Symbol:
        name_node = definition.child_by_field_name("name")
        name = _text(name_node) if name_node is not None else "<anonymous>"
        start, end = _line(outer), _end_line(outer)
        calls: list[str] = []
        _collect_calls(definition, calls, skip_defs=False)
        return Symbol(
            qualname=f"{parent}.{name}" if parent else name,
            name=name,
            kind="method" if parent else "function",
            start_line=start,
            end_line=end,
            signature=_signature(definition, source),
            docstring=_docstring(definition.child_by_field_name("body")),
            decorators=decorators,
            parent=parent,
            text="\n".join(lines[start - 1 : end]),
            calls=_dedupe(calls),
        )

    def _class(
        self,
        definition: Any,
        decorators: list[str],
        outer: Any,
        source: bytes,
        lines: list[str],
        *,
        prefix: str,
    ) -> list[Symbol]:
        name_node = definition.child_by_field_name("name")
        name = _text(name_node) if name_node is not None else "<anonymous>"
        qualname = f"{prefix}{name}"
        body = definition.child_by_field_name("body")
        bases: list[str] = []
        supers = definition.child_by_field_name("superclasses")
        if supers is not None:
            for arg in supers.named_children:
                if arg.type in ("identifier", "attribute"):
                    bases.append(re.sub(r"\s+", "", _text(arg)))

        start, end = _line(outer), _end_line(outer)
        members: list[Symbol] = []
        header_lines: list[int] = list(range(start, (_line(body) if body is not None else end)))
        calls: list[str] = []
        first_member_line: int | None = None
        if body is not None:
            for child in body.named_children:
                inner, decs, inner_outer = _unwrap(child)
                if inner is not None and inner.type == "function_definition":
                    members.append(
                        self._function(inner, decs, inner_outer, source, lines, parent=qualname)
                    )
                    first_member_line = first_member_line or _line(inner_outer)
                elif inner is not None and inner.type == "class_definition":
                    members.extend(
                        self._class(inner, decs, inner_outer, source, lines, prefix=f"{qualname}.")
                    )
                    first_member_line = first_member_line or _line(inner_outer)
                else:
                    header_lines.extend(range(_line(child), _end_line(child) + 1))
                    _collect_calls(child, calls, skip_defs=True)
        header_lines = sorted(set(header_lines))
        header_text = "\n".join(lines[i - 1] for i in header_lines if 0 < i <= len(lines))
        # The header leaf's span runs up to the first method (or the whole class).
        header_end = first_member_line - 1 if first_member_line is not None else end
        while header_end > start and not lines[header_end - 1].strip():
            header_end -= 1
        cls = Symbol(
            qualname=qualname,
            name=name,
            kind="class",
            start_line=start,
            end_line=end,
            signature=_signature(definition, source),
            docstring=_docstring(body),
            decorators=decorators,
            parent=prefix.rstrip(".") or None,
            text=header_text,
            calls=_dedupe(calls),
            bases=bases,
            header_end_line=header_end,
        )
        return [cls, *members]


def _walk_types(node: Any, types: tuple[str, ...]) -> list[Any]:
    out = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type in types:
            out.append(cur)
            continue
        if cur.type in ("function_definition", "class_definition"):
            continue
        stack.extend(cur.children)
    return out


def _uncovered(
    num_lines: int, covered: list[tuple[int, int]], lines: list[str]
) -> list[tuple[int, int]]:
    """Contiguous non-blank line ranges outside ``covered``."""
    mask = [False] * (num_lines + 2)
    for a, b in covered:
        for i in range(a, min(b, num_lines) + 1):
            mask[i] = True
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(1, num_lines + 1):
        free = not mask[i]
        if free and start is None:
            start = i
        if not free and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, num_lines))
    out = []
    for a, b in ranges:
        while a <= b and not lines[a - 1].strip():
            a += 1
        while b >= a and not lines[b - 1].strip():
            b -= 1
        if a <= b:
            out.append((a, b))
    return out
