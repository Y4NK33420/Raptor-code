"""Best-effort static code graph: imports, calls and inheritance.

Edges are computed between *symbol keys* ``(path, qualname)``; a ``qualname`` of ``None``
means the file itself. :class:`raptor.code.index.CodeIndex` maps the keys to tree nodes.

Resolution is purely syntactic and deliberately conservative: a call is linked only
when it resolves through ``self``/``super()``, an import binding, the same file, or a
repo-wide *unique* name. Dynamic dispatch and duck typing are not followed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .languages.base import LanguageExtractor, ParsedFile, Symbol

SymKey = tuple[str, str | None]

#: Method names too generic to resolve by repo-wide uniqueness.
_GENERIC_METHODS = frozenset(
    {
        "get",
        "set",
        "update",
        "append",
        "extend",
        "pop",
        "items",
        "keys",
        "values",
        "copy",
        "close",
        "open",
        "read",
        "write",
        "format",
        "join",
        "split",
        "strip",
        "encode",
        "decode",
        "add",
        "remove",
        "run",
        "start",
        "send",
        "lower",
        "upper",
        "replace",
        "startswith",
        "endswith",
    }
)


@dataclass(frozen=True)
class SymEdge:
    src: SymKey
    dst: SymKey
    kind: str  # "imports" | "calls" | "inherits"


@dataclass
class _Binding:
    path: str | None  # target file (None: external)
    qualname: str | None  # target symbol in that file (None: the module itself)
    module: str


class CodeGraphBuilder:
    def __init__(
        self, parsed: dict[str, ParsedFile], extractors: dict[str, LanguageExtractor]
    ) -> None:
        self.parsed = parsed
        self.module_to_path: dict[str, str] = {}
        self.path_to_module: dict[str, str] = {}
        for path in parsed:
            ex = next((e for ext, e in extractors.items() if path.lower().endswith(ext)), None)
            mod = ex.module_name(path) if ex else None
            if mod:
                self.module_to_path.setdefault(mod, path)
                self.path_to_module[path] = mod
        self.symbols: dict[str, dict[str, Symbol]] = {
            path: {s.qualname: s for s in pf.symbols} for path, pf in parsed.items()
        }
        self.by_name: dict[str, list[tuple[str, Symbol]]] = defaultdict(list)
        for path, syms in self.symbols.items():
            for s in syms.values():
                self.by_name[s.name].append((path, s))
        self._bindings: dict[str, dict[str, _Binding]] = {}

    # ------------------------------------------------------------- modules
    def _resolve_module(self, name: str) -> str | None:
        if name in self.module_to_path:
            return self.module_to_path[name]
        hits = [p for m, p in self.module_to_path.items() if m.endswith("." + name)]
        return hits[0] if len(hits) == 1 else None

    def _absolute(self, path: str, module: str, level: int) -> str:
        if level == 0:
            return module
        own = self.path_to_module.get(path, "")
        is_pkg = path.endswith(("__init__.py", "__init__.pyi"))
        parts = own.split(".") if own else []
        if not is_pkg:
            parts = parts[:-1]
        if level > 1:
            parts = parts[: max(0, len(parts) - (level - 1))]
        base = ".".join(parts)
        return f"{base}.{module}" if base and module else (base or module)

    def bindings(self, path: str, _depth: int = 0) -> dict[str, _Binding]:
        if path in self._bindings:
            return self._bindings[path]
        out: dict[str, _Binding] = {}
        self._bindings[path] = out  # guards against import cycles
        for imp in self.parsed[path].imports:
            mod = self._absolute(path, imp.module, imp.level)
            if imp.name is None:  # import a.b [as x]
                if imp.alias:
                    out[imp.alias] = _Binding(self._resolve_module(mod), None, mod)
                else:
                    top = mod.split(".")[0]
                    out[top] = _Binding(self._resolve_module(top), None, top)
                continue
            if imp.name == "*":
                target = self._resolve_module(mod)
                if target and _depth < 3:
                    for qual, s in self.symbols.get(target, {}).items():
                        if s.parent is None:
                            out.setdefault(qual, _Binding(target, qual, mod))
                continue
            bound = imp.alias or imp.name
            sub = self._resolve_module(f"{mod}.{imp.name}") if mod else None
            if sub is not None:
                out[bound] = _Binding(sub, None, f"{mod}.{imp.name}")
                continue
            target = self._resolve_module(mod) if mod else None
            out[bound] = self._follow(target, imp.name, mod, _depth)
        return out

    def _follow(self, target: str | None, name: str, mod: str, depth: int) -> _Binding:
        """Resolve ``name`` in file ``target``, following re-exports a few hops."""
        if target is None:
            return _Binding(None, None, mod)
        if name in self.symbols.get(target, {}):
            return _Binding(target, name, mod)
        if depth < 3:
            b = self.bindings(target, depth + 1).get(name)
            if b is not None and b.path is not None:
                return b
        return _Binding(target, None, mod)

    # -------------------------------------------------------------- symbols
    def _class_of(self, path: str, sym: Symbol) -> Symbol | None:
        if sym.kind == "class":
            return sym
        if sym.parent:
            return self.symbols[path].get(sym.parent)
        return None

    def _resolve_class_ref(self, path: str, ref: str) -> SymKey | None:
        """Resolve a base-class / class reference like ``Base`` or ``models.Base``."""
        key = self._resolve_name(path, ref)
        if key is None:
            return None
        p, q = key
        if q is not None and p is not None and self.symbols[p][q].kind == "class":
            return key
        return None

    def _method_in(
        self, path: str, cls: Symbol, name: str, seen: set | None = None
    ) -> SymKey | None:
        seen = seen or set()
        if (path, cls.qualname) in seen:
            return None
        seen.add((path, cls.qualname))
        qual = f"{cls.qualname}.{name}"
        if qual in self.symbols[path]:
            return (path, qual)
        for base in cls.bases:
            key = self._resolve_class_ref(path, base)
            if key is None or key[1] is None:
                continue
            found = self._method_in(key[0], self.symbols[key[0]][key[1]], name, seen)
            if found:
                return found
        return None

    def _resolve_name(self, path: str, ref: str) -> SymKey | None:
        parts = ref.split(".")
        head, rest = parts[0], parts[1:]
        syms = self.symbols[path]
        # Same-file top-level symbol.
        if head in syms and syms[head].parent is None:
            return self._descend(path, head, rest)
        b = self.bindings(path).get(head)
        if b is not None and b.path is not None:
            if b.qualname is not None:
                return self._descend(b.path, b.qualname, rest)
            return self._in_module(b.module, b.path, rest)
        if b is not None or rest:
            return None
        # Repo-wide unique top-level name.
        cands = [(p, s) for p, s in self.by_name.get(head, []) if s.parent is None]
        if len(cands) == 1:
            return (cands[0][0], cands[0][1].qualname)
        return None

    def _in_module(self, module: str, path: str, rest: list[str]) -> SymKey | None:
        if not rest:
            return (path, None)
        sub = self._resolve_module(f"{module}.{rest[0]}")
        if sub is not None:
            return self._in_module(f"{module}.{rest[0]}", sub, rest[1:])
        if rest[0] in self.symbols.get(path, {}):
            return self._descend(path, rest[0], rest[1:])
        b = self.bindings(path).get(rest[0])
        if b is not None and b.path is not None and b.qualname is not None:
            return self._descend(b.path, b.qualname, rest[1:])
        return None

    def _descend(self, path: str, qual: str, rest: list[str]) -> SymKey | None:
        if not rest:
            return (path, qual)
        sym = self.symbols[path].get(qual)
        if sym is None or sym.kind != "class":
            return None
        found = self._method_in(path, sym, rest[0])
        if found and len(rest) == 1:
            return found
        return None

    def resolve_call(self, path: str, sym: Symbol, callee: str) -> SymKey | None:
        parts = callee.split(".")
        cls = self._class_of(path, sym)
        if parts[0] in ("self", "cls") and len(parts) == 2 and cls is not None:
            return self._method_in(path, cls, parts[1])
        if parts[0] == "super" and len(parts) == 2 and cls is not None:
            for base in cls.bases:
                key = self._resolve_class_ref(path, base)
                if key and key[1]:
                    found = self._method_in(key[0], self.symbols[key[0]][key[1]], parts[1])
                    if found:
                        return found
            return None
        key = self._resolve_name(path, callee)
        if key is not None:
            return key
        # obj.method(): link only when the method name is unique in the repo.
        if len(parts) == 2 and parts[0] not in ("self", "cls", "super"):
            name = parts[1]
            if name in _GENERIC_METHODS or name.startswith("__"):
                return None
            cands = [(p, s) for p, s in self.by_name.get(name, []) if s.kind == "method"]
            if len(cands) == 1:
                return (cands[0][0], cands[0][1].qualname)
        return None

    # ---------------------------------------------------------------- edges
    def edges(self) -> list[SymEdge]:
        out: set[SymEdge] = set()
        for path, pf in self.parsed.items():
            for b in self.bindings(path).values():
                if b.path is not None and b.path != path:
                    out.add(SymEdge((path, None), (b.path, None), "imports"))
            for sym in pf.symbols:
                src: SymKey = (path, sym.qualname)
                for callee in sym.calls:
                    dst = self.resolve_call(path, sym, callee)
                    if dst is not None and dst != src and dst[1] is not None:
                        out.add(SymEdge(src, dst, "calls"))
                if sym.kind == "class":
                    for base in sym.bases:
                        dst2 = self._resolve_class_ref(path, base)
                        if dst2 is not None and dst2 != src:
                            out.add(SymEdge(src, dst2, "inherits"))
        return sorted(
            out, key=lambda e: (e.kind, e.src[0], e.src[1] or "", e.dst[0], e.dst[1] or "")
        )
