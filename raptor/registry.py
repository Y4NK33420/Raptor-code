"""A tiny name → factory registry used for pluggable strategies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = factory
            return factory

        return deco

    def create(self, name: str, **kwargs: Any) -> T:
        try:
            factory = self._items[name]
        except KeyError:
            raise ValueError(
                f"unknown {self.kind} '{name}'. Available: {sorted(self._items)}"
            ) from None
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items
