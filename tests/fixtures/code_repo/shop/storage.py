"""Persist orders as JSON lines."""

import json
from pathlib import Path

ORDERS = Path("orders.jsonl")


def save_order(order: dict, path: Path = ORDERS) -> None:
    """Append one order to the orders file."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(order) + "\n")


def load_orders(path: Path = ORDERS) -> list[dict]:
    """Read every saved order (an empty list if there are none yet)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
