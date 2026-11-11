"""SQLite persistence for carts."""

from __future__ import annotations

import json
import sqlite3

from ..cart import Cart


def get_connection(path: str) -> sqlite3.Connection:
    """Open the SQLite database at ``path`` and create the carts table."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS carts (owner TEXT PRIMARY KEY, items TEXT)")
    return conn


class CartStore:
    """Saves and loads carts as JSON rows in SQLite."""

    def __init__(self, path: str) -> None:
        self.conn = get_connection(path)

    def save_cart(self, cart: Cart) -> None:
        """Insert or replace the row for ``cart.owner``."""
        payload = json.dumps(cart.items)
        self.conn.execute("INSERT OR REPLACE INTO carts VALUES (?, ?)", (cart.owner, payload))
        self.conn.commit()

    def load_cart(self, owner: str) -> Cart:
        """Load the cart saved for ``owner`` (an empty cart if there is none)."""
        cart = Cart(owner)
        row = self.conn.execute("SELECT items FROM carts WHERE owner = ?", (owner,)).fetchone()
        if row:
            cart.items = {k: tuple(v) for k, v in json.loads(row[0]).items()}
        return cart
