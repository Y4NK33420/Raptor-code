"""Command-line entry point for the shop fixture."""

import sys

from shop.cart import Cart
from shop.storage.db import CartStore


def main(argv: list[str]) -> int:
    """Create a cart for ``argv[0]``, add one item, save it and print the total."""
    store = CartStore(":memory:")
    cart = Cart(argv[0] if argv else "guest")
    cart.add_item("apple", 3, 0.5)
    store.save_cart(cart)
    print(f"{cart.owner}: {cart.total()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
