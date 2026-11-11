"""Checkout: price the cart, validate payment, and record the order."""

import uuid

from .models import Cart
from .pricing import quote
from .storage import save_order


class PaymentError(Exception):
    """Raised when the payment amount does not cover the order total."""


def validate_payment(amount_cents: int, total_cents: int) -> None:
    if amount_cents < total_cents:
        raise PaymentError(f"payment {amount_cents} is less than total {total_cents}")


def checkout(cart: Cart, paid_cents: int) -> dict:
    """Price the cart, check the payment covers it, save and return the order."""
    if not cart.items:
        raise ValueError("cannot check out an empty cart")
    prices = quote(cart)
    validate_payment(paid_cents, prices["total"])
    order = {
        "id": uuid.uuid4().hex,
        "lines": [(i.product.sku, i.quantity) for i in cart.items],
        **prices,
        "change": paid_cents - prices["total"],
    }
    save_order(order)
    return order
