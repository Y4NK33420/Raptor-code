"""Shopping cart model."""

from __future__ import annotations

from .pricing import PercentRule, apply_discount


class Cart:
    """A shopping cart that holds line items and computes totals."""

    currency = "EUR"

    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.items: dict[str, tuple[int, float]] = {}

    def add_item(self, sku: str, quantity: int, unit_price: float) -> None:
        """Add ``quantity`` units of ``sku`` to the cart."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        old_qty, _ = self.items.get(sku, (0, unit_price))
        self.items[sku] = (old_qty + quantity, unit_price)

    def remove_item(self, sku: str) -> None:
        """Remove every unit of ``sku`` from the cart."""
        self.items.pop(sku, None)

    def subtotal(self) -> float:
        """Sum of quantity times unit price over all items."""
        return sum(qty * price for qty, price in self.items.values())

    def total(self, discount_code: str | None = None) -> float:
        """Subtotal after applying the discount code, if any."""
        amount = self.subtotal()
        if discount_code:
            amount = apply_discount(amount, discount_code)
        return round(amount, 2)

    def loyalty_total(self) -> float:
        """Total for loyalty members: a flat ten percent off."""
        rule = PercentRule(10)
        return rule.apply(self.subtotal())
