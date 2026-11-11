"""Domain objects: products and the shopping cart."""

from dataclasses import dataclass, field


@dataclass
class Product:
    """A product for sale, priced in cents to avoid float rounding."""

    sku: str
    name: str
    price_cents: int
    category: str = "general"


@dataclass
class LineItem:
    product: Product
    quantity: int = 1

    def subtotal(self) -> int:
        """Price of this line before discounts."""
        return self.product.price_cents * self.quantity


@dataclass
class Cart:
    """Items a customer intends to buy."""

    items: list[LineItem] = field(default_factory=list)
    coupon: str | None = None

    def add(self, product: Product, quantity: int = 1) -> None:
        """Add a product, merging with an existing line for the same SKU."""
        for item in self.items:
            if item.product.sku == product.sku:
                item.quantity += quantity
                return
        self.items.append(LineItem(product, quantity))

    def remove(self, sku: str) -> None:
        self.items = [i for i in self.items if i.product.sku != sku]

    def subtotal(self) -> int:
        """Sum of all line subtotals, in cents."""
        return sum(item.subtotal() for item in self.items)
