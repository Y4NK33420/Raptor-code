"""Pricing rules: coupons, bulk discounts and sales tax."""

from .models import Cart

COUPONS = {"WELCOME10": 0.10, "VIP25": 0.25}
TAX_RATES = {"general": 0.08, "food": 0.02, "books": 0.0}


def coupon_discount(cart: Cart) -> int:
    """Discount in cents from the cart's coupon code, if it is valid."""
    rate = COUPONS.get((cart.coupon or "").upper(), 0.0)
    return round(cart.subtotal() * rate)


def bulk_discount(cart: Cart) -> int:
    """Five percent off any line with ten or more units."""
    return sum(round(i.subtotal() * 0.05) for i in cart.items if i.quantity >= 10)


def sales_tax(cart: Cart, discount: int) -> int:
    """Tax per product category, applied after discounts (spread proportionally)."""
    subtotal = cart.subtotal() or 1
    tax = 0.0
    for item in cart.items:
        share = item.subtotal() - discount * item.subtotal() / subtotal
        tax += share * TAX_RATES.get(item.product.category, TAX_RATES["general"])
    return round(tax)


def quote(cart: Cart) -> dict[str, int]:
    """Full price breakdown for a cart."""
    discount = coupon_discount(cart) + bulk_discount(cart)
    tax = sales_tax(cart, discount)
    total = cart.subtotal() - discount + tax
    return {"subtotal": cart.subtotal(), "discount": discount, "tax": tax, "total": total}
