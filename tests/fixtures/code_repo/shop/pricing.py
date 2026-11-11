"""Pricing rules and discount codes."""

from __future__ import annotations

DISCOUNT_CODES = {"WELCOME10": 10, "VIP25": 25}


class PriceRule:
    """Base class for pricing rules."""

    def apply(self, amount: float) -> float:
        """Return the price after this rule; the base rule changes nothing."""
        return amount


class PercentRule(PriceRule):
    """Takes a fixed percentage off the amount."""

    def __init__(self, percent: float) -> None:
        if not 0 <= percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        self.percent = percent

    def apply(self, amount: float) -> float:
        """Reduce ``amount`` by ``percent`` percent."""
        return amount * (100 - self.percent) / 100


def apply_discount(amount: float, code: str) -> float:
    """Apply the discount registered for ``code``; unknown codes raise KeyError."""
    percent = DISCOUNT_CODES[code.upper()]
    return PercentRule(percent).apply(amount)


def tax_for(region: str, amount: float) -> float:
    """Value-added tax owed on ``amount`` in ``region`` (default 20 percent)."""
    rates = {"de": 0.19, "fr": 0.2, "uk": 0.2}
    return amount * rates.get(region.lower(), 0.2)
