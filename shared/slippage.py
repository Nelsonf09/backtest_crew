"""Slippage helpers shared by the fast and step-by-step engines."""
from __future__ import annotations

from typing import Literal

Side = Literal["buy", "sell"]
Mode = Literal["points", "percent", "none"]


def apply_slippage(price: float, side: Side, mode: Mode, value: float) -> float:
    """Apply slippage to *price* according to *side* and *mode*.

    A ``buy`` order worsens the fill by increasing the price, while a ``sell`` order
    worsens it by decreasing the price. When ``mode`` is ``'none'`` or ``value`` is
    zero the input price is returned unchanged.
    """
    if price is None:
        raise ValueError("Price must be provided to compute slippage")

    mode_normalized = (mode or "none").lower()
    side_normalized = (side or "").lower()
    if mode_normalized == "none" or value == 0:
        return float(price)
    if side_normalized not in {"buy", "sell"}:
        raise ValueError(f"Unsupported side '{side}'. Expected 'buy' or 'sell'.")

    slippage_amount: float
    if mode_normalized == "points":
        slippage_amount = abs(float(value))
    elif mode_normalized == "percent":
        slippage_amount = abs(float(price) * float(value))
    else:
        raise ValueError(f"Unsupported slippage mode '{mode}'.")

    if side_normalized == "buy":
        return float(price) + slippage_amount
    adjusted = float(price) - slippage_amount
    return adjusted if adjusted > 0 else 0.0


__all__ = ["apply_slippage"]
