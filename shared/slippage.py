"""Slippage helpers shared by the fast and step-by-step engines."""
from __future__ import annotations

from typing import Literal

Side = Literal["buy", "sell"]
Mode = Literal["points", "percent", "none"]


def apply_slippage(price: float, side: Side, mode: Mode, value: float) -> float:
    """Apply slippage to price according to side and mode.

    - buy: increases the price by points/percent
    - sell: decreases the price by points/percent (clamped to >=0)
    - none: returns the price unchanged
    """
    if price is None:
        raise ValueError("Price must be provided to compute slippage")

    mode_normalized = (mode or "none").lower()
    side_normalized = (side or "").lower()
    if mode_normalized == "none" or float(value) == 0.0:
        return float(price)
    if side_normalized not in {"buy", "sell"}:
        raise ValueError(f"Unsupported side '{side}'. Expected 'buy' or 'sell'.")

    if mode_normalized == "points":
        adj = abs(float(value))
    elif mode_normalized == "percent":
        adj = abs(float(price) * float(value))
    else:
        raise ValueError(f"Unsupported slippage mode '{mode}'.")

    if side_normalized == "buy":
        return float(price) + adj
    out = float(price) - adj
    return out if out > 0 else 0.0


__all__ = ["apply_slippage"]
