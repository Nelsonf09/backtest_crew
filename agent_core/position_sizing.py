"""Position sizing utilities shared across execution engines."""
from __future__ import annotations

from decimal import Decimal
from math import floor
from typing import Optional


def _quantize_step(value: float, step: Optional[float]) -> float:
    if not step or step <= 0:
        return float(value)
    multiples = floor(float(value) / float(step))
    return float(multiples * float(step))


def compute_position_size(
    market: str,
    equity: float,
    leverage: float,
    entry_price: float,
    lot_step: float | None,
    min_qty: float | None,
    allow_fractional_crypto: bool,
) -> int | float:
    """Compute the desired position size following the fast-engine policy."""
    if equity <= 0 or leverage <= 0 or entry_price <= 0:
        return 0

    notional = float(equity) * float(leverage)
    if notional <= 0:
        return 0

    raw_size = notional / float(entry_price)
    market_lower = (market or "").lower()
    fractional_allowed = market_lower == "crypto" and allow_fractional_crypto

    size = _quantize_step(raw_size, lot_step if fractional_allowed else (lot_step or 1.0))

    if min_qty and min_qty > 0 and size < float(min_qty):
        return 0

    if size <= 0:
        return 0

    if fractional_allowed:
        return float(size)

    quantized = Decimal(str(size))
    if quantized == quantized.to_integral():
        return int(quantized)
    return float(quantized)


__all__ = ["compute_position_size"]
