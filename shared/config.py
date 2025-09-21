"""Shared configuration flags for reproducibility across execution engines."""
from __future__ import annotations

EXACT_MATCH_MODE: bool = True
TRADING_WINDOW_START: str = "09:30"
TRADING_WINDOW_END: str = "11:30"
FORCE_CLOSE_TIME: str = "13:00"
SLIPPAGE_MODE: str = "points"  # 'points' | 'percent' | 'none'
SLIPPAGE_VALUE: float = 0.0      # 0 para equivalencia exacta por defecto
ALLOW_FRACTIONAL_SIZE_CRYPTO: bool = True

__all__ = [
    "EXACT_MATCH_MODE",
    "TRADING_WINDOW_START",
    "TRADING_WINDOW_END",
    "FORCE_CLOSE_TIME",
    "SLIPPAGE_MODE",
    "SLIPPAGE_VALUE",
    "ALLOW_FRACTIONAL_SIZE_CRYPTO",
]
