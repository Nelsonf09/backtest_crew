"""Utility functions for centralized performance metrics.

This module provides a single source of truth for drawdown and global
performance metrics.  All values returned are raw ``float`` numbers so that
calling code can handle any desired formatting (percentages, currency
symbols, rounding, etc.) in the UI layer.
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd


def compute_drawdown_series_pct(equity: pd.Series) -> pd.Series:
    """Return the drawdown curve in percentage values (``<= 0``).

    Parameters
    ----------
    equity:
        Series of equity values expressed as floats.

    Returns
    -------
    pd.Series
        Drawdown series as percentages where ``0`` means no drawdown and
        negative numbers represent drawdowns.
    """

    equity = equity.astype(float)
    peak = equity.cummax()
    return ((equity / peak) - 1.0) * 100.0  # values <= 0


def compute_drawdown_pct(equity: pd.Series) -> float:
    """Return the maximum drawdown as a positive percentage.

    The value is derived from :func:`compute_drawdown_series_pct`.
    """

    dd_pct_series = compute_drawdown_series_pct(equity)
    return float(-dd_pct_series.min()) if len(dd_pct_series) else 0.0


def compute_global_metrics(
    equity: pd.Series, trades: List[Dict], initial_capital: float
) -> dict:
    """Compute global performance metrics from equity and trade data.

    Parameters
    ----------
    equity:
        Equity curve as a :class:`pandas.Series`.
    trades:
        List of trade dictionaries that contain a ``pnl`` value.
    initial_capital:
        Starting capital for the backtest.

    Returns
    -------
    dict
        Dictionary of raw (unformatted) metrics.
    """

    dd_pct = compute_drawdown_pct(equity)

    net = float(equity.iloc[-1] - initial_capital) if len(equity) else 0.0
    net_pct = (
        float((equity.iloc[-1] / initial_capital - 1.0) * 100.0)
        if len(equity)
        else 0.0
    )

    total = len(trades)
    wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
    losses = sum(1 for t in trades if t.get("pnl", 0.0) < 0)
    win_rate = (wins / total * 100.0) if total else 0.0

    gp = sum(t.get("pnl", 0.0) for t in trades if t.get("pnl", 0.0) > 0)
    gl = -sum(t.get("pnl", 0.0) for t in trades if t.get("pnl", 0.0) < 0)
    profit_factor = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

    return {
        "Ganancia Neta Total ($)": net,
        "Ganancia Neta Total (%)": net_pct,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Max Drawdown (%)": dd_pct,
        "Trades Totales": total,
    }


__all__ = [
    "compute_drawdown_series_pct",
    "compute_drawdown_pct",
    "compute_global_metrics",
]

