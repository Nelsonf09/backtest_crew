"""Utility functions for computing global performance metrics."""
from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd


def compute_global_metrics(equity: pd.Series, trades: List[Dict], initial_capital: float) -> dict:
    """Return performance metrics from an equity curve and trade list.

    Parameters
    ----------
    equity: pd.Series
        Series of cumulative equity values in USD.
    trades: List[Dict]
        Each trade dict is expected to contain a ``pnl`` or ``pnl_net`` key.
    initial_capital: float
        Starting capital for the backtest.
    """

    equity = equity.dropna()
    peak = equity.cummax()
    dd_series = (equity / peak) - 1.0
    max_dd_pct = float(-dd_series.min() * 100.0) if len(dd_series) else 0.0

    net_profit = float(equity.iloc[-1] - initial_capital) if len(equity) else 0.0
    net_profit_pct = (
        float((equity.iloc[-1] / initial_capital - 1.0) * 100.0)
        if len(equity) and initial_capital
        else 0.0
    )

    pnls = [t.get("pnl", t.get("pnl_net", 0.0)) for t in trades]
    total_trades = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    win_rate = float((wins / total_trades) * 100.0) if total_trades else 0.0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    return {
        "Ganancia Neta Total ($)": round(net_profit, 2),
        "Ganancia Neta Total (%)": round(net_profit_pct, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf"),
        "Max Drawdown (%)": round(max_dd_pct, 2),
        "Trades Totales": total_trades,
    }
