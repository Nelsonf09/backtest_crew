"""Utility functions for performance metrics.

This module centralizes drawdown calculations so that every part of the
application uses the exact same logic.  All functions expect equity values in
US dollars (not returns).
"""

from __future__ import annotations

import logging
from decimal import Decimal

import pandas as pd


logger = logging.getLogger(__name__)


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Return the drawdown series (values ``<= 0``) from an equity curve.

    Parameters
    ----------
    equity: pd.Series
        Equity in **USD** indexed by datetime. ``NaN`` values are allowed and
        will be filled with ``0`` in the result.

    Returns
    -------
    pd.Series
        Drawdown values expressed as decimal fractions (e.g. ``-0.25`` for
        ``-25%``).
    """

    if equity is None or equity.empty:
        return pd.Series(dtype=float)

    peak = equity.cummax()
    dd = equity.div(peak).sub(1.0)
    # Ensure values are <= 0 and replace any remaining NaNs with 0.0
    return dd.fillna(0.0).clip(upper=0.0)


def compute_max_drawdown_pct(equity: pd.Series) -> float:
    """Return the maximum drawdown as a positive percentage.

    Parameters
    ----------
    equity: pd.Series
        Equity in **USD** indexed by datetime.

    Returns
    -------
    float
        Maximum drawdown as a positive percentage (e.g. ``30.73``).
    """

    dd = compute_drawdown_series(equity)
    if dd.empty:
        return 0.0
    return abs(dd.min()) * 100.0


def calculate_performance_metrics(
    trades: list, initial_capital: float, equity_history: list | None = None
) -> dict:
    """Calculate performance metrics for a backtest.

    Notes
    -----
    ``equity_history`` must contain equity values in USD.  ``Max Drawdown`` is
    returned as a positive percentage.
    """

    metrics = {
        "Total Trades": 0,
        "Ganancia Neta Total ($)": 0.0,
        "Ganancia Neta Total (%)": 0.0,
        "Trades Ganadores": 0,
        "Trades Perdedores": 0,
        "Win Rate (%)": 0.0,
        "Profit Factor": "N/A",
        "Ganancia Promedio ($)": 0.0,
        "Pérdida Promedio ($)": 0.0,
        "Ratio Ganancia/Pérdida Prom.": "N/A",
        "Max Drawdown (%)": 0.0,
    }

    if not trades and (equity_history is None or len(equity_history) <= 1):
        logger.warning(
            "No hay trades cerrados ni historial de equity para calcular métricas."
        )
        return metrics

    if trades:
        try:
            trades_df = pd.DataFrame(trades)
            trades_df["pnl_net"] = pd.to_numeric(trades_df["pnl_net"])
            metrics["Total Trades"] = len(trades_df)
            total_net_pnl = trades_df["pnl_net"].sum()
            metrics["Ganancia Neta Total ($)"] = round(total_net_pnl, 2)
            metrics["Ganancia Neta Total (%)"] = (
                round((total_net_pnl / initial_capital) * 100, 2)
                if initial_capital
                else 0.0
            )
            winning_trades = trades_df[trades_df["pnl_net"] > 0]
            losing_trades = trades_df[trades_df["pnl_net"] <= 0]
            metrics["Trades Ganadores"] = len(winning_trades)
            metrics["Trades Perdedores"] = len(losing_trades)
            if metrics["Total Trades"] > 0:
                metrics["Win Rate (%)"] = round(
                    (metrics["Trades Ganadores"] / metrics["Total Trades"]) * 100, 2
                )
            gross_profit = winning_trades["pnl_net"].sum()
            gross_loss = abs(losing_trades["pnl_net"].sum())
            if metrics["Trades Ganadores"] > 0:
                metrics["Ganancia Promedio ($)"] = round(
                    gross_profit / metrics["Trades Ganadores"], 2
                )
            if metrics["Trades Perdedores"] > 0:
                metrics["Pérdida Promedio ($)"] = round(
                    gross_loss / metrics["Trades Perdedores"], 2
                )

            if gross_loss > 0:
                metrics["Profit Factor"] = round(gross_profit / gross_loss, 2)
            else:
                metrics["Profit Factor"] = "Inf" if gross_profit > 0 else "N/A"

            if metrics["Pérdida Promedio ($)"] > 0:
                metrics["Ratio Ganancia/Pérdida Prom."] = round(
                    metrics["Ganancia Promedio ($)"]
                    / metrics["Pérdida Promedio ($)"],
                    2,
                )
            else:
                metrics["Ratio Ganancia/Pérdida Prom."] = (
                    "Inf" if metrics["Ganancia Promedio ($)"] > 0 else "N/A"
                )
        except Exception as e_metrics:  # pragma: no cover - logging
            logger.error(
                f"Error calculando métricas de trades: {e_metrics}", exc_info=True
            )

    if equity_history and len(equity_history) > 1:
        try:
            equity_df = pd.DataFrame(equity_history, columns=["time", "equity"])
            equity_df["equity"] = pd.to_numeric(equity_df["equity"])
            metrics["Max Drawdown (%)"] = compute_max_drawdown_pct(
                equity_df["equity"]
            )
        except Exception as e_dd:  # pragma: no cover - logging
            logger.error(
                f"Error calculando Max Drawdown: {e_dd}", exc_info=True
            )
            metrics["Max Drawdown (%)"] = 0.0

    return metrics


if __name__ == "__main__":  # pragma: no cover
    # Pequeña prueba manual del cálculo de Max Drawdown
    equity_example = pd.Series([100, 120, 80, 90, 70, 150])
    print("Equity:", equity_example.tolist())
    dd = compute_drawdown_series(equity_example) * 100.0
    print("Drawdown curve (%):", dd.tolist())
    print("Max Drawdown (%):", compute_max_drawdown_pct(equity_example))

