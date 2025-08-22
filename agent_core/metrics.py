# core/metrics.py
import pandas as pd
import numpy as np
import logging
from decimal import Decimal

# Logger específico
logger = logging.getLogger(__name__)

def calculate_performance_metrics(trades: list, initial_capital: float,
                                  equity_history: list | None = None) -> dict:
    """ Calcula métricas de rendimiento del backtest. """
    metrics = { "Total Trades": 0, "Ganancia Neta Total ($)": 0.0, "Ganancia Neta Total (%)": 0.0,
                "Trades Ganadores": 0, "Trades Perdedores": 0, "Win Rate (%)": 0.0, "Profit Factor": None,
                "Ganancia Promedio ($)": 0.0, "Pérdida Promedio ($)": 0.0, "Ratio Ganancia/Pérdida Prom.": None,
                "Max Drawdown (%)": None }
    if not trades:
        logger.warning("No hay trades cerrados para calcular métricas.") # WARNING
        metrics["Ganancia Neta Total ($)"] = 0.0; metrics["Ganancia Neta Total (%)"] = 0.0
        return metrics
    try:
        trades_df = pd.DataFrame(trades)
        trades_df['pnl_net'] = pd.to_numeric(trades_df['pnl_net'])
        metrics["Total Trades"] = len(trades_df)
        total_net_pnl = trades_df['pnl_net'].sum()
        metrics["Ganancia Neta Total ($)"] = round(total_net_pnl, 2)
        metrics["Ganancia Neta Total (%)"] = round((total_net_pnl / initial_capital) * 100, 2) if initial_capital else 0.0
        winning_trades = trades_df[trades_df['pnl_net'] > 0]; losing_trades = trades_df[trades_df['pnl_net'] <= 0]
        metrics["Trades Ganadores"] = len(winning_trades); metrics["Trades Perdedores"] = len(losing_trades)
        if metrics["Total Trades"] > 0: metrics["Win Rate (%)"] = round((metrics["Trades Ganadores"] / metrics["Total Trades"]) * 100, 2)
        gross_profit = winning_trades['pnl_net'].sum(); gross_loss = abs(losing_trades['pnl_net'].sum())
        if metrics["Trades Ganadores"] > 0: metrics["Ganancia Promedio ($)"] = round(gross_profit / metrics["Trades Ganadores"], 2)
        if metrics["Trades Perdedores"] > 0: metrics["Pérdida Promedio ($)"] = round(gross_loss / metrics["Trades Perdedores"], 2)
        if gross_loss > 0: metrics["Profit Factor"] = round(gross_profit / gross_loss, 2)
        else: metrics["Profit Factor"] = float('inf') if gross_profit > 0 else None
        if metrics["Pérdida Promedio ($)"] > 0: metrics["Ratio Ganancia/Pérdida Prom."] = round(metrics["Ganancia Promedio ($)"] / metrics["Pérdida Promedio ($)"], 2)
        else: metrics["Ratio Ganancia/Pérdida Prom."] = float('inf') if metrics["Ganancia Promedio ($)"] > 0 else None

        # Calcular Max Drawdown
        if equity_history and len(equity_history) > 1:
            try:
                equity_df = pd.DataFrame(equity_history, columns=['time', 'equity'])
                equity_df['equity'] = pd.to_numeric(equity_df['equity'])
                equity_df['time'] = pd.to_datetime(equity_df['time'])
                equity_df = equity_df.set_index('time').sort_index() # Asegurar orden
                equity_df = equity_df[~equity_df.index.duplicated(keep='last')] # Eliminar duplicados de índice
                if not equity_df.empty: # Chequeo adicional
                    equity_df['peak'] = equity_df['equity'].cummax()
                    equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
                    equity_df['drawdown_pct'] = np.where(equity_df['peak'] != 0, (equity_df['drawdown'] / equity_df['peak']) * 100, 0)
                    max_dd_pct = equity_df['drawdown_pct'].min()
                    metrics["Max Drawdown (%)"] = round(abs(max_dd_pct), 2) if pd.notna(max_dd_pct) else 0.0
                else: logger.warning("Historial de equity vacío tras procesar para Drawdown.") # WARNING
            except Exception as e_dd:
                 logger.error(f"Error calculando Max Drawdown: {e_dd}", exc_info=True) # ERROR
                 metrics["Max Drawdown (%)"] = None
    except Exception as e_metrics:
        logger.error(f"Error general calculando métricas: {e_metrics}", exc_info=True) # ERROR
    return metrics