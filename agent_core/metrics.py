# core/metrics.py
import pandas as pd
import numpy as np
import logging
from decimal import Decimal

from agent_core.performance import drawdown_curve_from_equity

# Logger específico
logger = logging.getLogger(__name__)


def compute_max_drawdown(equity: pd.Series) -> float:
    """Calcula el Max Drawdown (peak→trough) en porcentaje positivo.

    Parameters
    ----------
    equity : pd.Series
        Serie de valores de equity para una ejecución.

    Returns
    -------
    float
        Máximo drawdown expresado como porcentaje positivo con dos decimales.
        Si la serie es None, vacía o constante, devuelve 0.0.
    """

    if equity is None or equity.empty:
        return 0.0

    # Si todos los valores son iguales no hay drawdown
    if equity.nunique() <= 1:
        return 0.0

    roll_max = equity.cummax()
    # Evitar divisiones por cero reemplazando picos en cero con NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (equity - roll_max) / roll_max.replace(0, np.nan)

    max_dd = dd.min()
    if pd.isna(max_dd):
        return 0.0

    return round(abs(max_dd) * 100, 2)

def calculate_performance_metrics(trades: list, initial_capital: float,
                                  equity_history: list | None = None) -> dict:
    """ Calcula métricas de rendimiento del backtest. """
    metrics = { "Total Trades": 0, "Ganancia Neta Total ($)": 0.0, "Ganancia Neta Total (%)": 0.0,
                "Trades Ganadores": 0, "Trades Perdedores": 0, "Win Rate (%)": 0.0, "Profit Factor": "N/A",
                "Ganancia Promedio ($)": 0.0, "Pérdida Promedio ($)": 0.0, "Ratio Ganancia/Pérdida Prom.": "N/A",
                "Max Drawdown (%)": 0.0 } # <-- Valor por defecto cambiado a 0.0

    if not trades and (equity_history is None or len(equity_history) <= 1):
        logger.warning("No hay trades cerrados ni historial de equity para calcular métricas.")
        return metrics

    if trades:
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
            else: metrics["Profit Factor"] = "Inf" if gross_profit > 0 else "N/A"
            
            if metrics["Pérdida Promedio ($)"] > 0: metrics["Ratio Ganancia/Pérdida Prom."] = round(metrics["Ganancia Promedio ($)"] / metrics["Pérdida Promedio ($)"], 2)
            else: metrics["Ratio Ganancia/Pérdida Prom."] = "Inf" if metrics["Ganancia Promedio ($)"] > 0 else "N/A"
        except Exception as e_metrics:
            logger.error(f"Error calculando métricas de trades: {e_metrics}", exc_info=True)


    # Calcular Max Drawdown
    if equity_history and len(equity_history) > 1:
        try:
            equity_df = pd.DataFrame(equity_history, columns=["time", "equity"])
            equity_df["equity"] = pd.to_numeric(equity_df["equity"])
            dd = drawdown_curve_from_equity(equity_df["equity"])
            metrics["Max Drawdown (%)"] = dd.attrs.get('max_dd_pct', float(dd['dd_pct'].min()))
        except Exception as e_dd:
            logger.error(f"Error calculando Max Drawdown: {e_dd}", exc_info=True)
            metrics["Max Drawdown (%)"] = 0.0  # Fallback a 0.0 en caso de error
             
    return metrics


if __name__ == "__main__":
    # Pequeña prueba manual del cálculo de Max Drawdown
    equity_example = pd.Series([100, 120, 80, 90, 70, 150])
    print("Equity:", equity_example.tolist())
    dd_example = drawdown_curve_from_equity(equity_example)
    print("Max Drawdown (%):", dd_example.attrs.get('max_dd_pct'))
