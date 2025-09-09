from __future__ import annotations
import pandas as pd


def drawdown_stats(equity: pd.Series) -> dict:
    """Convención única:
    peak_t = equity.cummax()
    dd_t   = equity/peak_t - 1.0      # serie <= 0
    MDD%   = (-dd_t.min())*100.0      # número positivo
    Devuelve: dd_series_pct (p/ gráfica), max_drawdown_pct (float, +),
    y los índices de inicio/fin del tramo MDD pico->valle."""
    if not isinstance(equity, pd.Series):
        equity = pd.Series(equity, dtype='float64')
    peak = equity.cummax()
    dd = (equity/peak) - 1.0
    mdd_end = dd.idxmin()
    start_slice = equity.loc[:mdd_end]
    start_peak = start_slice.cummax()
    mdd_start = (start_slice/start_peak - 1.0).idxmax()
    return {
        'dd_series_pct': (dd*100.0).astype('float64'),
        'max_drawdown_pct': float(-dd.min()*100.0),
        'mdd_start_idx': mdd_start,
        'mdd_end_idx': mdd_end
    }
