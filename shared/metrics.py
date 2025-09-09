from __future__ import annotations
import numpy as np
import pandas as pd


def drawdown_stats(equity: pd.Series) -> dict:
    """
    Calcula serie de drawdown y Max Drawdown (%) con convención única.
    dd_t = equity_t/peak_t - 1  (serie <= 0).
    max_drawdown_pct = (-dd.min())*100  (positivo).
    Devuelve dd_series_pct (para graficar), max_drawdown_pct (float),
    y los índices del tramo pico->valle del MDD.
    """
    if not isinstance(equity, pd.Series):
        equity = pd.Series(equity, dtype='float64')
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    mdd_idx = dd.idxmin()
    # localizar pico previo al valle mdd_idx
    peak_until = equity.loc[:mdd_idx].cummax()
    start_idx = (equity.loc[:mdd_idx] / peak_until - 1.0).idxmax()  # primer 0 tras el pico
    result = {
        'dd_series_pct': (dd * 100.0).astype('float64'),
        'max_drawdown_pct': float(-dd.min() * 100.0),
        'mdd_start_idx': start_idx,
        'mdd_end_idx': mdd_idx
    }
    return result
