import numpy as np
import pandas as pd


def drawdown_curve_from_equity(equity: pd.Series) -> pd.DataFrame:
    """
    Recibe Serie de equity ($) indexada por datetime.
    Devuelve DataFrame con:
      - equity
      - rolling_peak (máximo acumulado)
      - dd_pct (drawdown %, negativo), y en attrs['max_dd_pct'] el mínimo (%).
    """
    eq = equity.astype(float).copy()
    rolling_peak = eq.cummax()
    dd_pct = (eq / rolling_peak - 1.0) * 100.0
    df = pd.DataFrame({'equity': eq, 'rolling_peak': rolling_peak, 'dd_pct': dd_pct})
    df.attrs['max_dd_pct'] = float(dd_pct.min()) if len(dd_pct) else 0.0
    return df
