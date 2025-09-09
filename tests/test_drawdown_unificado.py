import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from shared.metrics import drawdown_stats


def test_mdd_coincide_con_curva_y_signo():
    eq = pd.Series([1000, 1100, 1050, 1300, 1180, 1500, 1350], dtype='float64')
    stats = drawdown_stats(eq)
    assert stats['max_drawdown_pct'] > 0
    assert abs(abs(stats['dd_series_pct'].min()) - stats['max_drawdown_pct']) < 1e-6


def test_mdd_casos_basicos():
    eq = pd.Series([100, 90, 80, 120, 100], dtype='float64')  # MDD 20%
    stats = drawdown_stats(eq)
    assert abs(stats['max_drawdown_pct'] - 20.0) < 1e-6
