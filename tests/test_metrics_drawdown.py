import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from shared.metrics import drawdown_stats

def test_mdd_simple():
    eq = pd.Series([100, 120, 115, 130, 110, 140], dtype='float64')
    res = drawdown_stats(eq)
    assert abs(res['max_drawdown_pct'] - 15.38461538) < 1e-6

def test_mdd_extremo():
    eq = pd.Series([100, 90, 80, 120, 100], dtype='float64')
    res = drawdown_stats(eq)
    assert abs(res['max_drawdown_pct'] - 20.0) < 1e-6

def test_curve_consistency():
    eq = pd.Series([100, 120, 115, 130, 110, 140], dtype='float64')
    res = drawdown_stats(eq)
    assert abs(abs(res['dd_series_pct'].min()) - res['max_drawdown_pct']) < 1e-6
