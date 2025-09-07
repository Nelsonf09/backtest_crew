import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent_core.metrics import compute_max_drawdown_pct


def test_compute_max_drawdown_pct_basic():
    equity = pd.Series([100, 120, 90, 95, 80, 110])
    mdd = compute_max_drawdown_pct(equity)
    assert mdd == pytest.approx(33.33, abs=0.01)

