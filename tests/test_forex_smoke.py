import datetime
import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure package import paths
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent_core.data_manager import DataManager
from strategies.vectorized_obr_exact import run_fast_backtest_exact
from agent_core.utils.metrics import compute_global_metrics


def test_forex_smoke():
    dm = DataManager()
    if not dm.connect_ib():
        pytest.skip("IB connection not available")

    today = datetime.date.today()
    start = today - datetime.timedelta(days=1)
    df = dm.get_main_data(
        symbol="EURUSD",
        timeframe="1 min",
        sec_type="FOREX",
        exchange="IDEALPRO",
        currency="USD",
        rth=False,
        what_to_show="MIDPOINT",
        download_start_date=start,
        download_end_date=today,
        use_cache=False,
    )
    dm.disconnect_ib()

    if df.empty:
        pytest.skip("No data for EURUSD")

    df = df.tail(300)
    trades, equity = run_fast_backtest_exact(
        df_day_with_context=df,
        day_start_index=0,
        day_levels={},
    )

    trade_list = [{"pnl": t[6]} for t in trades.tolist()]
    equity_series = pd.Series([e[1] for e in equity.tolist()])
    metrics = compute_global_metrics(equity_series, trade_list, 1000.0)
    result = {"symbol": "EURUSD", "market": "forex", "metrics": metrics}

    assert result["symbol"] == "EURUSD"
    assert result["market"] == "forex"
    assert isinstance(result["metrics"], dict)
