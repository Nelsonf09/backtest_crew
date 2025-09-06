import datetime
import sys
from pathlib import Path
import pytest

# Ensure package import paths
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent_core.data_manager import DataManager
from strategies.vectorized_obr_exact import run_fast_backtest_exact
from agent_core.metrics import calculate_performance_metrics


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

    metrics = calculate_performance_metrics(trades.tolist(), 1000.0, equity.tolist())
    result = {"symbol": "EURUSD", "market": "forex", "metrics": metrics}

    assert result["symbol"] == "EURUSD"
    assert result["market"] == "forex"
    assert isinstance(result["metrics"], dict)
