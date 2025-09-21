import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agent_core.execution as execution_mod
import strategies.vectorized_obr_exact as fast_mod
from agent_core.execution import ExecutionSimulator
from strategies.vectorized_obr_exact import run_fast_backtest_exact


class StubStrategy:
    def __init__(self, signals: dict[pd.Timestamp, dict]):
        self._signals = signals

    def reset_for_new_day(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_signal(self, data: pd.DataFrame, current_day_levels: dict | None = None, daily_candle_index: int = -1):
        ts = data.index[-1]
        return self._signals.get(ts, "HOLD")


def _build_day_dataframe() -> pd.DataFrame:
    idx = pd.date_range(
        "2024-01-02 09:30", "2024-01-02 13:00", freq="min", tz="America/New_York"
    )
    base = pd.DataFrame(
        {
            "open": np.full(len(idx), 100.0),
            "high": np.full(len(idx), 100.5),
            "low": np.full(len(idx), 99.5),
            "close": np.full(len(idx), 100.0),
        },
        index=idx,
    )
    base.loc[idx[0], ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    base.loc[idx[1], ["open", "high", "low", "close"]] = [100.2, 102.4, 100.1, 102.0]
    base.loc[idx[30], ["open", "high", "low", "close"]] = [98.0, 99.0, 97.5, 98.0]
    base.loc[idx[31], ["open", "high", "low", "close"]] = [97.8, 98.5, 95.5, 96.0]
    return base


def _prepare_signals(idx: pd.DatetimeIndex) -> dict[pd.Timestamp, dict]:
    return {
        idx[0]: {"type": "BUY", "sl_price": 99.0, "tp1_price": 102.0},
        idx[30]: {"type": "SELL", "sl_price": 99.5, "tp1_price": 96.0},
    }


def _sync_module_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (execution_mod, fast_mod):
        monkeypatch.setattr(module, "EXACT_MATCH_MODE", True, raising=False)
        monkeypatch.setattr(module, "SLIPPAGE_MODE", "none", raising=False)
        monkeypatch.setattr(module, "SLIPPAGE_VALUE", 0.0, raising=False)
        monkeypatch.setattr(module, "TRADING_WINDOW_START", "09:30", raising=False)
        monkeypatch.setattr(module, "TRADING_WINDOW_END", "11:30", raising=False)
        monkeypatch.setattr(module, "FORCE_CLOSE_TIME", "13:00", raising=False)


def test_equivalence_fast_vs_visual(monkeypatch: pytest.MonkeyPatch) -> None:
    _sync_module_flags(monkeypatch)

    df = _build_day_dataframe()
    signals = _prepare_signals(df.index)

    monkeypatch.setattr(
        fast_mod,
        "OpeningBreakRetestStrategy",
        lambda *args, **kwargs: StubStrategy(signals),
        raising=False,
    )

    executor = ExecutionSimulator(initial_capital=1000.0, commission_per_trade=0.85, leverage=5)
    executor.configure_market(market="stocks")
    executor.set_strategy_reset_hook(lambda: None)

    for ts, row in df.iterrows():
        payload = signals.get(ts, "HOLD")
        executor.process_signal(payload, row)

    step_trades = executor.get_closed_trades()
    fast_trades, _ = run_fast_backtest_exact(
        df_day_with_context=df,
        day_start_index=0,
        day_levels={},
        initial_capital=1000.0,
        commission_per_side=0.85,
        leverage=5.0,
        market="stocks",
        stop_after_first_win=False,
        max_trades_per_day=5,
    )

    assert len(step_trades) == fast_trades.shape[0] == 2

    fast_records = []
    for row in fast_trades:
        entry_ts = pd.to_datetime(row[0], unit="s", utc=True).tz_convert("America/New_York")
        exit_ts = pd.to_datetime(row[1], unit="s", utc=True).tz_convert("America/New_York")
        fast_records.append(
            {
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "direction": "LONG" if int(row[2]) == 1 else "SHORT",
                "size": float(row[3]),
                "entry_price": float(row[4]),
                "exit_price": float(row[5]),
                "pnl": float(row[6]),
            }
        )

    for step_trade, fast_trade in zip(step_trades, fast_records):
        assert step_trade["entry_time"] == fast_trade["entry_time"]
        assert step_trade["exit_time"] == fast_trade["exit_time"]
        assert step_trade["direction"] == fast_trade["direction"]
        assert step_trade["size"] == pytest.approx(fast_trade["size"], rel=0, abs=1e-9)
        assert step_trade["entry_price"] == pytest.approx(fast_trade["entry_price"], rel=0, abs=1e-9)
        assert step_trade["exit_price"] == pytest.approx(fast_trade["exit_price"], rel=0, abs=1e-9)
        assert step_trade["pnl_net"] == pytest.approx(fast_trade["pnl"], rel=0, abs=1e-9)

    total_step = sum(trade["pnl_net"] for trade in step_trades)
    total_fast = sum(record["pnl"] for record in fast_records)
    assert total_step == pytest.approx(total_fast, rel=0, abs=1e-9)
