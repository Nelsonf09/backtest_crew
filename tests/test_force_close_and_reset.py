import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agent_core.execution as execution_mod
import strategies.vectorized_obr_exact as fast_mod
from agent_core.execution import ExecutionSimulator, TradeState
from strategies.vectorized_obr_exact import run_fast_backtest_exact


class StubStrategy:
    def __init__(self, signal_ts: pd.Timestamp, payload: dict):
        self._signal_ts = signal_ts
        self._payload = payload

    def reset_for_new_day(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_signal(self, data: pd.DataFrame, current_day_levels=None, daily_candle_index: int = -1):
        return self._payload if data.index[-1] == self._signal_ts else "HOLD"


class ResetRecorder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def _build_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-05 09:30", "2024-01-05 13:05", freq="min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "open": np.full(len(idx), 100.0),
            "high": np.full(len(idx), 100.2),
            "low": np.full(len(idx), 99.8),
            "close": np.full(len(idx), 100.0),
        },
        index=idx,
    )
    return frame


def _sync_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (execution_mod, fast_mod):
        monkeypatch.setattr(module, "EXACT_MATCH_MODE", True, raising=False)
        monkeypatch.setattr(module, "SLIPPAGE_MODE", "none", raising=False)
        monkeypatch.setattr(module, "SLIPPAGE_VALUE", 0.0, raising=False)
        monkeypatch.setattr(module, "TRADING_WINDOW_START", "09:30", raising=False)
        monkeypatch.setattr(module, "TRADING_WINDOW_END", "11:30", raising=False)
        monkeypatch.setattr(module, "FORCE_CLOSE_TIME", "13:00", raising=False)


def test_force_close_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    _sync_flags(monkeypatch)
    df = _build_df()
    signal_ts = pd.Timestamp("2024-01-05 10:30", tz="America/New_York")
    payload = {"type": "BUY", "sl_price": 95.0, "tp1_price": 120.0}

    monkeypatch.setattr(
        fast_mod,
        "OpeningBreakRetestStrategy",
        lambda *args, **kwargs: StubStrategy(signal_ts, payload),
        raising=False,
    )

    recorder = ResetRecorder()
    executor = ExecutionSimulator(initial_capital=1000.0, commission_per_trade=0.0, leverage=1)
    executor.configure_market(market="stocks")
    executor.set_strategy_reset_hook(recorder)

    for ts, row in df.iterrows():
        signal = payload if ts == signal_ts else "HOLD"
        executor.process_signal(signal, row)

    step_trades = executor.get_closed_trades()
    assert len(step_trades) == 1
    step_trade = step_trades[0]
    assert step_trade["exit_reason"] == "ForceClose"
    assert step_trade["exit_time"] == pd.Timestamp("2024-01-05 13:00", tz="America/New_York")
    assert executor.account_fsm.is_in_state(TradeState.FLAT)
    assert recorder.calls == 1

    fast_trades, _ = run_fast_backtest_exact(
        df_day_with_context=df,
        day_start_index=0,
        day_levels={},
        initial_capital=1000.0,
        commission_per_side=0.0,
        leverage=1.0,
        market="stocks",
        max_trades_per_day=1,
    )
    assert fast_trades.shape[0] == 1
    fast_trade = fast_trades[0]
    exit_ts = pd.to_datetime(fast_trade[1], unit="s", utc=True).tz_convert("America/New_York")
    assert exit_ts == pd.Timestamp("2024-01-05 13:00", tz="America/New_York")
    assert int(fast_trade[7]) == 3
