import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agent_core.execution as execution_mod
import strategies.vectorized_obr_exact as fast_mod
from agent_core.execution import ExecutionSimulator
from shared.slippage import apply_slippage
from strategies.vectorized_obr_exact import run_fast_backtest_exact


class StubStrategy:
    def __init__(self, signal_ts: pd.Timestamp, payload: dict):
        self._timestamp = signal_ts
        self._payload = payload

    def reset_for_new_day(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_signal(self, data: pd.DataFrame, current_day_levels=None, daily_candle_index: int = -1):
        ts = data.index[-1]
        if ts == self._timestamp:
            return self._payload
        return "HOLD"


def _sync_flags(monkeypatch: pytest.MonkeyPatch, mode: str, value: float) -> None:
    for module in (execution_mod, fast_mod):
        monkeypatch.setattr(module, "EXACT_MATCH_MODE", True, raising=False)
        monkeypatch.setattr(module, "SLIPPAGE_MODE", mode, raising=False)
        monkeypatch.setattr(module, "SLIPPAGE_VALUE", value, raising=False)
        monkeypatch.setattr(module, "TRADING_WINDOW_START", "09:30", raising=False)
        monkeypatch.setattr(module, "TRADING_WINDOW_END", "11:30", raising=False)
        monkeypatch.setattr(module, "FORCE_CLOSE_TIME", "13:00", raising=False)


def _build_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-03 09:30", periods=5, freq="min", tz="America/New_York")
    base = pd.DataFrame(
        {
            "open": np.full(5, 100.0),
            "high": np.full(5, 100.5),
            "low": np.full(5, 99.5),
            "close": np.full(5, 100.0),
        },
        index=idx,
    )
    base.loc[idx[0], ["open", "high", "low", "close"]] = [100.0, 100.5, 99.5, 100.0]
    base.loc[idx[1], ["open", "high", "low", "close"]] = [100.0, 102.0, 100.0, 102.0]
    return base


@pytest.mark.parametrize("mode,value", [("none", 0.0), ("points", 0.5), ("percent", 0.01)])
def test_slippage_modes(monkeypatch: pytest.MonkeyPatch, mode: str, value: float) -> None:
    _sync_flags(monkeypatch, mode, value)

    df = _build_df()
    signal_ts = df.index[0]
    payload = {"type": "BUY", "sl_price": 99.0, "tp1_price": 102.0}
    monkeypatch.setattr(
        fast_mod,
        "OpeningBreakRetestStrategy",
        lambda *args, **kwargs: StubStrategy(signal_ts, payload),
        raising=False,
    )

    executor = ExecutionSimulator(initial_capital=1000.0, commission_per_trade=0.0, leverage=1)
    executor.configure_market(market="stocks")
    executor.set_strategy_reset_hook(lambda: None)

    for ts, row in df.iterrows():
        signal = payload if ts == signal_ts else "HOLD"
        executor.process_signal(signal, row)

    step_trade = executor.get_closed_trades()[0]
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
    fast_trade = fast_trades[0]

    raw_entry = 100.0
    raw_exit = 102.0
    expected_entry = apply_slippage(raw_entry, "buy", mode, value)
    expected_exit = apply_slippage(raw_exit, "sell", mode, value)

    assert step_trade["entry_price"] == pytest.approx(expected_entry, rel=0, abs=1e-9)
    assert step_trade["exit_price"] == pytest.approx(expected_exit, rel=0, abs=1e-9)
    assert fast_trade[4] == pytest.approx(expected_entry, rel=0, abs=1e-9)
    assert fast_trade[5] == pytest.approx(expected_exit, rel=0, abs=1e-9)
