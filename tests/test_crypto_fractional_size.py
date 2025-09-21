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
    def __init__(self, signal_ts: pd.Timestamp, payload: dict):
        self._signal_ts = signal_ts
        self._payload = payload

    def reset_for_new_day(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_signal(self, data: pd.DataFrame, current_day_levels=None, daily_candle_index: int = -1):
        return self._payload if data.index[-1] == self._signal_ts else "HOLD"


def _base_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-04 09:30", periods=4, freq="min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "open": np.full(4, 200.0),
            "high": np.full(4, 201.0),
            "low": np.full(4, 199.0),
            "close": np.full(4, 200.0),
        },
        index=idx,
    )
    frame.loc[idx[1], ["open", "high", "low", "close"]] = [200.0, 210.0, 199.0, 210.0]
    return frame


def _sync_flags(monkeypatch: pytest.MonkeyPatch, allow_fractional: bool) -> None:
    for module in (execution_mod, fast_mod):
        monkeypatch.setattr(module, "EXACT_MATCH_MODE", True, raising=False)
        monkeypatch.setattr(module, "SLIPPAGE_MODE", "none", raising=False)
        monkeypatch.setattr(module, "SLIPPAGE_VALUE", 0.0, raising=False)
        monkeypatch.setattr(module, "ALLOW_FRACTIONAL_SIZE_CRYPTO", allow_fractional, raising=False)
        monkeypatch.setattr(module, "TRADING_WINDOW_START", "09:30", raising=False)
        monkeypatch.setattr(module, "TRADING_WINDOW_END", "11:30", raising=False)
        monkeypatch.setattr(module, "FORCE_CLOSE_TIME", "13:00", raising=False)


def _run_case(monkeypatch: pytest.MonkeyPatch, allow_fractional: bool):
    _sync_flags(monkeypatch, allow_fractional)
    df = _base_df()
    signal_ts = df.index[0]
    payload = {"type": "BUY", "sl_price": 190.0, "tp1_price": 210.0}

    monkeypatch.setattr(
        fast_mod,
        "OpeningBreakRetestStrategy",
        lambda *args, **kwargs: StubStrategy(signal_ts, payload),
        raising=False,
    )

    executor = ExecutionSimulator(initial_capital=100.0, commission_per_trade=0.0, leverage=1)
    executor.configure_market(market="crypto", allow_fractional_crypto=allow_fractional)
    executor.set_strategy_reset_hook(lambda: None)
    for ts, row in df.iterrows():
        signal = payload if ts == signal_ts else "HOLD"
        executor.process_signal(signal, row)

    step_trades = executor.get_closed_trades()
    fast_trades, _ = run_fast_backtest_exact(
        df_day_with_context=df,
        day_start_index=0,
        day_levels={},
        initial_capital=100.0,
        commission_per_side=0.0,
        leverage=1.0,
        market="crypto",
        max_trades_per_day=1,
    )
    return step_trades, fast_trades


def test_crypto_fractional_size(monkeypatch: pytest.MonkeyPatch) -> None:
    step_trades, fast_trades = _run_case(monkeypatch, allow_fractional=True)
    assert len(step_trades) == 1
    assert fast_trades.shape[0] == 1
    step_size = step_trades[0]["size"]
    fast_size = fast_trades[0][3]
    assert step_size == pytest.approx(fast_size, rel=0, abs=1e-9)
    assert step_size > 0
    assert step_size != pytest.approx(round(step_size), rel=0, abs=1e-12)

    step_trades_off, fast_trades_off = _run_case(monkeypatch, allow_fractional=False)
    assert len(step_trades_off) == 0
    assert fast_trades_off.shape[0] == 0
