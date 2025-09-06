import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent_core.data_manager import DataManager


def _dummy_contract(symbol):
    return SimpleNamespace(symbol=symbol, secType='CASH')


def _forex_fetch_mock(self, contract, end_dt, duration, timeframe, rth, what_to_show):
    if what_to_show == 'MIDPOINT':
        return pd.DataFrame()
    index = pd.DatetimeIndex([pd.Timestamp('2024-01-01', tz='UTC')])
    data = {'open': [1.0], 'high': [1.0], 'low': [1.0], 'close': [1.0], 'volume': [0]}
    return pd.DataFrame(data, index=index)


def test_fallback_to_bid_ask(monkeypatch):
    dm = DataManager()
    monkeypatch.setattr(dm, '_resolve_contract', lambda *args, **kwargs: _dummy_contract('EURUSD'))
    monkeypatch.setattr(DataManager, '_fetch_data_core', _forex_fetch_mock)
    monkeypatch.setattr('agent_core.data_manager.time.sleep', lambda x: None)

    start = end = datetime.date(2024, 1, 1)
    df = dm.get_main_data(
        symbol='EURUSD',
        timeframe='1 min',
        sec_type='FOREX',
        exchange='IDEALPRO',
        currency='USD',
        rth=False,
        what_to_show='MIDPOINT',
        download_start_date=start,
        download_end_date=end,
        use_cache=False,
        market='forex',
    )

    assert not df.empty


def test_no_fallback_for_stocks(monkeypatch):
    dm = DataManager()
    monkeypatch.setattr(dm, '_resolve_contract', lambda *args, **kwargs: _dummy_contract('AAPL'))
    calls = []

    def _stock_fetch_mock(self, contract, end_dt, duration, timeframe, rth, what_to_show):
        calls.append(what_to_show)
        return pd.DataFrame()

    monkeypatch.setattr(DataManager, '_fetch_data_core', _stock_fetch_mock)
    monkeypatch.setattr('agent_core.data_manager.time.sleep', lambda x: None)

    start = end = datetime.date(2024, 1, 1)
    dm.get_main_data(
        symbol='AAPL',
        timeframe='1 min',
        sec_type='STK',
        exchange='SMART',
        currency='USD',
        rth=False,
        what_to_show='MIDPOINT',
        download_start_date=start,
        download_end_date=end,
        use_cache=False,
        market='stocks',
    )

    assert calls == ['MIDPOINT', 'MIDPOINT', 'MIDPOINT']
