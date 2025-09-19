from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

from backtest_crew.bridge.config import BridgeConfig
from backtest_crew.bridge.datalake_feed import DatalakeFeed


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "datalake_sample"


def _cfg(tf: str, start: str, end: str, rename: str | None = None) -> BridgeConfig:
    return BridgeConfig(
        lake_root=str(FIXTURE_ROOT),
        source="binance",
        symbol="BTC-USD",
        tf=tf,
        date_from=start,
        date_to=end,
        rename_ts_to=rename,
    )


def test_m1_full_day_rows_and_bounds():
    feed = DatalakeFeed()
    cfg = _cfg("M1", "2025-08-01T00:00:00Z", "2025-08-02T00:00:00Z")
    df = feed.load_df(cfg)
    assert not df.empty
    assert len(df) == 1440
    assert pd.to_datetime("2025-08-01T00:00:00Z", utc=True) == df["ts"].min()
    assert pd.to_datetime("2025-08-01T23:59:00Z", utc=True) == df["ts"].max()


def test_m1_hourly_clipping_range():
    feed = DatalakeFeed()
    start = "2025-08-01T09:00:00Z"
    end = "2025-08-01T12:00:00Z"
    cfg = _cfg("M1", start, end)
    df = feed.load_df(cfg)
    # Half-open: [09:00, 12:00) -> 3h * 60 = 180 barras
    assert len(df) == 180
    assert df["ts"].min() == pd.to_datetime(start, utc=True)
    assert df["ts"].max() == pd.to_datetime("2025-08-01T11:59:00Z", utc=True)


def test_m5_full_day_rows():
    feed = DatalakeFeed()
    cfg = _cfg("M5", "2025-08-01T00:00:00Z", "2025-08-02T00:00:00Z")
    df = feed.load_df(cfg)
    assert not df.empty
    assert len(df) == 288


def test_missing_partition_returns_empty():
    feed = DatalakeFeed()
    # El fixture sólo contiene 2025-08-01
    cfg = _cfg("M1", "2025-08-02T00:00:00Z", "2025-08-03T00:00:00Z")
    df = feed.load_df(cfg)
    assert df is not None
    assert df.empty


def test_rename_ts_to():
    feed = DatalakeFeed()
    cfg = _cfg("M1", "2025-08-01T00:00:00Z", "2025-08-01T01:00:00Z", rename="timestamp")
    df = feed.load_df(cfg)
    assert "timestamp" in df.columns
    assert pd.Series.equals(
        df["timestamp"].reset_index(drop=True),
        df["ts"].reset_index(drop=True),
    )
