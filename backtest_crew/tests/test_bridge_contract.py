import os
import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest_crew.bridge import BridgeConfig, DatalakeFeed

READER_AVAILABLE = True
try:
    from datalake.read.api import read_range_df  # type: ignore
except Exception:  # pragma: no cover
    READER_AVAILABLE = False

REQUIRED_COUNTS = {"M1": 1440, "M5": 288, "M15": 96, "M30": 48}

LAKE_DAY = "2025-08-01"
DATE_FROM = f"{LAKE_DAY}T00:00:00Z"
DATE_TO = f"2025-08-02T00:00:00Z"


def lake_root_or_skip():
    lake_root = os.getenv("LAKE_ROOT")
    if not lake_root:
        pytest.skip("LAKE_ROOT no configurado; se omiten tests de bridge")
    return lake_root

@pytest.mark.parametrize("tf", ["M1", "M5", "M15", "M30"])
def test_counts_per_timeframe(tf):
    if not READER_AVAILABLE:
        pytest.skip("Reader datalake no disponible")
    lake_root = lake_root_or_skip()
    cfg = BridgeConfig(lake_root=lake_root, source="binance", symbol="BTC-USD", tf=tf, date_from=DATE_FROM, date_to=DATE_TO)
    df = DatalakeFeed().load_df(cfg)
    assert len(df) == REQUIRED_COUNTS[tf], f"Conteo inesperado para {tf}: {len(df)}"
    assert df['ts'].is_monotonic_increasing
    assert df['ts'].is_unique
    assert df['ts'].min() >= pd.Timestamp(DATE_FROM)
    # Última vela debe ser < DATE_TO (exclusivo), y cubrir hasta 23:59.
    assert df['ts'].max() < pd.Timestamp(DATE_TO)
    # Chequeo de cobertura: para M1 la última debería ser 23:59
    if tf == 'M1':
        assert df['ts'].max().hour == 23 and df['ts'].max().minute == 59


def test_rename_ts():
    if not READER_AVAILABLE:
        pytest.skip("Reader datalake no disponible")
    lake_root = lake_root_or_skip()
    cfg = BridgeConfig(lake_root=lake_root, source="binance", symbol="BTC-USD", tf="M1", date_from=DATE_FROM, date_to=DATE_TO, rename_ts_to="timestamp")
    df = DatalakeFeed().load_df(cfg)
    assert 'timestamp' in df.columns
    assert (df['timestamp'] == df['ts']).all()
