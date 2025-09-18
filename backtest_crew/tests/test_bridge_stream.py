import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest_crew.bridge import BridgeConfig, DatalakeFeed

READER_AVAILABLE = True
try:
    from datalake.read.api import read_range_df  # type: ignore
except Exception:  # pragma: no cover
    READER_AVAILABLE = False

RANGE_FROM = "2025-08-01T14:30:00Z"
RANGE_TO = "2025-08-01T14:50:00Z"  # 20 minutos -> 20 barras M1 esperadas


def lake_root_or_skip():
    lake_root = os.getenv("LAKE_ROOT")
    if not lake_root:
        pytest.skip("LAKE_ROOT no configurado; se omite test stream")
    return lake_root


def test_iter_bars_range():
    if not READER_AVAILABLE:
        pytest.skip("Reader datalake no disponible")
    lake_root = lake_root_or_skip()
    cfg = BridgeConfig(lake_root=lake_root, source="binance", symbol="BTC-USD", tf="M1", date_from=RANGE_FROM, date_to=RANGE_TO, mode="stream", speed_bps=0)
    feed = DatalakeFeed()
    bars = list(feed.iter_bars(cfg))
    assert len(bars) == 20, f"Esperadas 20 barras, obtuvimos {len(bars)}"
    ts_values = [b['ts'] for b in bars]
    assert all(ts_values[i] < ts_values[i+1] for i in range(len(ts_values)-1)), "No monótono"
