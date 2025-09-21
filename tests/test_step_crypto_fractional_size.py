import pytest
from agent_core.position_sizing import compute_position_size


def test_crypto_fractional_size_allows_small_equity():
    size = compute_position_size(
        market='crypto', equity=100.0, leverage=5, entry_price=50000.0, allow_fractional_crypto=True
    )
    assert size > 0


def test_min_qty_blocks_when_above_size():
    size = compute_position_size(
        market='crypto', equity=100.0, leverage=1, entry_price=100000.0, min_qty=0.01, allow_fractional_crypto=True
    )
    assert size == 0
