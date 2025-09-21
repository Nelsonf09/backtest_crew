import pytest
from shared.slippage import apply_slippage


def test_slippage_none_default():
    entry = apply_slippage(100.0, 'long', 'none', 0.0)
    exitp = apply_slippage(105.0, 'short', 'none', 0.0)
    assert entry == 100.0
    assert exitp == 105.0
