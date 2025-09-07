"""Performance helpers for drawdown calculations.

This module centralizes the computation of drawdown metrics so that the
implementation is consistent across the project.  Only ``numpy`` and
``pandas`` are required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_curve_pct(equity: pd.Series) -> pd.Series:
    """Return the drawdown percentage curve of an equity series.

    Parameters
    ----------
    equity : pd.Series
        Equity values indexed by time. Values that are ``NaN`` or ``<= 0``
        are ignored for the calculation.

    Returns
    -------
    pd.Series
        Drawdown in percentage (values ``<= 0``) aligned with the input index.
        Points where the equity was ``NaN`` or non-positive are returned as
        ``NaN`` in the result.
    """

    if equity is None:
        return pd.Series(dtype=float)

    # Work on a clean copy, dropping NaNs and non-positive entries
    eq = equity.copy().dropna()
    eq = eq[eq > 0]

    if eq.empty:
        # Return an empty series with the same (possibly empty) index
        return pd.Series(dtype=float, index=equity.index)

    dd = 100.0 * (eq / eq.cummax() - 1.0)

    # Align back to the original index (NaN for removed values)
    return dd.reindex(equity.index)


def max_drawdown_pct(equity: pd.Series) -> float:
    """Return the maximum drawdown as a positive percentage.

    Parameters
    ----------
    equity : pd.Series
        Equity values.

    Returns
    -------
    float
        Maximum drawdown (e.g. ``25.58`` for ``-25.58%``). Returns ``0.0`` if
        the series is empty or contains no valid data.
    """

    dd = drawdown_curve_pct(equity)
    if dd.empty:
        return 0.0

    min_dd = dd.min(skipna=True)
    return float(-min_dd) if pd.notna(min_dd) else 0.0


def drawdown_curve_from_equity(equity: pd.Series) -> pd.DataFrame:
    """Compute drawdown data from an equity series.

    This helper is retained for backwards compatibility.  It builds upon
    :func:`drawdown_curve_pct` and returns a ``DataFrame`` with the equity,
    rolling peak and drawdown percentage.  The maximum drawdown percentage is
    stored in ``attrs['max_dd_pct']``.
    """

    eq = equity.astype(float).copy()
    eq_clean = eq.dropna()
    eq_clean = eq_clean[eq_clean > 0]
    rolling_peak = eq_clean.cummax().reindex(eq.index)
    dd_pct = drawdown_curve_pct(eq)

    df = pd.DataFrame({"equity": eq, "rolling_peak": rolling_peak, "dd_pct": dd_pct})
    df.attrs["max_dd_pct"] = float(dd_pct.min()) if len(dd_pct.dropna()) else 0.0
    return df


if __name__ == "__main__":
    # Quick manual tests
    sample = pd.Series([100, 120, 110, 130, 90, 150])
    print("Drawdown curve:")
    print(drawdown_curve_pct(sample))
    print("Max drawdown:", max_drawdown_pct(sample))
