import pandas as pd


def drawdown_curve_pct(equity: pd.Series) -> pd.DataFrame:
    """Return drawdown curve (%) from a dollar-denominated equity series.

    Parameters
    ----------
    equity : pd.Series
        Equity curve in dollars indexed by datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ``equity``: original equity values.
        - ``rolling_peak``: running peak of the equity curve.
        - ``dd_pct``: drawdown percentage (negative values).
    """

    eq = equity.astype(float).copy()
    rolling_peak = eq.cummax()
    dd_pct = (eq / rolling_peak - 1.0) * 100.0
    return pd.DataFrame({"equity": eq, "rolling_peak": rolling_peak, "dd_pct": dd_pct})


def max_drawdown_pct(equity: pd.Series) -> float:
    """Compute the maximum drawdown percentage (as a positive value)."""

    if equity is None or len(equity) == 0:
        return 0.0

    dd_curve = drawdown_curve_pct(equity)
    return float(abs(dd_curve["dd_pct"].min())) if len(dd_curve) else 0.0


def drawdown_curve_from_equity(equity: pd.Series) -> pd.DataFrame:
    """Backward compatible helper that also stores max drawdown in ``attrs``."""

    df = drawdown_curve_pct(equity)
    df.attrs["max_dd_pct"] = -float(abs(df["dd_pct"].min())) if len(df) else 0.0
    return df
