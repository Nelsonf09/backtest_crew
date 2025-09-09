from shared.metrics import drawdown_stats


def assert_dd_row_consistency(equity_series, mdd_in_row, tol=0.01):
    stats = drawdown_stats(equity_series)
    mdd_true = stats['max_drawdown_pct']
    dd_series = stats['dd_series_pct']
    assert abs(mdd_in_row - mdd_true) <= tol, f"Resumen: MDD={mdd_in_row} != {mdd_true}"
    assert abs(abs(dd_series.min()) - mdd_true) <= tol, f"Curva vs resumen difieren: {dd_series.min()} vs {mdd_true}"
