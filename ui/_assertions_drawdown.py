from shared.metrics import drawdown_stats


def assert_dd_consistency(equity, mdd_card_value, mdd_table_value, tol=1e-2):
    stats = drawdown_stats(equity)
    mdd = stats['max_drawdown_pct']
    # 1) Tarjeta = mdd
    assert abs(mdd_card_value - mdd) <= tol, f"Card MDD {mdd_card_value} != {mdd}"
    # 2) Tabla = mdd
    assert abs(mdd_table_value - mdd) <= tol, f"Table MDD {mdd_table_value} != {mdd}"
    # 3) Curva: |min(dd_series_pct)| = mdd
    dd_series = stats['dd_series_pct']
    assert abs(abs(dd_series.min()) - mdd) <= tol, f"Curve min {dd_series.min()} != -{mdd}"
