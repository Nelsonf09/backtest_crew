from shared.metrics import drawdown_stats


def normalize_mdd_from_equity(equity_series, mdd_value):
    """Si viene un valor sospechoso (>50%) o difiere >0.5 pp del cálculo de serie, corrige usando drawdown_stats."""
    stats = drawdown_stats(equity_series)
    mdd_true = stats['max_drawdown_pct']
    try:
        if (mdd_value is None) or (mdd_value > 50.0) or (abs(mdd_value - mdd_true) > 0.5):
            return mdd_true
    except Exception:
        return mdd_true
    return mdd_value
