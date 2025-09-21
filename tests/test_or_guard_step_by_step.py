import pytest

pd = pytest.importorskip("pandas")

from strategies.opening_br_strategy import OpeningBreakRetestStrategy
from strategies.level_fsm import State


def _build_guard_dataset() -> pd.DataFrame:
    idx = pd.date_range(
        "2024-01-02 09:30",
        periods=16,
        freq="min",
        tz="America/New_York",
    )
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.4,
            "low": 99.6,
            "close": 100.0,
        },
        index=idx,
    )

    # Candle that breaks ORH after the guard window (09:41)
    df.loc[idx[11], ["open", "high", "low", "close"]] = [100.4, 101.6, 99.9, 101.1]
    # Stabilise above ORH waiting for retest
    df.loc[idx[12], ["open", "high", "low", "close"]] = [101.0, 101.4, 100.8, 101.2]
    # Retest within the tolerance zone and close back above ORH (09:43)
    df.loc[idx[13], ["open", "high", "low", "close"]] = [101.1, 101.3, 100.6, 101.05]

    return df


def test_or_levels_guard_delays_processing_until_or_end():
    df = _build_guard_dataset()
    levels = {"ORH": 100.5, "ORL": 99.5}

    strat = OpeningBreakRetestStrategy(
        or_guard_enabled=True,
        or_exec_minutes=10,
        or_tz="America/New_York",
    )
    strat.reset_for_new_day()

    last_signal = "HOLD"
    for i, ts in enumerate(df.index):
        window = df.iloc[: i + 1]
        signal = strat.get_signal(window, current_day_levels=levels, daily_candle_index=i)
        if window.shape[0] < 2:
            continue

        if ts < pd.Timestamp("2024-01-02 09:40", tz="America/New_York"):
            assert strat.level_fsms["ORH"].state == State.IDLE
        elif ts == pd.Timestamp("2024-01-02 09:41", tz="America/New_York"):
            assert strat.level_fsms["ORH"].state == State.BROKEN
        elif ts == pd.Timestamp("2024-01-02 09:43", tz="America/New_York"):
            assert isinstance(signal, dict)
            assert signal["type"].upper() == "BUY"
            assert strat.level_fsms["ORH"].state == State.SIGNAL_EMITTED
        last_signal = signal

    assert isinstance(last_signal, dict)
    assert last_signal["level"] == "ORH"
