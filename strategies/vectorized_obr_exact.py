# strategies/vectorized_obr_exact.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Usamos la MISMA estrategia paso-a-paso para garantizar equivalencia exacta
from .opening_br_strategy import OpeningBreakRetestStrategy
from shared.config import (
    EXACT_MATCH_MODE,
    SLIPPAGE_MODE,
    SLIPPAGE_VALUE,
    ALLOW_FRACTIONAL_SIZE_CRYPTO,
    TRADING_WINDOW_START,
    TRADING_WINDOW_END,
    FORCE_CLOSE_TIME,
)
from shared.slippage import apply_slippage
from shared.time_windows import is_trading_window, is_force_close
from agent_core.position_sizing import compute_position_size

def _is_trading_window(ts: pd.Timestamp) -> bool:
    if EXACT_MATCH_MODE:
        return is_trading_window(ts, TRADING_WINDOW_START, TRADING_WINDOW_END)
    h, m = ts.hour, ts.minute
    return (h == 9 and m >= 30) or (h == 10) or (h == 11 and m <= 30)


def _force_close_time(ts: pd.Timestamp) -> bool:
    if EXACT_MATCH_MODE:
        return is_force_close(ts, FORCE_CLOSE_TIME)
    return ts.hour >= 13

def _to_epoch_seconds(ts: pd.Timestamp) -> float:
    if ts.tzinfo is None:
        # Asume UTC si viene naive (evita errores)
        return pd.Timestamp(ts).tz_localize("UTC").timestamp()
    return ts.timestamp()

def run_fast_backtest_exact(
    # --- Datos OHLC del día (DataFrame con contexto de lookback) ---
    df_day_with_context: pd.DataFrame,
    day_start_index: int,
    # --- Niveles del día en dict, como los usa la estrategia paso-a-paso ---
    day_levels: Dict[str, float],
    # --- Parámetros de estrategia (EXACTOS al paso-a-paso) ---
    ema_filter_mode: str = "Desactivado",
    max_retest_candles: int = 15,
    sl_method: str = "LOOKBACK_MIN_MAX",
    sl_lookback: int = 2,
    risk_reward_ratio: float = 2.0,
    level_ranges: Optional[Dict[str, float]] = None,
    # --- Parámetros de ejecución (como el motor rápido) ---
    initial_capital: float = 1000.0,
    commission_per_side: float = 0.85,
    leverage: float = 5.0,
    # --- Nuevo parámetro para soportar distintos mercados ---
    market: str = "stocks",
    symbol: str = "",
    # --- Reglas de sesión del motor rápido (se mantienen) ---
    stop_after_first_win: bool = True,
    # --- INICIO DE LA MODIFICACIÓN ---
    # El valor ahora es dinámico, calculado a partir de un % del capital
    first_trade_loss_stop: float = -60.0,
    # --- FIN DE LA MODIFICACIÓN ---
    max_trades_per_day: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
      closed_trades: np.ndarray shape (N,8): [entry_time, exit_time, direction(1/-1), size, entry, exit, pnl_net, exit_reason]
      equity_history: np.ndarray shape (T,2): [timestamp, equity]
    Donde exit_reason: 1=TP, 2=SL, 3=Timeout (13:00)
    """
    if df_day_with_context is None or df_day_with_context.empty or day_start_index >= len(df_day_with_context):
        return np.zeros((0, 8), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    pip_size = None
    if market.lower() == "forex":
        pip_size = 0.01 if symbol.upper().endswith("JPY") else 0.0001

    def _quant(price: float) -> float:
        if pip_size is None or not np.isfinite(price):
            return float(price)
        return float(np.round(price / pip_size) * pip_size)

    df = df_day_with_context.copy()
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if pip_size is not None:
            df[c] = np.round(df[c] / pip_size) * pip_size

    if pip_size is not None:
        day_levels = {k: _quant(v) for k, v in day_levels.items()}
    else:
        day_levels = dict(day_levels)

    # === Instancia de estrategia paso-a-paso con los MISMOS parámetros ===
    strat = OpeningBreakRetestStrategy(
        max_retest_candles=max_retest_candles,
        level_ranges=level_ranges or {},
        ema_periods=[9, 21, 50],
        ema_filter_mode=ema_filter_mode,
        sl_method=sl_method,
        sl_lookback=sl_lookback,
        risk_reward_ratio=risk_reward_ratio,
    )

    # Se llama al reseteo completo UNA SOLA VEZ al inicio del día.
    strat.reset_for_new_day()

    # --- Estado de ejecución (idéntico al motor rápido) ---
    equity = float(initial_capital)
    cash = float(initial_capital)
    position_size = 0.0
    entry_price = 0.0
    direction = 0
    current_sl = np.nan
    current_tp = np.nan
    active_trade_entry_time = None

    trades_today = 0
    stop_trading_for_day = False

    max_trades = 1000
    closed_trades = np.zeros((max_trades, 8), dtype=np.float64)
    equity_history = np.zeros((len(df) - day_start_index, 2), dtype=np.float64)
    trade_count = 0
    equity_hist_count = 0

    # === Bucle vela a vela, pero con SEÑALES del paso-a-paso ===
    for i in range(day_start_index, len(df)):
        ts = df.index[i]
        row = df.iloc[i]

        if direction != 0:
            is_closed, exit_reason = False, 0
            exit_price_raw = 0.0
            if direction == 1:
                if row["high"] >= current_tp:
                    is_closed, exit_price_raw, exit_reason = True, float(current_tp), 1
                elif row["close"] <= current_sl:
                    is_closed, exit_price_raw, exit_reason = True, float(current_sl), 2
            else:
                if row["low"] <= current_tp:
                    is_closed, exit_price_raw, exit_reason = True, float(current_tp), 1
                elif row["close"] >= current_sl:
                    is_closed, exit_price_raw, exit_reason = True, float(current_sl), 2

            if not is_closed and _force_close_time(ts):
                is_closed, exit_price_raw, exit_reason = True, _quant(float(row["close"])), 3

            if is_closed:
                exit_price = _quant(exit_price_raw)
                if EXACT_MATCH_MODE:
                    exit_side = 'sell' if direction == 1 else 'buy'
                    exit_price = _quant(apply_slippage(exit_price, exit_side, SLIPPAGE_MODE, SLIPPAGE_VALUE))
                pnl_gross = (exit_price - entry_price) * position_size if direction == 1 else (entry_price - exit_price) * position_size
                pnl_net = pnl_gross - (commission_per_side * 2.0)
                cash += pnl_gross
                equity += pnl_net

                if trade_count < max_trades:
                    closed_trades[trade_count, :] = [
                        _to_epoch_seconds(active_trade_entry_time), _to_epoch_seconds(ts),
                        float(direction), float(position_size), float(entry_price),
                        float(exit_price), float(pnl_net), float(exit_reason),
                    ]
                    trade_count += 1

                trades_today += 1
                if stop_after_first_win and pnl_net > 0:
                    stop_trading_for_day = True
                elif trades_today == 1 and pnl_net <= first_trade_loss_stop:
                    stop_trading_for_day = True
                elif trades_today >= max_trades_per_day:
                    stop_trading_for_day = True

                direction, position_size = 0, 0.0

                strat.reset()

        unrealized = 0.0
        if direction != 0:
            unrealized = (float(row["close"]) - entry_price) * position_size if direction == 1 else (entry_price - float(row["close"])) * position_size
        
        equity_point = cash + unrealized + (position_size * entry_price if direction != 0 else 0.0)
        
        if equity_hist_count < len(equity_history):
            equity_history[equity_hist_count, 0] = _to_epoch_seconds(ts)
            equity_history[equity_hist_count, 1] = float(equity_point)
            equity_hist_count += 1

        if direction == 0 and not stop_trading_for_day and _is_trading_window(ts):
            # El índice diario es relativo al inicio del día, no al inicio del dataframe combinado
            daily_candle_index = i - day_start_index
            signal = strat.get_signal(df.iloc[: i + 1], current_day_levels=day_levels, daily_candle_index=daily_candle_index)

            if isinstance(signal, dict):
                sig_type = str(signal.get("type", "HOLD")).upper()
                if sig_type in ("BUY", "SELL"):
                    raw_entry_price = _quant(float(row["close"]))
                    direction = 1 if sig_type == "BUY" else -1
                    sl_price = _quant(float(signal.get("sl_price", np.nan)))
                    tp_price = _quant(float(signal.get("tp1_price", np.nan)))

                    if not (np.isfinite(sl_price) and np.isfinite(tp_price) and raw_entry_price > 0):
                        direction = 0
                    else:
                        entry_price = raw_entry_price
                        if EXACT_MATCH_MODE:
                            entry_side = 'buy' if direction == 1 else 'sell'
                            entry_price = _quant(apply_slippage(entry_price, entry_side, SLIPPAGE_MODE, SLIPPAGE_VALUE))

                        size = compute_position_size(
                            market=market,
                            equity=equity,
                            leverage=float(leverage),
                            entry_price=entry_price,
                            lot_step=None,
                            min_qty=None,
                            allow_fractional_crypto=ALLOW_FRACTIONAL_SIZE_CRYPTO,
                        )
                        if size <= 0:
                            direction = 0
                        else:
                            position_size = float(size)
                            cash -= commission_per_side
                            current_sl = sl_price
                            current_tp = tp_price
                            active_trade_entry_time = ts

    return closed_trades[:trade_count], equity_history[:equity_hist_count]
