# strategies/vectorized_obr.py
"""
Motor de backtesting rápido para la estrategia Opening Break & Retest,
acelerado con Numba.
VERSIÓN DE ALTA FIDELIDAD: Sincronizado con la lógica detallada del motor visual
y con las REGLAS DE SESIÓN reincorporadas.
"""
import numpy as np
from numba import jit

# --- Constantes para los estados de la FSM ---
IDLE = 0
BROKEN = 1
INVALIDATED = 3

@jit(nopython=True)
def calculate_emas_numba(prices, periods):
    """ Calcula EMAs de forma eficiente con Numba. """
    num_periods = len(periods)
    emas = np.full((len(prices), num_periods), np.nan, dtype=np.float64)
    alphas = 2.0 / (periods + 1.0)
    
    for p_idx in range(num_periods):
        period = periods[p_idx]
        alpha = alphas[p_idx]
        
        if len(prices) >= period:
            emas[int(period)-1, p_idx] = np.mean(prices[:int(period)])
            for i in range(int(period), len(prices)):
                emas[i, p_idx] = prices[i] * alpha + emas[i-1, p_idx] * (1 - alpha)
    return emas

@jit(nopython=True)
def calculate_sl_numba(
    signal_type: int, break_idx: int, level_name_code: int,
    low_prices: np.ndarray, high_prices: np.ndarray,
    sl_default_lookback: int, sl_reduced_lookback: int,
    orh_price: float, orl_price: float
) -> float:
    """
    Función de ayuda para calcular el Stop Loss con la misma lógica que el motor visual.
    Incluye el ajuste especial para niveles ORH/ORL.
    """
    start_idx_default = max(0, break_idx - (sl_default_lookback - 1))
    end_idx = break_idx + 1
    
    sl_price_default = np.nan
    if signal_type == 1: # BUY
        sl_price_default = np.min(low_prices[start_idx_default:end_idx])
    else: # SELL
        sl_price_default = np.max(high_prices[start_idx_default:end_idx])

    is_orh_or_orl = (level_name_code == 1 or level_name_code == 2)
    if not is_orh_or_orl or np.isnan(orh_price) or np.isnan(orl_price):
        return sl_price_default

    or_midpoint = (orh_price + orl_price) / 2.0
    reduce_lookback = False
    if signal_type == 1 and sl_price_default < or_midpoint: # BUY
        reduce_lookback = True
    elif signal_type == -1 and sl_price_default > or_midpoint: # SELL
        reduce_lookback = True

    if reduce_lookback:
        start_idx_reduced = max(0, break_idx - (sl_reduced_lookback - 1))
        sl_price_reduced = np.nan
        if signal_type == 1: # BUY
            sl_price_reduced = np.min(low_prices[start_idx_reduced:end_idx])
        else: # SELL
            sl_price_reduced = np.max(high_prices[start_idx_reduced:end_idx])
        
        return sl_price_reduced if not np.isnan(sl_price_reduced) else sl_price_default
    
    return sl_price_default


@jit(nopython=True)
def run_fast_backtest(
    # --- Datos de las velas ---
    open_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray,
    timestamps: np.ndarray, hours: np.ndarray, minutes: np.ndarray,
    # --- Datos de niveles ---
    levels_names: np.ndarray, levels_prices: np.ndarray, levels_directions: np.ndarray,
    # --- Parámetros de la estrategia ---
    max_retest_candles: int, risk_reward_ratio: float,
    sl_default_lookback: int, sl_reduced_lookback: int,
    level_ranges: np.ndarray,
    orh_price: float, orl_price: float,
    # --- Parámetros de ejecución ---
    initial_capital: float, commission_per_side: float, leverage: float,
    # --- Parámetros del filtro EMA ---
    ema_filter_mode: int,
    ema_periods: np.ndarray
):
    num_candles = len(close_prices)
    
    valid_indices = []
    if not np.isnan(orh_price) and not np.isnan(orl_price):
        for i in range(len(levels_prices)):
            if levels_names[i] == 1 or levels_names[i] == 2:
                valid_indices.append(i)
                continue
            if not (orl_price < levels_prices[i] < orh_price):
                valid_indices.append(i)
    else:
        for i in range(len(levels_prices)):
            valid_indices.append(i)
            
    valid_indices_np = np.array(valid_indices, dtype=np.int32)
    
    filtered_level_names = levels_names[valid_indices_np]
    filtered_level_prices = levels_prices[valid_indices_np]
    # --- CORRECCIÓN DEL ERROR AQUÍ ---
    filtered_level_dirs = levels_directions[valid_indices_np]
    filtered_level_ranges = level_ranges[valid_indices_np]
    num_levels = len(filtered_level_prices)

    fsm_states = np.full(num_levels, IDLE, dtype=np.int8)
    break_candle_indices = np.full(num_levels, -1, dtype=np.int32)
    
    equity, cash = initial_capital, initial_capital
    position_size, entry_price, direction, current_sl, current_tp = 0.0, 0.0, 0, 0.0, 0.0
    active_trade_entry_time = 0.0
    
    trades_today, stop_trading_for_day = 0, False
    
    max_trades = 10000
    closed_trades = np.zeros((max_trades, 8), dtype=np.float64)
    trade_count = 0
    equity_history = np.zeros((num_candles, 2), dtype=np.float64)

    all_emas = calculate_emas_numba(close_prices, ema_periods)
    ema_s, ema_m, ema_l = all_emas[:, 0], all_emas[:, 1], all_emas[:, 2]

    for i in range(1, num_candles):
        current_hour, current_minute = hours[i], minutes[i]
        
        if direction != 0:
            is_closed, exit_reason, exit_price = False, 0, 0.0
            if direction == 1:
                if high_prices[i] >= current_tp: is_closed, exit_price, exit_reason = True, current_tp, 1
                elif close_prices[i] <= current_sl: is_closed, exit_price, exit_reason = True, current_sl, 2
            elif direction == -1:
                if low_prices[i] <= current_tp: is_closed, exit_price, exit_reason = True, current_tp, 1
                elif close_prices[i] >= current_sl: is_closed, exit_price, exit_reason = True, current_sl, 2

            if not is_closed and current_hour >= 13:
                is_closed, exit_price, exit_reason = True, close_prices[i], 3

            if is_closed and trade_count < max_trades:
                pnl_gross = (exit_price - entry_price) * position_size if direction == 1 else (entry_price - exit_price) * position_size
                pnl_net = pnl_gross - (commission_per_side * 2)
                cash += pnl_gross
                equity += pnl_net
                
                closed_trades[trade_count, :] = [active_trade_entry_time, timestamps[i], direction, position_size, entry_price, exit_price, pnl_net, exit_reason]
                trade_count += 1
                
                direction, position_size = 0, 0.0
                
                trades_today += 1
                if pnl_net > 0:
                    stop_trading_for_day = True
                elif trades_today == 1 and pnl_net <= -60.0:
                    stop_trading_for_day = True
                elif trades_today >= 2:
                    stop_trading_for_day = True
                
                fsm_states[:] = INVALIDATED

        is_trading_window = (current_hour == 9 and current_minute >= 30) or \
                            (current_hour == 10) or \
                            (current_hour == 11 and current_minute <= 30)

        if direction == 0 and not stop_trading_for_day and is_trading_window:
            body_mid_curr = (open_prices[i] + close_prices[i]) / 2.0
            body_mid_prev = (open_prices[i-1] + close_prices[i-1]) / 2.0

            for j in range(num_levels):
                if fsm_states[j] == INVALIDATED: continue

                level_p, level_dir = filtered_level_prices[j], filtered_level_dirs[j]
                
                if fsm_states[j] == IDLE:
                    is_break = (level_dir == 1 and body_mid_prev < level_p and body_mid_curr > level_p) or \
                               (level_dir == -1 and body_mid_prev > level_p and body_mid_curr < level_p)
                    if is_break:
                        fsm_states[j] = BROKEN
                        break_candle_indices[j] = i

                elif fsm_states[j] == BROKEN:
                    if i - break_candle_indices[j] > max_retest_candles:
                        fsm_states[j] = INVALIDATED
                        continue
                    
                    if (level_dir == 1 and close_prices[i] < level_p) or \
                       (level_dir == -1 and close_prices[i] > level_p):
                        fsm_states[j] = INVALIDATED
                        continue

                    retest_range = filtered_level_ranges[j]
                    retest_zone_high = level_p + retest_range
                    retest_zone_low = level_p - retest_range
                    
                    retest_detected = False
                    if level_dir == 1 and low_prices[i] <= retest_zone_high and close_prices[i] > level_p:
                        retest_detected = True
                    elif level_dir == -1 and high_prices[i] >= retest_zone_low and close_prices[i] < level_p:
                        retest_detected = True

                    if retest_detected:
                        passes_ema = False
                        if ema_filter_mode == 0: passes_ema = True
                        else:
                            s, m, l = ema_s[i], ema_m[i], ema_l[i]
                            if not (np.isnan(s) or np.isnan(m) or np.isnan(l)):
                                if level_dir == 1:
                                    strong, moderate = s > m and m > l, close_prices[i] > m and s > m
                                    if ema_filter_mode == 1: passes_ema = moderate
                                    elif ema_filter_mode == 2: passes_ema = strong
                                elif level_dir == -1:
                                    strong, moderate = s < m and m < l, close_prices[i] < m and s < m
                                    if ema_filter_mode == 1: passes_ema = moderate
                                    elif ema_filter_mode == 2: passes_ema = strong
                        
                        if not passes_ema:
                            fsm_states[j] = INVALIDATED
                            continue

                        direction, entry_price = level_dir, close_prices[i]
                        
                        current_sl = calculate_sl_numba(
                            direction, break_candle_indices[j], int(filtered_level_names[j]),
                            low_prices, high_prices, sl_default_lookback, sl_reduced_lookback,
                            orh_price, orl_price
                        )
                        
                        risk_dist = abs(entry_price - current_sl)
                        if risk_dist < 1e-9:
                            direction = 0
                            continue
                        
                        current_tp = entry_price + (risk_dist * risk_reward_ratio) if direction == 1 else entry_price - (risk_dist * risk_reward_ratio)
                        
                        position_size = np.floor((equity * leverage) / entry_price)
                        
                        if position_size > 0:
                            cash -= commission_per_side
                            active_trade_entry_time = timestamps[i]
                            fsm_states[:] = INVALIDATED
                            break
                        else:
                            direction = 0
        
        unrealized_pnl = 0.0
        if direction != 0:
            unrealized_pnl = (close_prices[i] - entry_price) * position_size if direction == 1 else (entry_price - close_prices[i]) * position_size
        
        equity = cash + unrealized_pnl + (position_size * entry_price if direction != 0 else 0)
        equity_history[i, 0], equity_history[i, 1] = timestamps[i], equity
    
    return closed_trades[:trade_count], equity_history
