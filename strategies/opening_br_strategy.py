# strategies/opening_br_strategy.py
import pandas as pd
import logging
import numpy as np
import numbers

from .base_strategy import BaseStrategy
from .level_fsm import LevelFSM, State # Importamos la FSM

logger = logging.getLogger(__name__)

class OpeningBreakRetestStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)
        self.fsm_config = {
            'max_retest_candles': self.params.get('max_retest_candles', 15),
            'level_ranges': self.params.get('level_ranges', {}),
        }
        self.ema_periods = self.params.get('ema_periods', [9, 21, 50])
        self.ema_filter_mode = self.params.get('ema_filter_mode', 'Desactivado').capitalize()
        self.use_ema_filter = self.ema_filter_mode != 'Desactivado'
        
        self.sl_method = self.params.get('sl_method', 'LOOKBACK_MIN_MAX').upper()
        self.sl_default_lookback_candles = max(1, int(self.params.get('sl_lookback', 2)) + 1)
        self.sl_reduced_lookback_candles = 2
        self.risk_reward_ratio = self.params.get('risk_reward_ratio', 2.0)

        self.level_fsms = {}
        self.last_levels_processed = None

        self.last_ema_values = {}
        self.ema_alphas = {p: 2 / (p + 1) for p in self.ema_periods if isinstance(p, int) and p > 0}
        self.last_processed_timestamp = None
        
        logger.debug("Estrategia OBR (FSM) inicializada con lógica de SL/TP detallada.")

    def reset(self):
        """ Resetea el estado completo de la estrategia, incluyendo las FSMs y las EMAs. """
        self.level_fsms = {}
        self.last_levels_processed = None
        # --- INICIO DE LA CORRECCIÓN ---
        # Se resetea el estado de las EMAs para asegurar que cada día comience limpio.
        self.last_ema_values = {}
        self.last_processed_timestamp = None
        # --- FIN DE LA CORRECCIÓN ---
        logger.debug("Estrategia OBR (FSM) reseteada.")

    def _initialize_fsms(self, current_day_levels: dict):
        if current_day_levels == self.last_levels_processed:
            return

        logger.info(f"Inicializando FSMs para nuevos niveles: {current_day_levels}")
        self.level_fsms = {}
        
        orh, orl = current_day_levels.get('ORH'), current_day_levels.get('ORL')
        high_levels, low_levels = ['ORH', 'PMH', 'PDH'], ['ORL', 'PML', 'PDL']

        for name, price in current_day_levels.items():
            if price is None or pd.isna(price): continue

            if name in ['PMH', 'PML', 'PDH', 'PDL']:
                if orh is not None and orl is not None and orl < price < orh:
                    logger.info(f"Ignorando nivel {name} ({price:.2f}) porque está dentro del rango de apertura (ORL: {orl:.2f}, ORH: {orh:.2f}).")
                    continue

            direction = 'up' if name in high_levels else 'down' if name in low_levels else None
            if direction:
                self.level_fsms[name] = LevelFSM(level_name=name, level_price=price, direction=direction, config=self.fsm_config)
        
        self.last_levels_processed = current_day_levels.copy()
        logger.info(f"FSMs activas después del filtrado: {list(self.level_fsms.keys())}")

    def get_signal(self, data: pd.DataFrame, current_day_levels: dict | None = None, daily_candle_index: int = -1) -> str | dict:
        if len(data) < 2: return 'HOLD'
        if current_day_levels: self._initialize_fsms(current_day_levels)

        ema_values = self._update_emas(data)
        current_candle, previous_candle = data.iloc[-1], data.iloc[-2]
        
        current_candle_idx = daily_candle_index if daily_candle_index != -1 else len(data) - 1
        
        for level_name, fsm in self.level_fsms.items():
            if fsm.state not in [State.SIGNAL_EMITTED, State.INVALIDATED]:
                signal_info = fsm.process_candle(current_candle, previous_candle, current_candle_idx)
                if signal_info:
                    return self._process_fsm_signal(signal_info, level_name, data, ema_values, current_day_levels)
        return 'HOLD'

    def _process_fsm_signal(self, signal_info: dict, level_name: str, data: pd.DataFrame, ema_values: dict, current_day_levels: dict) -> str | dict:
        direction = self.level_fsms[level_name].direction
        signal_type = 'BUY' if direction == 'up' else 'SELL'

        if not self._passes_ema_filter(signal_type, signal_info['retest_price'], ema_values):
            logger.info(f"Señal de {level_name} FILTRADA por condiciones de EMA.")
            self.level_fsms[level_name].state = State.INVALIDATED
            return 'HOLD'

        sl_price = self._calculate_sl(signal_type, signal_info['break_info'], level_name, data, current_day_levels)
        if pd.isna(sl_price):
            logger.warning(f"No se pudo calcular SL para la señal de {level_name}. Invalidando.")
            self.level_fsms[level_name].state = State.INVALIDATED
            return 'HOLD'
            
        entry_price = signal_info['retest_price']
        risk_distance = abs(entry_price - sl_price)
        if risk_distance < 1e-9:
             logger.warning(f"Distancia de riesgo es cero para {level_name}. Invalidando señal.")
             self.level_fsms[level_name].state = State.INVALIDATED
             return 'HOLD'

        tp_price = entry_price + (risk_distance * self.risk_reward_ratio) if signal_type == 'BUY' else entry_price - (risk_distance * self.risk_reward_ratio)

        final_signal = {'type': signal_type, 'sl_price': float(sl_price), 'tp1_price': float(tp_price), 'level': level_name, 'emas': ema_values}
        logger.info(f"SEÑAL FINAL GENERADA por FSM-{level_name}: {final_signal}")
        
        for fsm in self.level_fsms.values(): fsm.state = State.SIGNAL_EMITTED
        return final_signal

    def _calculate_sl(self, signal_type: str, break_info: dict, level_name: str, data: pd.DataFrame, current_day_all_levels: dict) -> float:
        try:
            sl_price = np.nan
            break_time = break_info.get('break_time')
            
            if break_time is None:
                logger.error("OBR Signal: No se encontró 'break_time' en break_info. No se puede calcular SL.")
                return np.nan
            
            try:
                break_idx_global = data.index.get_loc(break_time)
            except KeyError:
                logger.error(f"OBR Signal: No se pudo encontrar el timestamp de ruptura {break_time} en el índice del historial de datos.")
                return np.nan

            sl_method_to_use = self.sl_method

            if sl_method_to_use == 'LOOKBACK_MIN_MAX':
                start_idx_default = max(0, break_idx_global - (self.sl_default_lookback_candles - 1))
                end_idx = break_idx_global + 1
                candles_default = data.iloc[start_idx_default:end_idx]
                sl_price_default_lookback = np.nan

                if not candles_default.empty:
                    lows = pd.to_numeric(candles_default['low'], errors='coerce')
                    highs = pd.to_numeric(candles_default['high'], errors='coerce')
                    if signal_type == 'BUY':
                        sl_price_default_lookback = lows.min()
                    else:
                        sl_price_default_lookback = highs.max()
                else:
                    logger.warning(f"OBR SL LOOKBACK: No hay velas en rango [{start_idx_default}:{end_idx}].")

                if pd.notna(sl_price_default_lookback):
                    sl_price = sl_price_default_lookback
                else:
                    logger.warning(f"OBR SL LOOKBACK: SL calculado es NaN. Fallback a BREAK_LOW_HIGH.")
                    sl_method_to_use = 'BREAK_LOW_HIGH'

                if sl_method_to_use == 'LOOKBACK_MIN_MAX' and pd.notna(sl_price) and level_name in ['ORH', 'ORL']:
                    orh_val_levels = current_day_all_levels.get('ORH')
                    orl_val_levels = current_day_all_levels.get('ORL')
                    
                    if (orh_val_levels is not None and orl_val_levels is not None and
                        pd.notna(orh_val_levels) and pd.notna(orl_val_levels) and
                        isinstance(orh_val_levels, numbers.Number) and isinstance(orl_val_levels, numbers.Number) and
                        orh_val_levels > orl_val_levels):
                        
                        or_midpoint = (float(orh_val_levels) + float(orl_val_levels)) / 2.0
                        reduce_lookback = False
                        
                        if signal_type == 'BUY' and sl_price < or_midpoint:
                            reduce_lookback = True
                            logger.info(f"OBR SL Adjust (BUY): SL default {sl_price:.4f} < OR Mid {or_midpoint:.4f}.")
                        elif signal_type == 'SELL' and sl_price > or_midpoint:
                            reduce_lookback = True
                            logger.info(f"OBR SL Adjust (SELL): SL default {sl_price:.4f} > OR Mid {or_midpoint:.4f}.")

                        if reduce_lookback:
                            start_idx_reduced = max(0, break_idx_global - (self.sl_reduced_lookback_candles - 1))
                            candles_reduced = data.iloc[start_idx_reduced:end_idx]
                            sl_price_reduced_lookback = np.nan
                            
                            if not candles_reduced.empty:
                                lows_r = pd.to_numeric(candles_reduced['low'], errors='coerce')
                                highs_r = pd.to_numeric(candles_reduced['high'], errors='coerce')
                                if signal_type == 'BUY':
                                    sl_price_reduced_lookback = lows_r.min()
                                else:
                                    sl_price_reduced_lookback = highs_r.max()
                                
                                if pd.notna(sl_price_reduced_lookback):
                                    sl_price = sl_price_reduced_lookback
                                    logger.info(f"OBR SL Adjust: Nuevo SL ({self.sl_reduced_lookback_candles} velas): {sl_price:.4f}")
                                else:
                                    logger.warning(f"OBR SL Adjust: Falló cálculo reducido. Mantiene SL default: {sl_price:.4f}")
                            else:
                                logger.warning(f"OBR SL Adjust: No hay velas para lookback reducido [{start_idx_reduced}:{end_idx}]. Mantiene SL default: {sl_price:.4f}")

            if sl_method_to_use == 'BREAK_LOW_HIGH':
                if signal_type == 'BUY':
                    sl_price = float(break_info.get('break_candle_low', np.nan))
                else:
                    sl_price = float(break_info.get('break_candle_high', np.nan))

            if pd.isna(sl_price):
                logger.error(f"OBR Signal: SL final es NaN para {level_name} ({sl_method_to_use}).")
                return np.nan
            
            return float(sl_price)

        except Exception as e:
            logger.error(f"Error fatal calculando SL para {level_name}: {e}", exc_info=True)
            return np.nan

    def _passes_ema_filter(self, signal_type: str, price: float, ema_values: dict) -> bool:
        if not self.use_ema_filter: return True
        try:
            ema_s, ema_m, ema_l = ema_values[f'EMA_{self.ema_periods[0]}'], ema_values[f'EMA_{self.ema_periods[1]}'], ema_values[f'EMA_{self.ema_periods[2]}']
            if pd.isna(ema_s) or pd.isna(ema_m) or pd.isna(ema_l): return False
        except (KeyError, IndexError): return False

        if signal_type == 'BUY':
            strong, moderate = ema_s > ema_m and ema_m > ema_l, price > ema_m and ema_s > ema_m
            if self.ema_filter_mode == 'Fuerte': return strong
            if self.ema_filter_mode == 'Moderado': return moderate
        elif signal_type == 'SELL':
            strong, moderate = ema_s < ema_m and ema_m < ema_l, price < ema_m and ema_s < ema_m
            if self.ema_filter_mode == 'Fuerte': return strong
            if self.ema_filter_mode == 'Moderado': return moderate
        return False

    def _update_emas(self, data: pd.DataFrame) -> dict:
        current_candle = data.iloc[-1]
        current_timestamp, current_close = current_candle.name, pd.to_numeric(current_candle['close'], errors='coerce')
        if self.last_processed_timestamp == current_timestamp: return self.last_ema_values.copy()
        if not self.last_ema_values or self.last_processed_timestamp is None: self._initialize_emas(data)
        else: self._update_emas_incrementally(current_close)
        self.last_processed_timestamp = current_timestamp
        return self.last_ema_values.copy()

    def _initialize_emas(self, data: pd.DataFrame):
        close_prices = pd.to_numeric(data['close'], errors='coerce')
        for period in self.ema_periods:
            key = f'EMA_{period}'
            if len(close_prices.dropna()) >= period: self.last_ema_values[key] = close_prices.ewm(span=period, adjust=False).mean().iloc[-1]
            else: self.last_ema_values[key] = np.nan

    def _update_emas_incrementally(self, current_close: float):
        if pd.isna(current_close): return
        for period, alpha in self.ema_alphas.items():
            key = f'EMA_{period}'
            prev_ema = self.last_ema_values.get(key)
            if prev_ema is not None and not pd.isna(prev_ema): self.last_ema_values[key] = prev_ema + alpha * (current_close - prev_ema)
            else: self.last_ema_values = {}; return
