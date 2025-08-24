# strategies/opening_br_strategy.py
import pandas as pd
import logging
import datetime
import pytz
import numbers
import numpy as np

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
        self.risk_reward_ratio = self.params.get('risk_reward_ratio', 2.0)

        self.level_fsms = {}
        self.last_levels_processed = None

        self.last_ema_values = {}
        self.ema_alphas = {p: 2 / (p + 1) for p in self.ema_periods if isinstance(p, int) and p > 0}
        self.last_processed_timestamp = None
        
        logger.debug("Estrategia OBR (FSM) inicializada.")

    def reset(self):
        """ Resetea el estado completo de la estrategia. """
        self.level_fsms = {}
        self.last_levels_processed = None
        self.last_ema_values = {}
        self.last_processed_timestamp = None
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
        
        # --- CORRECCIÓN: Usar el índice diario si está disponible ---
        current_candle_idx = daily_candle_index if daily_candle_index != -1 else len(data) - 1
        
        for level_name, fsm in self.level_fsms.items():
            if fsm.state not in [State.SIGNAL_EMITTED, State.INVALIDATED]:
                signal_info = fsm.process_candle(current_candle, previous_candle, current_candle_idx)
                if signal_info:
                    return self._process_fsm_signal(signal_info, level_name, data, ema_values)
        return 'HOLD'

    def _process_fsm_signal(self, signal_info: dict, level_name: str, data: pd.DataFrame, ema_values: dict) -> str | dict:
        direction = self.level_fsms[level_name].direction
        signal_type = 'BUY' if direction == 'up' else 'SELL'

        if not self._passes_ema_filter(signal_type, signal_info['retest_price'], ema_values):
            logger.info(f"Señal de {level_name} FILTRADA por condiciones de EMA.")
            self.level_fsms[level_name].state = State.INVALIDATED
            return 'HOLD'

        sl_price = self._calculate_sl(signal_type, signal_info['break_info'], data)
        if pd.isna(sl_price):
            logger.warning(f"No se pudo calcular SL para la señal de {level_name}. Invalidando.")
            self.level_fsms[level_name].state = State.INVALIDATED
            return 'HOLD'
            
        entry_price = signal_info['retest_price']
        risk_distance = abs(entry_price - sl_price)
        tp_price = entry_price + (risk_distance * self.risk_reward_ratio) if signal_type == 'BUY' else entry_price - (risk_distance * self.risk_reward_ratio)

        final_signal = {'type': signal_type, 'sl_price': float(sl_price), 'tp1_price': float(tp_price), 'level': level_name, 'emas': ema_values}
        logger.info(f"SEÑAL FINAL GENERADA por FSM-{level_name}: {final_signal}")
        
        for fsm in self.level_fsms.values(): fsm.state = State.SIGNAL_EMITTED
        return final_signal

    def _calculate_sl(self, signal_type: str, break_info: dict, data: pd.DataFrame) -> float:
        break_idx = break_info['break_candle_index']
        if self.sl_method == 'LOOKBACK_MIN_MAX':
            start_idx = max(0, break_idx - (self.sl_default_lookback_candles - 1))
            lookback_candles = data.iloc[start_idx : break_idx + 1]
            return float(lookback_candles['low'].min()) if signal_type == 'BUY' else float(lookback_candles['high'].max())
        elif self.sl_method == 'BREAK_LOW_HIGH':
            return break_info['break_candle_low'] if signal_type == 'BUY' else break_info['break_candle_high']
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
