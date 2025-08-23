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
        # Parámetros de configuración que se pasarán a cada FSM
        self.fsm_config = {
            'max_retest_candles': self.params.get('max_retest_candles', 15),
            'level_ranges': self.params.get('level_ranges', {}),
        }
        self.ema_periods = self.params.get('ema_periods', [9, 21, 50])
        self.ema_filter_mode = self.params.get('ema_filter_mode', 'Desactivado').capitalize()
        self.use_ema_filter = self.ema_filter_mode != 'Desactivado'
        
        # Parámetros para la lógica de SL
        self.sl_method = self.params.get('sl_method', 'LOOKBACK_MIN_MAX').upper()
        self.sl_default_lookback_candles = max(1, int(self.params.get('sl_lookback', 2)) + 1)
        self.risk_reward_ratio = self.params.get('risk_reward_ratio', 2.0)

        # Contenedor para las instancias de las FSM
        self.level_fsms = {}
        self.last_levels_processed = None

        # Lógica de EMA incremental
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
        """ 
        Crea o resetea las FSMs para los niveles del día actual, aplicando la lógica
        de ignorar niveles dentro del rango de apertura.
        """
        if current_day_levels == self.last_levels_processed:
            return

        logger.info(f"Inicializando FSMs para nuevos niveles: {current_day_levels}")
        self.level_fsms = {}
        
        orh = current_day_levels.get('ORH')
        orl = current_day_levels.get('ORL')
        
        high_levels = ['ORH', 'PMH', 'PDH']
        low_levels = ['ORL', 'PML', 'PDL']

        for name, price in current_day_levels.items():
            if price is None or pd.isna(price):
                continue

            # --- LÓGICA DE FILTRADO RESTAURADA ---
            # Si el nivel es PMH/PML o PDH/PDL, comprobar si está dentro del rango ORH/ORL.
            if name in ['PMH', 'PML', 'PDH', 'PDL']:
                if orh is not None and orl is not None:
                    if orl < price < orh:
                        logger.info(f"Ignorando nivel {name} ({price:.2f}) porque está dentro del rango de apertura (ORL: {orl:.2f}, ORH: {orh:.2f}).")
                        continue # Saltar a la siguiente iteración, no crear FSM para este nivel.
            # --- FIN DE LA LÓGICA DE FILTRADO ---

            direction = None
            if name in high_levels:
                direction = 'up'
            elif name in low_levels:
                direction = 'down'
            
            if direction:
                self.level_fsms[name] = LevelFSM(
                    level_name=name,
                    level_price=price,
                    direction=direction,
                    config=self.fsm_config
                )
        self.last_levels_processed = current_day_levels.copy()
        logger.info(f"FSMs activas después del filtrado: {list(self.level_fsms.keys())}")


    def get_signal(self, data: pd.DataFrame, current_day_levels: dict | None = None) -> str | dict:
        if len(data) < 2:
            return 'HOLD'

        if current_day_levels:
            self._initialize_fsms(current_day_levels)

        ema_values = self._update_emas(data)

        current_candle = data.iloc[-1]
        previous_candle = data.iloc[-2]
        current_candle_idx = len(data) - 1
        
        for level_name, fsm in self.level_fsms.items():
            if fsm.state not in [State.SIGNAL_EMITTED, State.INVALIDATED]:
                signal_info = fsm.process_candle(current_candle, previous_candle, current_candle_idx)

                if signal_info:
                    return self._process_fsm_signal(signal_info, level_name, data, ema_values)
        
        return 'HOLD'

    def _process_fsm_signal(self, signal_info: dict, level_name: str, data: pd.DataFrame, ema_values: dict) -> str | dict:
        """ Construye la señal de trading final a partir de la salida de la FSM. """
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
        tp_price = np.nan
        if abs(entry_price - sl_price) > 1e-9:
            risk_distance = abs(entry_price - sl_price)
            reward_distance = risk_distance * self.risk_reward_ratio
            tp_price = entry_price + reward_distance if signal_type == 'BUY' else entry_price - reward_distance

        final_signal = {
            'type': signal_type,
            'sl_price': float(sl_price),
            'tp1_price': float(tp_price) if pd.notna(tp_price) else None,
            'level': level_name,
            'emas': ema_values
        }
        logger.info(f"SEÑAL FINAL GENERADA por FSM-{level_name}: {final_signal}")
        
        for fsm in self.level_fsms.values():
            fsm.state = State.SIGNAL_EMITTED

        return final_signal

    def _calculate_sl(self, signal_type: str, break_info: dict, data: pd.DataFrame) -> float:
        """ Calcula el precio del Stop Loss basado en la configuración. """
        break_idx = break_info['break_candle_index']
        if self.sl_method == 'LOOKBACK_MIN_MAX':
            start_idx = max(0, break_idx - (self.sl_default_lookback_candles - 1))
            lookback_candles = data.iloc[start_idx : break_idx + 1]
            if signal_type == 'BUY':
                return float(pd.to_numeric(lookback_candles['low'], errors='coerce').min())
            else:
                return float(pd.to_numeric(lookback_candles['high'], errors='coerce').max())
        elif self.sl_method == 'BREAK_LOW_HIGH':
            if signal_type == 'BUY':
                return break_info['break_candle_low']
            else:
                return break_info['break_candle_high']
        return np.nan

    def _passes_ema_filter(self, signal_type: str, price: float, ema_values: dict) -> bool:
        """ Comprueba si la señal pasa el filtro de tendencia EMA. """
        if not self.use_ema_filter:
            return True

        try:
            ema_s = ema_values[f'EMA_{self.ema_periods[0]}']
            ema_m = ema_values[f'EMA_{self.ema_periods[1]}']
            ema_l = ema_values[f'EMA_{self.ema_periods[2]}']
            if pd.isna(ema_s) or pd.isna(ema_m) or pd.isna(ema_l):
                return False
        except (KeyError, IndexError):
            logger.warning("Faltan periodos de EMA para el filtro.")
            return False

        if signal_type == 'BUY':
            strong_cond = ema_s > ema_m and ema_m > ema_l
            moderate_cond = price > ema_m and ema_s > ema_m
            if self.ema_filter_mode == 'Fuerte': return strong_cond
            if self.ema_filter_mode == 'Moderado': return moderate_cond
            if self.ema_filter_mode == 'Cualquiera': return strong_cond or moderate_cond
        elif signal_type == 'SELL':
            strong_cond = ema_s < ema_m and ema_m < ema_l
            moderate_cond = price < ema_m and ema_s < ema_m
            if self.ema_filter_mode == 'Fuerte': return strong_cond
            if self.ema_filter_mode == 'Moderado': return moderate_cond
            if self.ema_filter_mode == 'Cualquiera': return strong_cond or moderate_cond
        
        return False

    def _update_emas(self, data: pd.DataFrame) -> dict:
        """ Gestiona la actualización incremental o la inicialización de las EMAs. """
        current_candle = data.iloc[-1]
        current_timestamp = current_candle.name
        current_close = pd.to_numeric(current_candle['close'], errors='coerce')
        
        if self.last_processed_timestamp == current_timestamp:
            return self.last_ema_values.copy()

        if not self.last_ema_values or self.last_processed_timestamp is None:
            self._initialize_emas(data)
        else:
            self._update_emas_incrementally(current_close)

        self.last_processed_timestamp = current_timestamp
        return self.last_ema_values.copy()

    def _initialize_emas(self, data: pd.DataFrame):
        close_prices = pd.to_numeric(data['close'], errors='coerce')
        for period in self.ema_periods:
            key = f'EMA_{period}'
            if len(close_prices.dropna()) >= period:
                self.last_ema_values[key] = close_prices.ewm(span=period, adjust=False).mean().iloc[-1]
            else:
                self.last_ema_values[key] = np.nan

    def _update_emas_incrementally(self, current_close: float):
        if pd.isna(current_close):
            return
        for period, alpha in self.ema_alphas.items():
            key = f'EMA_{period}'
            prev_ema = self.last_ema_values.get(key)
            if prev_ema is not None and not pd.isna(prev_ema):
                self.last_ema_values[key] = prev_ema + alpha * (current_close - prev_ema)
            else:
                self.last_ema_values = {} 
                return
