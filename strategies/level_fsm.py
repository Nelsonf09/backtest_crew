# strategies/level_fsm.py
import pandas as pd
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

class State(Enum):
    """ Define los posibles estados de un nivel de precios. """
    IDLE = auto()           # Esperando una ruptura
    BROKEN = auto()         # El nivel ha sido roto, esperando retest
    IN_RETEST_WINDOW = auto() # Se ha producido un retest, ventana de tiempo para generar señal
    SIGNAL_EMITTED = auto() # Se ha generado una señal de trading
    INVALIDATED = auto()    # La oportunidad de trading ha sido invalidada

class LevelFSM:
    """
    Gestiona el estado de un único nivel de precios (ej. ORH, PDL)
    usando una Máquina de Estados Finitos (FSM).
    """
    def __init__(self, level_name: str, level_price: float, direction: str, config: dict):
        self.level_name = level_name
        self.level_price = float(level_price)
        self.direction = direction # 'up' para niveles de resistencia, 'down' para soporte
        self.config = config
        self.state = State.IDLE
        self.break_info = {}
        self.retest_info = {}
        logger.debug(f"FSM creada para {self.level_name} @ {self.level_price:.2f}, Dirección: {self.direction}")

    def reset(self):
        """ Resetea la FSM a su estado inicial. """
        self.state = State.IDLE
        self.break_info = {}
        self.retest_info = {}

    def process_candle(self, current_candle: pd.Series, previous_candle: pd.Series, current_candle_idx: int):
        """
        Procesa una nueva vela y actualiza el estado de la FSM.
        Este es el motor principal de la máquina de estados.
        """
        # Las transiciones dependen del estado actual
        if self.state == State.IDLE:
            self._transition_from_idle(current_candle, previous_candle, current_candle_idx)
        elif self.state == State.BROKEN:
            self._transition_from_broken(current_candle, current_candle_idx)
        elif self.state == State.IN_RETEST_WINDOW:
            self._transition_from_retest_window(current_candle, current_candle_idx)
        
        # Si en algún momento se emite una señal, se retorna la información
        if self.state == State.SIGNAL_EMITTED:
            return self.retest_info
        return None

    # --- Lógica de Transiciones de Estado ---

    def _transition_from_idle(self, current_candle: pd.Series, previous_candle: pd.Series, current_candle_idx: int):
        """ Lógica para decidir si se sale del estado IDLE (buscando una ruptura). """
        try:
            o_curr, c_curr = float(current_candle['open']), float(current_candle['close'])
            o_prev, c_prev = float(previous_candle['open']), float(previous_candle['close'])
            current_body_mid = (o_curr + c_curr) / 2.0
            previous_body_mid = (o_prev + c_prev) / 2.0
        except (ValueError, TypeError):
            return

        is_break = False
        if self.direction == 'up' and previous_body_mid < self.level_price and current_body_mid > self.level_price:
            is_break = True
        elif self.direction == 'down' and previous_body_mid > self.level_price and current_body_mid < self.level_price:
            is_break = True
        
        if is_break:
            self.state = State.BROKEN
            self.break_info = {
                'break_price': float(c_curr),
                'break_time': current_candle.name,
                'break_candle_index': current_candle_idx,
                'break_candle_low': float(current_candle['low']),
                'break_candle_high': float(current_candle['high'])
            }
            logger.info(f"FSM-{self.level_name}: IDLE -> BROKEN en vela {current_candle_idx}")

    def _transition_from_broken(self, current_candle: pd.Series, current_candle_idx: int):
        """ Lógica para salir del estado BROKEN (buscando retest o invalidación). """
        break_idx = self.break_info['break_candle_index']
        candles_since_break = current_candle_idx - break_idx

        # Condición de expiración por tiempo
        if candles_since_break > self.config.get('max_retest_candles', 15):
            self.state = State.INVALIDATED
            logger.info(f"FSM-{self.level_name}: BROKEN -> INVALIDATED (Expirado por tiempo)")
            return

        # Condición de invalidación por precio
        try:
            c_curr = float(current_candle['close'])
            if self.direction == 'up' and c_curr < self.level_price:
                self.state = State.INVALIDATED
                logger.info(f"FSM-{self.level_name}: BROKEN -> INVALIDATED (Precio invalidó ruptura alcista)")
                return
            elif self.direction == 'down' and c_curr > self.level_price:
                self.state = State.INVALIDATED
                logger.info(f"FSM-{self.level_name}: BROKEN -> INVALIDATED (Precio invalidó ruptura bajista)")
                return
        except (ValueError, TypeError):
            return

        # Condición de retest
        self._check_retest_condition(current_candle, current_candle_idx)

    def _transition_from_retest_window(self, current_candle: pd.Series, current_candle_idx: int):
        """
        Lógica después de un retest. En esta implementación, si el retest es válido,
        se emite la señal inmediatamente.
        """
        # Esta estrategia emite la señal en la misma vela del retest,
        # por lo que este estado es transitorio.
        # Podría expandirse para tener una ventana de N velas post-retest.
        self.state = State.SIGNAL_EMITTED
        logger.info(f"FSM-{self.level_name}: IN_RETEST_WINDOW -> SIGNAL_EMITTED")

    def _check_retest_condition(self, current_candle: pd.Series, current_candle_idx: int):
        """ Comprueba si la vela actual cumple las condiciones de retest. """
        try:
            l_curr, h_curr, c_curr = float(current_candle['low']), float(current_candle['high']), float(current_candle['close'])
            level_range = self.config.get('level_ranges', {}).get(self.level_name, 0.2)
            retest_zone_high = self.level_price + level_range
            retest_zone_low = self.level_price - level_range
            
            retest_detected = False
            if self.direction == 'up' and l_curr <= retest_zone_high and c_curr > self.level_price:
                retest_detected = True
            elif self.direction == 'down' and h_curr >= retest_zone_low and c_curr < self.level_price:
                retest_detected = True

            if retest_detected:
                self.state = State.IN_RETEST_WINDOW
                self.retest_info = {
                    'retest_price': c_curr,
                    'retest_time': current_candle.name,
                    'retest_candle_index': current_candle_idx,
                    'break_info': self.break_info.copy() # Adjuntamos info de la ruptura
                }
                logger.info(f"FSM-{self.level_name}: BROKEN -> IN_RETEST_WINDOW (Retest detectado)")
        except (ValueError, TypeError):
            pass
