# agent_core/execution.py
"""
Simulador de Ejecución y Gestión de Cuenta refactorizado con una FSM para el ciclo de vida de las operaciones.
Este módulo simula una cuenta de trading, gestiona el capital, las comisiones, el apalancamiento
y el ciclo de vida completo de cada operación individual a través de una FSM.
"""
import pandas as pd
import logging
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import pytz
import math
import uuid

# Importaciones de módulos compartidos
from shared.timezone_handler import ensure_timezone_utc
from shared.fsm import FSM, TradeState

logger = logging.getLogger(__name__)

# --- Constantes de Configuración ---
TP_PRIORITY_OVER_SL = True  # Si SL y TP se activan en la misma vela, el TP tiene prioridad.
PRECISION = Decimal('0.0001') # Precisión para los cálculos de precios.

def quantize(value):
    """Función de ayuda para convertir valores a Decimal con una precisión fija."""
    try:
        dec_value = Decimal(str(value))
        if dec_value.is_nan() or dec_value.is_infinite(): return Decimal('NaN')
    except (TypeError, ValueError, InvalidOperation):
        return Decimal('NaN')
    return dec_value.quantize(PRECISION, rounding=ROUND_HALF_UP)

class Trade:
    """
    Representa una única operación de trading con su propio ciclo de vida gestionado por una FSM.
    Cada instancia de esta clase es una operación independiente.
    """
    def __init__(self, direction: str, entry_time, entry_price: Decimal, size: int,
                 sl_price: Decimal, tp_price: Decimal, entry_emas: dict, level_triggered: str | None):
        self.id = uuid.uuid4()
        self.fsm = FSM(TradeState.ACTIVE)
        self.direction = direction
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.size = size
        self.initial_sl_price = sl_price
        self.initial_tp_price = tp_price
        self.current_sl_price = sl_price
        self.current_tp_price = tp_price
        self.entry_emas = entry_emas
        self.level_triggered = level_triggered
        self.candles_held = 0
        
        # Atributos que se rellenan al cerrar la operación
        self.exit_time = None
        self.exit_price = Decimal('NaN')
        self.exit_reason = None
        self.pnl_gross = Decimal('NaN')
        self.pnl_net = Decimal('NaN')
        self.commission = Decimal('NaN')

    def update_on_candle(self, candle: pd.Series):
        """Actualiza el estado de la operación con cada nueva vela."""
        if self.fsm.is_in_state(TradeState.ACTIVE):
            self.candles_held += 1

    def close(self, exit_time, exit_price: Decimal, exit_reason: str, commission_total: Decimal):
        """Cierra la operación, calcula el PNL y transiciona el estado a CLOSED."""
        if not self.fsm.is_in_state(TradeState.ACTIVE):
            logger.warning(f"Intento de cerrar un trade que no está activo (Estado: {self.fsm.state}). Trade ID: {self.id}")
            return

        self.fsm.transition_to(TradeState.CLOSED)
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.commission = commission_total

        if self.direction == 'LONG':
            self.pnl_gross = (self.exit_price - self.entry_price) * self.size
        else:  # SHORT
            self.pnl_gross = (self.entry_price - self.exit_price) * self.size
        
        self.pnl_net = self.pnl_gross - self.commission
        logger.info(f"Trade {self.id} CERRADO. Razón: {self.exit_reason}, PNL Neto: {self.pnl_net:.2f}")

    def to_dict(self) -> dict:
        """Convierte la información del trade a un diccionario para los registros y la UI."""
        return {
            "entry_time": self.entry_time, "exit_time": self.exit_time, "direction": self.direction,
            "size": self.size, "entry_price": float(self.entry_price) if not self.entry_price.is_nan() else None,
            "exit_price": float(self.exit_price) if not self.exit_price.is_nan() else None,
            "pnl_gross": float(self.pnl_gross) if not self.pnl_gross.is_nan() else None,
            "commission": float(self.commission) if not self.commission.is_nan() else None,
            "pnl_net": float(self.pnl_net) if not self.pnl_net.is_nan() else None,
            "exit_reason": self.exit_reason, "entry_emas": self.entry_emas,
            "sl_at_entry": float(self.initial_sl_price) if not self.initial_sl_price.is_nan() else None,
            "tp_at_entry": float(self.initial_tp_price) if not self.initial_tp_price.is_nan() else None,
            "level_triggered": self.level_triggered
        }

class ExecutionSimulator:
    """
    Simula una cuenta de trading, procesando señales y gestionando operaciones
    a través de su ciclo de vida FSM.
    """
    def __init__(self, initial_capital: float = 10000.0,
                 commission_per_trade: float = 1.0,
                 slippage_points: float = 0.01,
                 leverage: int = 1,
                 max_holding_candles: int | None = None):

        self.initial_capital = quantize(initial_capital)
        self.commission_per_side = quantize(commission_per_trade)
        self.slippage_points = quantize(slippage_points)
        self.leverage = max(1, int(leverage))
        self.max_holding_candles = max_holding_candles

        self.cash = self.initial_capital
        self.current_trade: Trade | None = None
        self.last_equity = self.initial_capital
        self.closed_pnl_total = Decimal('0.0')

        self.closed_trades = []
        self.equity_history = []
        
        self.account_fsm = FSM(TradeState.FLAT)

        logger.info(f"ExecutionSimulator (FSM) inicializado: Capital={self.initial_capital:.2f}, Leverage={self.leverage}:1")

    def set_leverage(self, new_leverage: int):
        """Permite actualizar el apalancamiento dinámicamente."""
        old_leverage = self.leverage
        self.leverage = max(1, int(new_leverage))
        logger.info(f"Apalancamiento actualizado de {old_leverage}:1 a {self.leverage}:1.")

    def _get_execution_price(self, target_price: float, signal_type_for_slippage: str) -> Decimal:
        price_dec = quantize(target_price)
        if price_dec.is_nan(): return price_dec
        if self.slippage_points > Decimal('0'):
            if signal_type_for_slippage in ['BUY', 'COVER_SHORT']:
                return price_dec + self.slippage_points
            elif signal_type_for_slippage in ['SELL', 'SHORT']:
                return max(Decimal('0.0'), price_dec - self.slippage_points)
        return price_dec

    def _open_position(self, signal_data: dict, candle: pd.Series) -> dict | None:
        timestamp = candle.name
        close_price_float = float(candle['close'])
        signal_type = signal_data.get('type', 'HOLD').upper()

        current_equity = self.get_equity(close_price_float)
        if current_equity.is_nan() or current_equity <= Decimal('0'):
            logger.warning("No se puede abrir posición: Equity inválido o cero.")
            return None

        entry_exec_price = self._get_execution_price(close_price_float, signal_type)
        if entry_exec_price.is_nan() or entry_exec_price <= Decimal('0.0'):
            logger.warning(f"Precio de entrada inválido: {entry_exec_price}. No se abre posición.")
            return None
            
        buying_power = current_equity * Decimal(self.leverage)
        calculated_size = math.floor(buying_power / entry_exec_price)

        if calculated_size <= 0:
            logger.info("Tamaño de posición calculado es 0. No se abre trade.")
            return None

        if self.cash < self.commission_per_side:
            logger.warning("Cash insuficiente para cubrir la comisión de entrada.")
            return None
        
        # --- LÓGICA DE SL/TP SIMPLIFICADA ---
        # Ahora se confía 100% en los valores que vienen del diccionario de la señal.
        sl_price = quantize(signal_data.get('sl_price'))
        tp_price = quantize(signal_data.get('tp1_price'))

        if sl_price.is_nan() or tp_price.is_nan():
            logger.error(f"Señal para abrir posición recibida sin SL/TP válidos. SL: {sl_price}, TP: {tp_price}. No se abre trade.")
            return None
        # --- FIN DE LA SIMPLIFICACIÓN ---

        self.cash -= self.commission_per_side
        self.account_fsm.transition_to(TradeState.ACTIVE)
        
        self.current_trade = Trade(
            direction='LONG' if signal_type == 'BUY' else 'SHORT',
            entry_time=timestamp,
            entry_price=entry_exec_price,
            size=calculated_size,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_emas=signal_data.get('emas', {}),
            level_triggered=signal_data.get('level')
        )
        
        logger.info(f"TRADE ABIERTO ({self.current_trade.direction}): Size={self.current_trade.size} @ {self.current_trade.entry_price:.4f}")

        marker_color = 'lime' if self.current_trade.direction == 'LONG' else 'magenta'
        marker_shape = 'arrowUp' if self.current_trade.direction == 'LONG' else 'arrowDown'
        marker_position = 'belowBar' if self.current_trade.direction == 'LONG' else 'aboveBar'
        marker_text = f"{self.current_trade.direction} {calculated_size} ({self.current_trade.level_triggered or ''})".strip()
        return {'marker': {'time': timestamp, 'position': marker_position, 'shape': marker_shape, 'color': marker_color, 'text': marker_text}}

    def _close_current_position(self, timestamp, exit_exec_price: Decimal, exit_reason: str) -> dict | None:
        if not self.current_trade or self.account_fsm.is_in_state(TradeState.FLAT):
            return None

        total_commission = self.commission_per_side * 2
        self.current_trade.close(timestamp, exit_exec_price, exit_reason, total_commission)
        
        self.closed_trades.append(self.current_trade.to_dict())
        self.closed_pnl_total += self.current_trade.pnl_net
        
        pnl_bruto = self.current_trade.pnl_gross
        self.cash = self.cash + pnl_bruto - self.commission_per_side

        pnl_net_display = f"{self.current_trade.pnl_net:.2f}"
        marker_color = "red" if self.current_trade.pnl_net < Decimal('0') else "green"
        if exit_reason.startswith("SL"): marker_color = "red"
        elif exit_reason.startswith("TP"): marker_color = "blue"
        elif exit_reason.startswith("Timeout"): marker_color = "orange"

        marker_shape = "arrowDown" if self.current_trade.direction == 'LONG' else "arrowUp"
        marker_position = "aboveBar" if self.current_trade.direction == 'LONG' else "belowBar"
        marker_text = f'{exit_reason} ({pnl_net_display})'
        
        result_marker = {
            'marker': {'time': timestamp, 'position': marker_position, 'shape': marker_shape, 'color': marker_color, 'text': marker_text},
            'trade': self.closed_trades[-1].copy()
        }

        self.current_trade = None
        self.account_fsm.transition_to(TradeState.FLAT)
        
        return result_marker

    def process_signal(self, signal, candle: pd.Series) -> dict | None:
        timestamp = candle.name
        try:
            close_price_dec = quantize(candle['close'])
            high_price_dec = quantize(candle['high'])
            low_price_dec = quantize(candle['low'])
            if any(p.is_nan() for p in [close_price_dec, high_price_dec, low_price_dec]):
                self._update_equity_history(float(close_price_dec) if not close_price_dec.is_nan() else None, timestamp)
                return None
            close_price_float = float(close_price_dec)
        except (KeyError, TypeError, ValueError, InvalidOperation) as e:
            logger.error(f"Error procesando precios de vela {timestamp}: {e}")
            self._update_equity_history(None, timestamp)
            return None

        if self.account_fsm.is_in_state(TradeState.ACTIVE) and self.current_trade:
            self.current_trade.update_on_candle(candle)
            
            exit_reason, exit_price_trigger = self._check_sl_tp(candle)
            if exit_reason:
                exit_signal_type = 'SELL' if self.current_trade.direction == 'LONG' else 'BUY'
                exit_exec_price = self._get_execution_price(float(exit_price_trigger), exit_signal_type)
                if not exit_exec_price.is_nan():
                    result = self._close_current_position(timestamp, exit_exec_price, exit_reason)
                    self._update_equity_history(close_price_float, timestamp)
                    return result

            if self.max_holding_candles and self.current_trade.candles_held >= self.max_holding_candles:
                exit_signal_type = 'SELL' if self.current_trade.direction == 'LONG' else 'BUY'
                exit_exec_price = self._get_execution_price(close_price_float, exit_signal_type)
                if not exit_exec_price.is_nan():
                    result = self._close_current_position(timestamp, exit_exec_price, "Timeout")
                    self._update_equity_history(close_price_float, timestamp)
                    return result

            signal_type = signal if isinstance(signal, str) else signal.get('type', 'HOLD')
            if (self.current_trade.direction == 'LONG' and signal_type == 'SELL') or \
               (self.current_trade.direction == 'SHORT' and signal_type == 'BUY'):
                exit_signal_type = 'SELL' if self.current_trade.direction == 'LONG' else 'BUY'
                exit_exec_price = self._get_execution_price(close_price_float, exit_signal_type)
                if not exit_exec_price.is_nan():
                    result = self._close_current_position(timestamp, exit_exec_price, f"Signal_{signal_type}")
                    self._update_equity_history(close_price_float, timestamp)
                    return result

        if self.account_fsm.is_in_state(TradeState.FLAT) and isinstance(signal, dict):
            signal_type = signal.get('type', 'HOLD').upper()
            if signal_type in ['BUY', 'SELL']:
                result = self._open_position(signal, candle)
                self._update_equity_history(close_price_float, timestamp)
                return result

        self._update_equity_history(close_price_float, timestamp)
        return None

    def _check_sl_tp(self, candle: pd.Series) -> tuple[str | None, Decimal | None]:
        if not self.current_trade: return None, None
        
        sl = self.current_trade.current_sl_price
        tp = self.current_trade.current_tp_price
        if sl.is_nan() or tp.is_nan(): return None, None

        high = quantize(candle['high'])
        low = quantize(candle['low'])
        close = quantize(candle['close'])

        is_tp_hit, is_sl_hit = False, False
        
        if self.current_trade.direction == 'LONG':
            if high >= tp: is_tp_hit = True
            if close <= sl: is_sl_hit = True
        elif self.current_trade.direction == 'SHORT':
            if low <= tp: is_tp_hit = True
            if close >= sl: is_sl_hit = True
        
        if is_tp_hit and is_sl_hit:
            return ("TP", tp) if TP_PRIORITY_OVER_SL else ("SL", sl)
        if is_tp_hit: return "TP", tp
        if is_sl_hit: return "SL", sl
        
        return None, None

    def _update_equity_history(self, current_close_price_float: float | None, timestamp):
        current_equity_val = self.get_equity(current_close_price_float)
        self.last_equity = current_equity_val
        timestamp_utc = ensure_timezone_utc(timestamp)
        if timestamp_utc:
            new_point = (timestamp_utc, self.last_equity)
            if not self.equity_history or self.equity_history[-1][0] < timestamp_utc:
                self.equity_history.append(new_point)
            elif self.equity_history[-1][0] == timestamp_utc:
                self.equity_history[-1] = new_point

    def get_equity(self, current_close_price: float | None = None) -> Decimal:
        unrealized_pnl = Decimal('0.0')
        if self.account_fsm.is_in_state(TradeState.ACTIVE) and self.current_trade:
            if current_close_price is not None:
                current_price_dec = quantize(current_close_price)
                if not current_price_dec.is_nan():
                    if self.current_trade.direction == 'LONG':
                        unrealized_pnl = (current_price_dec - self.current_trade.entry_price) * self.current_trade.size
                    else: # SHORT
                        unrealized_pnl = (self.current_trade.entry_price - current_price_dec) * self.current_trade.size
        
        current_equity = self.initial_capital + self.closed_pnl_total + unrealized_pnl
        return quantize(current_equity)

    def get_closed_trades(self) -> list:
        return self.closed_trades.copy()
    
    def get_equity_history(self) -> list:
        return self.equity_history.copy()
