# agent_core/execution.py
"""
Simulador de Ejecución y Gestión de Cuenta con Apalancamiento
y Logs de Depuración para SL/TP.
Modificado: SL se activa únicamente por el precio de CIERRE de la vela.
"""
import pandas as pd
import logging
from decimal import Decimal, ROUND_HALF_UP, getcontext, InvalidOperation
import pytz
import math

# <--- CAMBIO CLAVE: Actualizamos la ruta de importación
from shared.timezone_handler import ensure_timezone_utc

logger = logging.getLogger(__name__)

# CHECK_SL_WITH_CLOSE ahora está efectivamente forzado a True para la lógica de SL.
# Se mantiene por si se quiere reintroducir la opción para SL, o para TP si se cambia su lógica.
CHECK_SL_WITH_CLOSE = True
TP_PRIORITY_OVER_SL = True # Si SL y TP se tocan en la misma vela

PRECISION = Decimal('0.0001')
NUM_DECIMAL_PLACES = abs(PRECISION.as_tuple().exponent)

def quantize(value):
    try:
        dec_value = Decimal(str(value))
        if dec_value.is_nan() or dec_value.is_infinite():
            return Decimal('NaN')
    except (TypeError, ValueError, InvalidOperation):
        if isinstance(value, Decimal) and value.is_nan():
            return value
        return Decimal('NaN')
    return dec_value.quantize(PRECISION, rounding=ROUND_HALF_UP)


class ExecutionSimulator:
    def __init__(self, initial_capital: float = 10000.0,
                 commission_per_trade: float = 1.0,
                 slippage_points: float = 0.01,
                 leverage: int = 1,
                 max_holding_candles: int | None = None):

        self.initial_capital = quantize(initial_capital)
        if self.initial_capital.is_nan() or self.initial_capital <= 0:
             logger.error(f"Capital inicial inválido ({initial_capital}). Usando 10000.0.")
             self.initial_capital = quantize(10000.0)

        self.commission_per_side = quantize(commission_per_trade)
        if self.commission_per_side.is_nan() or self.commission_per_side < 0:
             logger.warning(f"Comisión inválida ({commission_per_trade}). Usando 0.0.")
             self.commission_per_side = quantize(0.0)

        self.slippage_points = quantize(slippage_points)
        if self.slippage_points.is_nan() or self.slippage_points < 0:
             logger.warning(f"Slippage inválido ({slippage_points}). Usando 0.0.")
             self.slippage_points = quantize(0.0)

        self.leverage = max(1, int(leverage))
        self.max_holding_candles = max_holding_candles
        if self.max_holding_candles is not None and self.max_holding_candles <= 0:
            logger.warning(f"max_holding_candles inválido ({self.max_holding_candles}). Se desactivará.")
            self.max_holding_candles = None

        self.cash = self.initial_capital
        self.position = 'FLAT'
        self.current_size = 0
        self.entry_price = Decimal('NaN')
        self.entry_time = None
        self.last_equity = self.initial_capital
        self.closed_pnl_total = Decimal('0.0')

        self.sl_price = Decimal('NaN')
        self.tp_price = Decimal('NaN')
        self.has_sl_tp = False
        
        self.current_trade_entry_emas = {}
        self.initial_sl_price_for_trade = Decimal('NaN')
        self.initial_tp_price_for_trade = Decimal('NaN')
        self.current_trade_level_triggered = None

        self.candles_in_current_trade = 0

        self.trades = []
        self.equity_history = []

        logger.info(f"ExecutionSimulator inicializado: Capital={self.initial_capital:.2f}, "
                    f"Comisión/Lado={self.commission_per_side:.2f}, Slippage={self.slippage_points:.4f}, "
                    f"Apalancamiento={self.leverage}:1, MaxHoldingCandles={self.max_holding_candles}")

    def _get_execution_price(self, target_price: float, signal_type_for_slippage: str) -> Decimal:
        price_dec = quantize(target_price)
        if price_dec.is_nan():
            return price_dec

        exec_price = price_dec
        if not self.slippage_points.is_nan() and self.slippage_points > Decimal('0'):
            if signal_type_for_slippage in ['BUY', 'COVER_SHORT']:
                exec_price = price_dec + self.slippage_points
            elif signal_type_for_slippage in ['SELL', 'SHORT']:
                exec_price = price_dec - self.slippage_points
        
        if exec_price < Decimal('0.0'):
             exec_price = Decimal('0.0')
        return exec_price

    def _record_trade(self, entry_time, exit_time, direction, size,
                      entry_price_dec, exit_price_dec,
                      pnl_gross_dec: Decimal,
                      commission_total_dec: Decimal,
                      pnl_net_dec: Decimal,
                      exit_reason: str = "Signal",
                      entry_emas: dict | None = None,
                      sl_price_at_entry: Decimal | None = None,
                      tp_price_at_entry: Decimal | None = None,
                      level_triggered: str | None = None):
        
        try: entry_p_f = float(entry_price_dec) if not entry_price_dec.is_nan() else None
        except: entry_p_f = None
        try: exit_p_f = float(exit_price_dec) if not exit_price_dec.is_nan() else None
        except: exit_p_f = None
        
        try: pnl_gross_f = float(pnl_gross_dec) if not pnl_gross_dec.is_nan() else None
        except: pnl_gross_f = None
        try: pnl_net_f = float(pnl_net_dec) if not pnl_net_dec.is_nan() else None
        except: pnl_net_f = None
        
        try: comm_f = float(commission_total_dec) if not commission_total_dec.is_nan() else None
        except: comm_f = None
        
        trade_info = {
            "entry_time": entry_time, "exit_time": exit_time, "direction": direction,
            "size": size, "entry_price": entry_p_f, "exit_price": exit_p_f,
            "pnl_gross": pnl_gross_f,
            "commission": comm_f,
            "pnl_net": pnl_net_f,
            "exit_reason": exit_reason,
            "entry_emas": entry_emas if entry_emas is not None else {},
            "sl_at_entry": float(sl_price_at_entry) if sl_price_at_entry is not None and not sl_price_at_entry.is_nan() else None,
            "tp_at_entry": float(tp_price_at_entry) if tp_price_at_entry is not None and not tp_price_at_entry.is_nan() else None,
            "level_triggered": level_triggered
        }
        self.trades.append(trade_info)

        if pnl_net_dec is not None and not pnl_net_dec.is_nan():
            self.closed_pnl_total += pnl_net_dec

        pnl_net_display = f"{pnl_net_f:.2f}" if pnl_net_f is not None else "Error PNLNet"
        pnl_gross_display = f"{pnl_gross_f:.2f}" if pnl_gross_f is not None else "Error PNLGross"
        comm_display = f"{comm_f:.2f}" if comm_f is not None else "Error Comm"
        level_display_log = f"Nivel: {level_triggered}" if level_triggered else ""
        logger.info(f"TRADE CERRADO ({direction}) por {exit_reason}: Size={size}, "
                    f"PNL Bruto=${pnl_gross_display}, Comm=${comm_display}, PNL Neto=${pnl_net_display}, "
                    f"PNL Acum Neto=${self.closed_pnl_total:.2f} {level_display_log}")

    def process_signal(self, signal, candle: pd.Series) -> dict | None:
        timestamp = candle.name
        try:
            close_price_dec = quantize(candle['close'])
            high_price_dec = quantize(candle['high'])
            low_price_dec = quantize(candle['low'])

            if any(p.is_nan() for p in [close_price_dec, high_price_dec, low_price_dec]):
                logger.error(f"Vela {timestamp}: Precios OHLC NaN. Saltando.")
                if not close_price_dec.is_nan(): self._update_equity_history(float(close_price_dec), timestamp)
                elif self.position == 'FLAT': self._update_equity_history(None, timestamp)
                return None
            close_price_float = float(close_price_dec)
        except (KeyError, TypeError, ValueError, InvalidOperation) as e_price:
             logger.error(f"Vela {timestamp}: Error procesando precios de la vela: {e_price}. Datos vela: {candle.to_dict()}")
             self._update_equity_history(None, timestamp)
             return None

        result = None
        signal_type = 'HOLD'
        signal_data = {}

        if isinstance(signal, str): signal_type = signal.upper()
        elif isinstance(signal, dict):
            signal_type = signal.get('type', 'HOLD').upper()
            signal_data = signal
        
        if self.position != 'FLAT':
            self.candles_in_current_trade += 1
            exit_reason = None
            exit_price_for_sl_tp_trigger = None

            if self.has_sl_tp:
                if self.sl_price.is_nan() or self.tp_price.is_nan():
                    self.has_sl_tp = False
                else:
                    is_tp_hit = False
                    is_sl_hit = False
                
                    if self.position == 'LONG':
                        if high_price_dec >= self.tp_price:
                            is_tp_hit = True
                            exit_price_for_sl_tp_trigger = self.tp_price
                        
                        if close_price_dec <= self.sl_price:
                            is_sl_hit = True
                            if not is_tp_hit or not TP_PRIORITY_OVER_SL:
                                exit_price_for_sl_tp_trigger = self.sl_price
                    
                    elif self.position == 'SHORT':
                        if low_price_dec <= self.tp_price:
                            is_tp_hit = True
                            exit_price_for_sl_tp_trigger = self.tp_price

                        if close_price_dec >= self.sl_price:
                            is_sl_hit = True
                            if not is_tp_hit or not TP_PRIORITY_OVER_SL:
                                 exit_price_for_sl_tp_trigger = self.sl_price
                    
                    if is_tp_hit and is_sl_hit:
                        exit_reason = "TP" if TP_PRIORITY_OVER_SL else "SL"
                        exit_price_for_sl_tp_trigger = self.tp_price if TP_PRIORITY_OVER_SL else self.sl_price
                    elif is_tp_hit:
                        exit_reason = "TP"
                    elif is_sl_hit:
                        exit_reason = "SL"

                    if exit_reason:
                        exit_signal_type_for_slippage = 'SELL' if self.position == 'LONG' else 'COVER_SHORT'
                        actual_exit_exec_price = self._get_execution_price(float(exit_price_for_sl_tp_trigger), exit_signal_type_for_slippage)
                    
                        if not actual_exit_exec_price.is_nan():
                            result = self._close_position(
                                timestamp=timestamp,
                                exit_exec_price=actual_exit_exec_price,
                                exit_reason=exit_reason
                            )
                            self._update_equity_history(close_price_float, timestamp)
                            return result
                        else:
                            self.has_sl_tp = False
            
            if self.position != 'FLAT' and self.max_holding_candles is not None and self.candles_in_current_trade >= self.max_holding_candles:
                exit_reason = "Timeout"
                exit_signal_type_for_slippage = 'SELL' if self.position == 'LONG' else 'COVER_SHORT'
                actual_exit_exec_price = self._get_execution_price(close_price_float, exit_signal_type_for_slippage)

                if not actual_exit_exec_price.is_nan():
                    result = self._close_position(
                        timestamp=timestamp,
                        exit_exec_price=actual_exit_exec_price,
                        exit_reason=exit_reason
                    )
                    self._update_equity_history(close_price_float, timestamp)
                    return result

        if self.position == 'FLAT':
            if signal_type in ['BUY', 'SELL']:
                current_equity = self.get_equity(close_price_float)
                if not (current_equity.is_nan() or current_equity <= Decimal('0')):
                    buying_power = current_equity * Decimal(self.leverage)
                    entry_exec_price = self._get_execution_price(close_price_float, signal_type)
                    calculated_size = 0
                    if not entry_exec_price.is_nan() and entry_exec_price > Decimal('0.0'):
                        try:
                            size_decimal = buying_power / entry_exec_price
                            calculated_size = math.floor(size_decimal)
                        except (InvalidOperation, ZeroDivisionError):
                            pass

                    if calculated_size > 0:
                        if self.cash >= self.commission_per_side:
                            self.current_size = calculated_size
                            self.entry_price = entry_exec_price
                            self.entry_time = timestamp
                            self.cash -= self.commission_per_side

                            if signal_type == 'BUY': self.position = 'LONG'
                            else: self.position = 'SHORT'
                            
                            self.candles_in_current_trade = 1
                            self.current_trade_entry_emas = signal_data.get('emas', {}).copy()
                            
                            self.current_trade_level_triggered = signal_data.get('level', None)
                            
                            self.sl_price = quantize(signal_data.get('sl_price', Decimal('NaN')))
                            self.tp_price = quantize(signal_data.get('tp1_price', Decimal('NaN')))
                            
                            self.initial_sl_price_for_trade = self.sl_price
                            self.initial_tp_price_for_trade = self.tp_price
                            self.has_sl_tp = False
                            sl_tp_valid = False

                            if not self.sl_price.is_nan() and not self.tp_price.is_nan():
                                if self.position == 'LONG' and self.sl_price < self.entry_price and self.tp_price > self.entry_price:
                                    sl_tp_valid = True
                                elif self.position == 'SHORT' and self.sl_price > self.entry_price and self.tp_price < self.entry_price:
                                    sl_tp_valid = True
                            
                            if sl_tp_valid:
                                self.has_sl_tp = True
                            else:
                                self.sl_price = Decimal('NaN')
                                self.tp_price = Decimal('NaN')
                                self.has_sl_tp = False

                            marker_color = 'lime' if self.position == 'LONG' else 'magenta'
                            marker_shape = 'arrowUp' if self.position == 'LONG' else 'arrowDown'
                            marker_position = 'belowBar' if self.position == 'LONG' else 'aboveBar'
                            marker_text_entry = f'{self.position} {self.current_size}'
                            if self.current_trade_level_triggered:
                                marker_text_entry += f' ({self.current_trade_level_triggered})'

                            result = {'marker': {'time': timestamp, 'position': marker_position, 'shape': marker_shape, 'color': marker_color, 'text': marker_text_entry}}
        
        elif self.position != 'FLAT':
            exit_by_signal = False
            if self.position == 'LONG' and signal_type == 'SELL':
                exit_by_signal = True
            elif self.position == 'SHORT' and signal_type == 'BUY':
                exit_by_signal = True

            if exit_by_signal:
                exit_signal_type_for_slippage = 'SELL' if self.position == 'LONG' else 'COVER_SHORT'
                actual_exit_exec_price = self._get_execution_price(close_price_float, exit_signal_type_for_slippage)

                if not actual_exit_exec_price.is_nan():
                    result = self._close_position(
                        timestamp=timestamp,
                        exit_exec_price=actual_exit_exec_price,
                        exit_reason=f"Signal_{signal_type}"
                    )

        self._update_equity_history(close_price_float, timestamp)
        return result

    def _close_position(self, timestamp, exit_exec_price: Decimal, exit_reason: str) -> dict | None:
        if self.position == 'FLAT' or self.current_size == 0:
            return None

        commission_exit = self.commission_per_side
        total_commission_for_trade = self.commission_per_side + commission_exit

        pnl_gross_trade = Decimal('NaN')
        cash_change_on_exit = Decimal('NaN')

        if self.position == 'LONG':
            pnl_gross_trade = (exit_exec_price - self.entry_price) * self.current_size
            cash_change_on_exit = self.current_size * exit_exec_price
        
        elif self.position == 'SHORT':
            pnl_gross_trade = (self.entry_price - exit_exec_price) * self.current_size
            cash_change_on_exit = pnl_gross_trade

        pnl_net_trade = pnl_gross_trade - total_commission_for_trade

        if not cash_change_on_exit.is_nan():
            self.cash += cash_change_on_exit
            self.cash -= commission_exit
        else:
            pnl_gross_trade = Decimal('NaN')
            pnl_net_trade = Decimal('NaN')

        pnl_net_display = f"{pnl_net_trade:.2f}" if not pnl_net_trade.is_nan() else "Error PNL"
        marker_color = "red" if pnl_net_trade < Decimal('0') else "green"
        if exit_reason.startswith("SL"): marker_color = "red"
        elif exit_reason.startswith("TP"): marker_color = "blue"
        elif exit_reason.startswith("Timeout"): marker_color = "orange"

        marker_shape = "arrowDown" if self.position == 'LONG' else "arrowUp"
        marker_position = "aboveBar" if self.position == 'LONG' else "belowBar"
        
        level_triggered_for_this_trade = getattr(self, 'current_trade_level_triggered', None)

        self._record_trade(
            entry_time=self.entry_time, exit_time=timestamp, direction=self.position,
            size=self.current_size, entry_price_dec=self.entry_price, exit_price_dec=exit_exec_price,
            pnl_gross_dec=pnl_gross_trade,
            commission_total_dec=total_commission_for_trade,
            pnl_net_dec=pnl_net_trade,
            exit_reason=exit_reason,
            entry_emas=self.current_trade_entry_emas,
            sl_price_at_entry=self.initial_sl_price_for_trade,
            tp_price_at_entry=self.initial_tp_price_for_trade,
            level_triggered=level_triggered_for_this_trade
        )
        
        marker_text = f'{exit_reason} ({pnl_net_display})' if not pnl_net_trade.is_nan() else f'{exit_reason} (PNL Err)'
        result_marker = {'marker': {'time': timestamp, 'position': marker_position, 'shape': marker_shape, 'color': marker_color, 'text': marker_text}, 'trade': self.trades[-1].copy()}

        self.position = 'FLAT'
        self.current_size = 0
        self.entry_price = Decimal('NaN')
        self.entry_time = None
        self.has_sl_tp = False
        self.sl_price = Decimal('NaN')
        self.tp_price = Decimal('NaN')
        self.current_trade_entry_emas = {}
        self.initial_sl_price_for_trade = Decimal('NaN')
        self.initial_tp_price_for_trade = Decimal('NaN')
        self.candles_in_current_trade = 0
        self.current_trade_level_triggered = None

        return result_marker

    def _update_equity_history(self, current_close_price_float: float | None, timestamp):
        current_equity_val = self.get_equity(current_close_price_float)
        self.last_equity = current_equity_val

        timestamp_utc = ensure_timezone_utc(timestamp)
        if timestamp_utc:
            new_equity_point = (timestamp_utc, self.last_equity)
            if not self.equity_history:
                self.equity_history.append(new_equity_point)
            else:
                last_ts_logged, last_eq_logged = self.equity_history[-1]
                if timestamp_utc > last_ts_logged:
                    self.equity_history.append(new_equity_point)
                elif timestamp_utc == last_ts_logged and self.last_equity != last_eq_logged :
                    self.equity_history[-1] = new_equity_point

    def get_closed_trades(self) -> list:
        return [trade.copy() for trade in self.trades]

    def get_equity_history(self) -> list:
        return [(ts, val) for ts, val in self.equity_history]

    def get_equity(self, current_close_price: float | None = None) -> Decimal:
        unrealized_pnl = Decimal('0.0')
        if self.position != 'FLAT' and self.current_size > 0 and not self.entry_price.is_nan():
            if current_close_price is not None:
                current_price_dec = quantize(current_close_price)
                if not current_price_dec.is_nan():
                    if self.position == 'LONG':
                        unrealized_pnl = (current_price_dec - self.entry_price) * self.current_size
                    elif self.position == 'SHORT':
                        unrealized_pnl = (self.entry_price - current_price_dec) * self.current_size
                    
                    if unrealized_pnl.is_nan():
                        unrealized_pnl = Decimal('0.0')
                else:
                    unrealized_pnl = Decimal('0.0')
            else:
                 unrealized_pnl = Decimal('0.0')

        current_equity = self.initial_capital + self.closed_pnl_total + unrealized_pnl
        
        final_equity = quantize(current_equity)
        return final_equity if not final_equity.is_nan() else Decimal('NaN')
