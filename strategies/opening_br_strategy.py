# strategies/opening_br_strategy.py
import pandas as pd
import logging
import datetime
import pytz
import numbers
import numpy as np

logger = logging.getLogger(__name__)

# <--- CAMBIO CLAVE: Corregimos la importación para que sea relativa
from .base_strategy import BaseStrategy

class OpeningBreakRetestStrategy(BaseStrategy):
    def __init__(self, max_retest_candles=15, risk_reward_ratio=2.0,
                 sl_method='LOOKBACK_MIN_MAX', sl_lookback=2,
                 level_ranges: dict | None = None,
                 ema_periods=[9, 21, 50],
                 ignore_premarket_previousday_breaks_minutes=5,
                 market_tz_str='America/New_York',
                 ema_filter_mode='Desactivado',
                 **params):
        super().__init__(**params)
        self.max_retest_candles = max(1, int(max_retest_candles))
        self.risk_reward_ratio = float(risk_reward_ratio)
        self.sl_method = sl_method.upper()
        self.sl_default_lookback_candles = max(1, int(sl_lookback) + 1)
        self.sl_reduced_lookback_candles = 2
        self.level_ranges = level_ranges if isinstance(level_ranges, dict) else {}
        self.active_breaks = {}
        self.high_levels = ['ORH', 'PMH', 'PDH']
        self.low_levels = ['ORL', 'PML', 'PDL']
        
        self.ema_periods = ema_periods
        if not self.ema_periods or len(self.ema_periods) < 3:
            logger.warning(f"OBR_Strategy: Se esperan 3 periodos EMA para el filtro de tendencia, se recibieron {self.ema_periods}.")

        self.ignore_premarket_previousday_breaks_minutes = int(ignore_premarket_previousday_breaks_minutes)

        self.market_tz_str = market_tz_str
        try:
            self.market_tz = pytz.timezone(self.market_tz_str)
        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"OBR_Strategy: Timezone desconocida: {self.market_tz_str}. Usando America/New_York por defecto.")
            self.market_tz = pytz.timezone('America/New_York')

        self.ema_filter_mode = str(ema_filter_mode).capitalize() if ema_filter_mode else 'Desactivado'
        self.use_ema_filter = self.ema_filter_mode != 'Desactivado'

        logger.debug(f"OBR_Strategy inicializada: max_retest={self.max_retest_candles}, R:R={self.risk_reward_ratio}, "
                     f"SL Method={self.sl_method}, EMA Filter Mode: {self.ema_filter_mode}")

    # ... (El resto del archivo no necesita cambios)
    def _detect_breaks(self, data: pd.DataFrame, levels_for_detection: dict, level_ranges_config: dict, current_candle_idx: int):
        if not levels_for_detection or self.active_breaks:
            return

        if len(data) < 2:
            return

        if current_candle_idx >= len(data) or current_candle_idx < 1:
            return
        
        current_candle = data.iloc[current_candle_idx]
        previous_candle = data.iloc[current_candle_idx - 1]

        try:
            o_curr = pd.to_numeric(current_candle.get('open'), errors='coerce')
            c_curr = pd.to_numeric(current_candle.get('close'), errors='coerce')
            o_prev = pd.to_numeric(previous_candle.get('open'), errors='coerce')
            c_prev = pd.to_numeric(previous_candle.get('close'), errors='coerce')
            if pd.isna(o_curr) or pd.isna(c_curr) or pd.isna(o_prev) or pd.isna(c_prev):
                 return
            current_body_mid = (o_curr + c_curr) / 2.0
            previous_body_mid = (o_prev + c_prev) / 2.0
        except Exception:
            return

        for level_name, level_price_orig in levels_for_detection.items():
            if level_price_orig is None or pd.isna(level_price_orig):
                continue
            try:
                level_price = float(level_price_orig)
            except (TypeError, ValueError):
                 continue
            
            level_range = float(level_ranges_config.get(level_name, 0.2))

            if level_name in self.high_levels:
                if previous_body_mid < level_price and current_body_mid > level_price:
                    break_candle_low_val = pd.to_numeric(current_candle.get('low'), errors='coerce')
                    break_price_val = pd.to_numeric(current_candle.get('close'), errors='coerce')
                    if pd.isna(break_candle_low_val) or pd.isna(break_price_val):
                        continue
                    self.active_breaks[level_name] = {
                        'direction': 'up', 'level_price': level_price, 'level_range': level_range,
                        'break_price': float(break_price_val), 'break_time': current_candle.name,
                        'break_candle_index': current_candle_idx, 'break_candle_low': float(break_candle_low_val),
                        'candles_since_break': 0, 'waiting_retest': True, 'retest_detected': False
                    }
                    return
            elif level_name in self.low_levels:
                if previous_body_mid > level_price and current_body_mid < level_price:
                    break_candle_high_val = pd.to_numeric(current_candle.get('high'), errors='coerce')
                    break_price_val = pd.to_numeric(current_candle.get('close'), errors='coerce')
                    if pd.isna(break_candle_high_val) or pd.isna(break_price_val):
                        continue
                    self.active_breaks[level_name] = {
                        'direction': 'down', 'level_price': level_price, 'level_range': level_range,
                        'break_price': float(break_price_val), 'break_time': current_candle.name,
                        'break_candle_index': current_candle_idx, 'break_candle_high': float(break_candle_high_val),
                        'candles_since_break': 0, 'waiting_retest': True, 'retest_detected': False
                    }
                    return

    def _detect_retests(self, data: pd.DataFrame, current_candle_idx: int):
        if not self.active_breaks or current_candle_idx < 0 or current_candle_idx >= len(data): return
        current_candle = data.iloc[current_candle_idx]
        breaks_to_remove = []

        for level_name in list(self.active_breaks.keys()):
            if level_name in breaks_to_remove: continue
            break_info = self.active_breaks[level_name]
            break_idx = break_info.get('break_candle_index', -1)
            if break_idx < 0 or current_candle_idx <= break_idx: continue

            break_info['candles_since_break'] = current_candle_idx - break_idx
            if break_info['candles_since_break'] > self.max_retest_candles:
                breaks_to_remove.append(level_name)
                continue

            try:
                o_curr=pd.to_numeric(current_candle.get('open'),errors='coerce')
                c_curr=pd.to_numeric(current_candle.get('close'),errors='coerce')
                if pd.isna(o_curr) or pd.isna(c_curr): raise ValueError("Open/Close inválido")
                current_body_mid=(o_curr+c_curr)/2.0
                level_price=break_info['level_price']

                if break_info['direction'] == 'up' and current_body_mid < level_price:
                    breaks_to_remove.append(level_name)
                    continue
                if break_info['direction'] == 'down' and current_body_mid > level_price:
                    breaks_to_remove.append(level_name)
                    continue
            except Exception:
                continue 

            if not break_info.get('waiting_retest', False): continue

            try:
                l_curr=pd.to_numeric(current_candle.get('low'),errors='coerce')
                h_curr=pd.to_numeric(current_candle.get('high'),errors='coerce')
                c_curr=pd.to_numeric(current_candle.get('close'),errors='coerce')
                if pd.isna(l_curr) or pd.isna(h_curr) or pd.isna(c_curr): raise ValueError("L/H/C inválido")
                
                level_price=break_info['level_price']
                level_range=break_info['level_range']
                retest_detected = False
                
                retest_zone_high = level_price + level_range
                retest_zone_low = level_price - level_range

                if break_info['direction'] == 'up':
                    retest_detected = (l_curr <= retest_zone_high and c_curr > level_price)
                elif break_info['direction'] == 'down':
                    retest_detected = (h_curr >= retest_zone_low and c_curr < level_price)

                if retest_detected:
                    break_info.update({
                        'retest_detected':True,'waiting_retest':False,
                        'retest_time':current_candle.name,'retest_price':float(c_curr),
                        'retest_candle_index':current_candle_idx
                    })
            except Exception:
                pass

        if breaks_to_remove:
            for level_name in breaks_to_remove:
                if level_name in self.active_breaks:
                    del self.active_breaks[level_name]

    def _check_for_active_signals(self, data: pd.DataFrame, current_candle_idx: int, current_day_all_levels: dict, ema_values_current_candle: dict) -> str | dict:
        if not self.active_breaks: return 'HOLD'

        for level_name, break_info in list(self.active_breaks.items()): 
            retest_candle_idx = break_info.get('retest_candle_index', -1)

            if (break_info.get('retest_detected', False) and not break_info.get('signal_generated', False)):
                if retest_candle_idx != current_candle_idx:
                    continue 

                signal_type = 'BUY' if break_info['direction'] == 'up' else 'SELL'
                
                passed_ema_filter = True
                if self.use_ema_filter:
                    if not self.ema_periods or len(self.ema_periods) < 3:
                        pass
                    else:
                        ema_short_period, ema_mid_period, ema_long_period = self.ema_periods[0], self.ema_periods[1], self.ema_periods[2]
                        
                        ema_short_val = ema_values_current_candle.get(f'EMA_{ema_short_period}')
                        ema_mid_val = ema_values_current_candle.get(f'EMA_{ema_mid_period}')
                        ema_long_val = ema_values_current_candle.get(f'EMA_{ema_long_period}')
                        retest_close_price_val = break_info.get('retest_price')

                        if not (pd.isna(ema_short_val) or pd.isna(ema_mid_val) or pd.isna(ema_long_val) or pd.isna(retest_close_price_val)):
                            ema_s = float(ema_short_val)
                            ema_m = float(ema_mid_val)
                            ema_l = float(ema_long_val)
                            retest_close = float(retest_close_price_val)
                            
                            current_filter_cond_met = False
                            if signal_type == 'BUY':
                                strong_bullish_cond = ema_s > ema_m and ema_m > ema_l
                                moderate_bullish_cond = retest_close > ema_m and ema_s > ema_m
                                if self.ema_filter_mode == 'Fuerte' and strong_bullish_cond: current_filter_cond_met = True
                                elif self.ema_filter_mode == 'Moderado' and moderate_bullish_cond: current_filter_cond_met = True
                                elif self.ema_filter_mode == 'Cualquiera' and (strong_bullish_cond or moderate_bullish_cond): current_filter_cond_met = True
                            
                            elif signal_type == 'SELL':
                                strong_bearish_cond = ema_s < ema_m and ema_m < ema_l
                                moderate_bearish_cond = retest_close < ema_m and ema_s < ema_m
                                if self.ema_filter_mode == 'Fuerte' and strong_bearish_cond: current_filter_cond_met = True
                                elif self.ema_filter_mode == 'Moderado' and moderate_bearish_cond: current_filter_cond_met = True
                                elif self.ema_filter_mode == 'Cualquiera' and (strong_bearish_cond or moderate_bearish_cond): current_filter_cond_met = True

                            if not current_filter_cond_met:
                                passed_ema_filter = False
                
                if not passed_ema_filter:
                    continue
                
                self.active_breaks[level_name]['signal_generated'] = True 
                
                try:
                    entry_price = float(break_info.get('retest_price', np.nan))
                    if pd.isna(entry_price):
                        self.active_breaks.pop(level_name, None); continue

                    sl_price = np.nan
                    break_idx = break_info.get('break_candle_index', -1)
                    if break_idx < 0 or break_idx >= len(data):
                        self.active_breaks.pop(level_name, None); continue

                    sl_method_to_use = self.sl_method
                    if sl_method_to_use == 'LOOKBACK_MIN_MAX':
                        start_idx_default = max(0, break_idx - (self.sl_default_lookback_candles - 1))
                        end_idx = break_idx + 1 
                        candles_default = data.iloc[start_idx_default:end_idx]
                        sl_price_default_lookback = np.nan
                        if not candles_default.empty:
                            lows = pd.to_numeric(candles_default['low'], errors='coerce')
                            highs = pd.to_numeric(candles_default['high'], errors='coerce')
                            if signal_type == 'BUY': sl_price_default_lookback = lows.min()
                            else: sl_price_default_lookback = highs.max()
                        
                        if pd.notna(sl_price_default_lookback): sl_price = sl_price_default_lookback
                        else: sl_method_to_use = 'BREAK_LOW_HIGH'
                        
                        if sl_method_to_use == 'LOOKBACK_MIN_MAX' and pd.notna(sl_price) and level_name in ['ORH','ORL']:
                            orh_val_levels = current_day_all_levels.get('ORH') 
                            orl_val_levels = current_day_all_levels.get('ORL') 
                            if (orh_val_levels is not None and orl_val_levels is not None and 
                                pd.notna(orh_val_levels) and pd.notna(orl_val_levels) and
                                isinstance(orh_val_levels, numbers.Number) and isinstance(orl_val_levels, numbers.Number) and
                                orh_val_levels > orl_val_levels):
                                or_midpoint=(float(orh_val_levels)+float(orl_val_levels))/2.0
                                reduce_lookback=False
                                if signal_type=='BUY' and sl_price < or_midpoint: reduce_lookback=True
                                elif signal_type=='SELL' and sl_price > or_midpoint: reduce_lookback=True

                                if reduce_lookback:
                                    start_idx_reduced=max(0, break_idx-(self.sl_reduced_lookback_candles-1)) 
                                    candles_reduced=data.iloc[start_idx_reduced:end_idx] 
                                    sl_price_reduced_lookback=np.nan
                                    if not candles_reduced.empty: 
                                        lows_r=pd.to_numeric(candles_reduced['low'],errors='coerce') 
                                        highs_r=pd.to_numeric(candles_reduced['high'],errors='coerce') 
                                        if signal_type == 'BUY': sl_price_reduced_lookback = lows_r.min() 
                                        else: sl_price_reduced_lookback = highs_r.max() 
                                    
                                        if pd.notna(sl_price_reduced_lookback): sl_price = sl_price_reduced_lookback
                            
                    if sl_method_to_use == 'BREAK_LOW_HIGH':
                        if signal_type=='BUY': sl_price = float(break_info.get('break_candle_low', np.nan))
                        else: sl_price = float(break_info.get('break_candle_high', np.nan))

                    if pd.isna(sl_price):
                        self.active_breaks.pop(level_name,None); continue

                    sl_price_f=float(sl_price)
                    entry_price_f=float(entry_price)
                    tp_price=np.nan
                    if abs(entry_price_f - sl_price_f) > 1e-9: 
                        risk_distance = abs(entry_price_f - sl_price_f) 
                        tp_price = entry_price_f + risk_distance * self.risk_reward_ratio if signal_type=='BUY' else entry_price_f - risk_distance * self.risk_reward_ratio 
                    
                    tp_price_final = float(tp_price) if pd.notna(tp_price) else np.nan
                    
                    level_used_for_signal = level_name 
                    self.active_breaks.pop(level_name, None) 

                    return {'type': signal_type, 'sl_price': sl_price_f, 'tp1_price': tp_price_final, 'level': level_name}
                except Exception:
                    self.active_breaks.pop(level_name, None); continue
        return 'HOLD'

    def reset(self):
        self.active_breaks = {}

    def get_signal(self, data: pd.DataFrame, current_day_levels: dict | None = None) -> str | dict:
        min_req_data = max(self.ema_periods) if self.ema_periods and all(isinstance(p, int) and p > 0 for p in self.ema_periods) else 1
        min_req_lookback_sl = self.sl_default_lookback_candles 
        min_req = max(min_req_data, min_req_lookback_sl, 2) 

        if len(data) < min_req:
            return 'HOLD'
        
        current_candle_idx = len(data) - 1
        current_candle_timestamp = data.index[-1]
        
        ema_values = {}
        try:
            close_prices = pd.to_numeric(data['close'], errors='coerce')
            if not close_prices.isna().sum() > 0.5 * len(close_prices):
                for period in self.ema_periods:
                    if isinstance(period, int) and period > 0:
                        if len(close_prices.dropna()) >= period: 
                            current_ema_value = close_prices.ewm(span=period, adjust=False).mean().iloc[-1] 
                            ema_values[f'EMA_{period}'] = current_ema_value 
                        else:
                            ema_values[f'EMA_{period}'] = np.nan 
        except Exception: 
            if self.ema_periods and isinstance(self.ema_periods, list):
                for period in self.ema_periods: 
                    if isinstance(period, int) and period > 0 : ema_values[f'EMA_{period}'] = np.nan

        all_levels_from_arg = current_day_levels if current_day_levels else {}

        signal_from_active_breaks = 'HOLD'
        if self.active_breaks:
            self._detect_retests(data, current_candle_idx)
            if self.active_breaks: 
                 signal_from_active_breaks = self._check_for_active_signals(data, current_candle_idx, all_levels_from_arg, ema_values)

        if isinstance(signal_from_active_breaks, dict) and signal_from_active_breaks.get('type') in ['BUY', 'SELL']: 
            final_signal = {**signal_from_active_breaks, 'emas': ema_values.copy()} 
            return final_signal

        if signal_from_active_breaks == 'HOLD' and not self.active_breaks:
            if all_levels_from_arg:
                levels_for_break_detection = all_levels_from_arg.copy() 

                orh_val = levels_for_break_detection.get('ORH') 
                orl_val = levels_for_break_detection.get('ORL') 

                if (orh_val is not None and orl_val is not None and pd.notna(orh_val) and pd.notna(orl_val) and
                    isinstance(orh_val, numbers.Number) and isinstance(orl_val, numbers.Number) and orh_val > orl_val): 
                    levels_to_filter_within_or = ['PDH', 'PDL', 'PMH', 'PML'] 
                    orl_f = float(orl_val) 
                    orh_f = float(orh_val) 
                    
                    for level_name_to_check in levels_to_filter_within_or: 
                        level_price_to_check = levels_for_break_detection.get(level_name_to_check) 
                        if (level_price_to_check is not None and pd.notna(level_price_to_check) and
                            isinstance(level_price_to_check, numbers.Number)): 
                            level_price_f_check = float(level_price_to_check) 
                            if orl_f < level_price_f_check < orh_f: 
                                if level_name_to_check in levels_for_break_detection: 
                                    del levels_for_break_detection[level_name_to_check] 
                
                if self.ignore_premarket_previousday_breaks_minutes > 0 and levels_for_break_detection:
                    current_candle_timestamp_market = current_candle_timestamp.tz_convert(self.market_tz)
                    current_candle_time_obj = current_candle_timestamp_market.time()
                    
                    market_open_time_obj = datetime.time(9, 30)  
                    
                    dummy_date = datetime.date.min 
                    ignore_period_end_dt_market = (datetime.datetime.combine(dummy_date, market_open_time_obj) + 
                                                   datetime.timedelta(minutes=self.ignore_premarket_previousday_breaks_minutes))
                    ignore_period_end_time_obj = ignore_period_end_dt_market.time()

                    levels_to_ignore_early_session = ['PML', 'PMH', 'PDL', 'PDH']
                    
                    if current_candle_time_obj >= market_open_time_obj and current_candle_time_obj < ignore_period_end_time_obj:
                        keys_to_remove_from_detection = []
                        for level_name in levels_to_ignore_early_session:
                            if level_name in levels_for_break_detection:
                                keys_to_remove_from_detection.append(level_name)
                        
                        if keys_to_remove_from_detection:
                            for key_to_remove in keys_to_remove_from_detection:
                                levels_for_break_detection.pop(key_to_remove) 
                
                if levels_for_break_detection:
                    self._detect_breaks(data, levels_for_break_detection, self.level_ranges, current_candle_idx)
        
        return 'HOLD'
