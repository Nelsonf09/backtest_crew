# ui/streamlit_chart.py
import logging
import pandas as pd
from decimal import Decimal
import datetime
import pytz

from shared.timezone_handler import ensure_timezone_utc

logger = logging.getLogger(__name__)

def prepare_chart_data_and_options(df_display: pd.DataFrame, markers: list = [], chart_height: int = 600,
                                   static_levels: dict | None = None, timezone_str: str = 'America/New_York',
                                   fit_content: bool = True, level_ranges: dict | None = None, **kwargs):
    """ Prepara datos y opciones para streamlit-lightweight-charts con manejo de TZ y rangos de niveles. """
    if df_display is None or df_display.empty:
        logger.warning("prepare_chart_data recibió un DataFrame vacío.")
        return {}, []
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df_display.columns for col in required_cols):
        logger.error(f"Faltan columnas OHLC requeridas. Columnas: {list(df_display.columns)}")
        return {}, []
    if not isinstance(df_display.index, pd.DatetimeIndex):
        logger.error("El índice de df_display no es DatetimeIndex.")
        return {}, []

    local_tz = pytz.timezone(timezone_str)
    if df_display.index.tz is None:
        df_display.index = df_display.index.tz_localize(local_tz, ambiguous='infer')
    elif str(df_display.index.tz) != str(local_tz):
         df_display = df_display.tz_convert(local_tz)

    chart_options = { "height": chart_height, "layout": {"background": {"type": "solid", "color": '#131722'}, "textColor": '#D9D9D9'},
                      "grid": {"vertLines": {"visible": False}, "horzLines": {"color": '#3C4043'}},
                      "timeScale": {"rightOffset": 12, "secondsVisible": True, "timeVisible": True, "borderColor": '#485158', "fitContent": fit_content, "barSpacing": 10},
                      "rightPriceScale": {"borderColor": '#485158', "autoScale": True,},
                      "crosshair": {"mode": 0, "vertLine": {"color": '#C3BCDB40', "labelBackgroundColor": '#9B7DFF'}, "horzLine": {"color": '#C3BCDB40', "labelBackgroundColor": '#9B7DFF'}},
                      "handleScroll": True, "handleScale": True, }

    df_chart_prep = df_display[required_cols].copy()
    for col in required_cols: df_chart_prep[col] = pd.to_numeric(df_chart_prep[col], errors='coerce')
    df_chart_prep = df_chart_prep.dropna(subset=required_cols)

    candlestick_data = []
    min_ts_unix_for_levels, max_ts_unix_for_levels = None, None
    if not df_chart_prep.empty:
        utc_tz = pytz.utc
        new_index_for_chart = [utc_tz.localize(ts_local.replace(tzinfo=None)).timestamp() for ts_local in df_chart_prep.index]
        df_chart = pd.DataFrame({ 'time': new_index_for_chart, 'open': df_chart_prep['open'].values, 'high': df_chart_prep['high'].values, 'low': df_chart_prep['low'].values, 'close': df_chart_prep['close'].values })
        candlestick_data = df_chart.to_dict(orient='records')
        if candlestick_data:
             min_ts_unix_for_levels = candlestick_data[0]['time']
             max_ts_unix_for_levels = candlestick_data[-1]['time']

    candlestick_options = { "upColor": '#26a69a', "downColor": '#ef5350', "borderVisible": False, "wickUpColor": '#26a69a', "wickDownColor": '#ef5350', "priceFormat": {"type": 'price', "precision": 4, "minMove": 0.0001,} }

    formatted_markers = []
    if markers and candlestick_data:
        for m in markers:
            try:
                ts_aware = ensure_timezone_utc(pd.Timestamp(m.get('time')))
                ts_aware_local = ts_aware.tz_convert(local_tz)
                ts_unix_for_chart = pytz.utc.localize(ts_aware_local.replace(tzinfo=None)).timestamp()
                if min_ts_unix_for_levels <= ts_unix_for_chart <= max_ts_unix_for_levels:
                    formatted_markers.append({ "time": ts_unix_for_chart, "position": m.get('position', 'belowBar'), "color": m.get('color', 'blue'), "shape": m.get('shape', 'arrowUp'), "text": str(m.get('text',''))[:20] })
            except Exception as e:
                logger.warning(f"Error procesando marcador: {e} - Data: {m}")

    series_data_list = []
    if candlestick_data:
        series_data_list.append({ "type": "Candlestick", "data": candlestick_data, "options": candlestick_options, "markers": formatted_markers })

    if static_levels and candlestick_data:
        level_styles = { 'PDH': {'c': '#E57373', 'ls': 2, 't': 'PDH', 'rc': '#E573734D'}, 'PDL': {'c': '#81C784', 'ls': 2, 't': 'PDL', 'rc': '#81C7844D'},
                         'PMH': {'c': '#FFB74D', 'ls': 1, 't': 'PMH', 'rc': '#FFB74D4D'}, 'PML': {'c': '#64B5F6', 'ls': 1, 't': 'PML', 'rc': '#64B5F64D'},
                         'ORH': {'c': '#FFF176', 'ls': 0, 't': 'ORH', 'rc': '#FFF1764D'}, 'ORL': {'c': '#FFF176', 'ls': 0, 't': 'ORL', 'rc': '#FFF1764D'},
                         'SL': {'c': '#F48FB1', 'ls': 4, 't': 'SL'}, 'TP': {'c': '#A5D6A7', 'ls': 4, 't': 'TP'}}
        for level_name, price in static_levels.items():
            if price is None or pd.isna(price): continue
            price_float = float(price)
            style = level_styles.get(level_name)
            if style:
                line_data = [{'time': min_ts_unix_for_levels, 'value': price_float}, {'time': max_ts_unix_for_levels, 'value': price_float}]
                line_opts = { "color": style['c'], "lineWidth": 1, "lineStyle": style['ls'], "priceLineVisible": False, "lastValueVisible": False, "crosshairMarkerVisible": False, "title": f"{style['t']} {price_float:.2f}" }
                series_data_list.append({"type": "Line", "data": line_data, "options": line_opts})
                
                # --- LÓGICA RESTAURADA PARA RANGOS ---
                range_val = level_ranges.get(level_name) if level_ranges and style.get('rc') else None
                if range_val:
                     range_f = float(range_val)
                     line_opts_r = {"lineWidth": 1, "lineStyle": 3, "priceLineVisible": False, "lastValueVisible": False, "crosshairMarkerVisible": False, "color": style['rc']}
                     up_data = [{'time': min_ts_unix_for_levels, 'value': price_float + range_f}, {'time': max_ts_unix_for_levels, 'value': price_float + range_f}]
                     series_data_list.append({"type":"Line", "data":up_data, "options": {**line_opts_r, "title": f"{style['t']}+"}})
                     lo_data = [{'time': min_ts_unix_for_levels, 'value': price_float - range_f}, {'time': max_ts_unix_for_levels, 'value': price_float - range_f}]
                     series_data_list.append({"type":"Line", "data":lo_data, "options": {**line_opts_r, "title": f"{style['t']}-"}})

    return chart_options, series_data_list
