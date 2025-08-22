# ui/streamlit_chart.py
import logging
import pandas as pd
from decimal import Decimal
import datetime
import pytz

# <--- CAMBIO CLAVE: Actualizamos la ruta de importación
from shared.timezone_handler import ensure_timezone_utc

# Logger específico para este módulo
logger = logging.getLogger(__name__)

def prepare_chart_data_and_options(df_display: pd.DataFrame, markers: list = [], chart_height: int = 600,
                                   static_levels: dict | None = None, timezone_str: str = 'America/New_York',
                                   fit_content: bool = True, level_ranges: dict | None = None, **kwargs):
    """ Prepara datos y opciones para streamlit-lightweight-charts con manejo de TZ. """
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

    local_tz = None
    try: local_tz = pytz.timezone(timezone_str)
    except Exception as e_tz: logger.error(f"Timezone inválida '{timezone_str}': {e_tz}"); return {}, []

    if df_display.index.tz is None:
        logger.warning(f"Índice df_display SIN Timezone. Asumiendo y localizando a {timezone_str}.")
        try: df_display.index = df_display.index.tz_localize(local_tz, ambiguous='infer')
        except Exception as e_loc: logger.error(f"Fallo localizando a {timezone_str}: {e_loc}"); return {}, []
    elif str(df_display.index.tz) != str(local_tz):
         logger.warning(f"Convirtiendo índice de {df_display.index.tz} a {timezone_str}")
         try: df_display = df_display.tz_convert(local_tz)
         except Exception as e_conv: logger.error(f"Fallo convirtiendo índice a {timezone_str}: {e_conv}"); return {}, []

    # --- Opciones del Gráfico ---
    chart_options = { "height": chart_height, "layout": {"background": {"type": "solid", "color": '#131722'}, "textColor": '#D9D9D9'},
                      "grid": {"vertLines": {"visible": False}, "horzLines": {"color": '#3C4043'}},
                      "timeScale": {"rightOffset": 12, "secondsVisible": True, "timeVisible": True, "borderColor": '#485158', "fitContent": fit_content, "barSpacing": 10},
                      "rightPriceScale": {"borderColor": '#485158', "autoScale": True,},
                      "crosshair": {"mode": 0, "vertLine": {"color": '#C3BCDB40', "labelBackgroundColor": '#9B7DFF'}, "horzLine": {"color": '#C3BCDB40', "labelBackgroundColor": '#9B7DFF'}},
                      "handleScroll": True, "handleScale": True, }

    # --- Datos de la Serie de Velas (con manejo de TZ para eje X) ---
    candlestick_data = []
    min_ts_unix_for_levels = None; max_ts_unix_for_levels = None
    try:
        df_chart_prep = df_display[required_cols].copy()
        for col in required_cols: df_chart_prep[col] = pd.to_numeric(df_chart_prep[col], errors='coerce')
        df_chart_prep = df_chart_prep.dropna(subset=required_cols)

        if not df_chart_prep.empty:
            utc_tz = pytz.utc
            new_index_for_chart = []
            for ts_local in df_chart_prep.index:
                naive_dt = ts_local.replace(tzinfo=None)
                utc_dt_for_unix = utc_tz.localize(naive_dt)
                unix_ts = utc_dt_for_unix.timestamp()
                new_index_for_chart.append(unix_ts)

            df_chart = pd.DataFrame({ 'time': new_index_for_chart, 'open': df_chart_prep['open'].values,
                                      'high': df_chart_prep['high'].values, 'low': df_chart_prep['low'].values,
                                      'close': df_chart_prep['close'].values, })
            candlestick_data = df_chart.to_dict(orient='records')
            if candlestick_data:
                 min_ts_unix_for_levels = candlestick_data[0]['time']
                 max_ts_unix_for_levels = candlestick_data[-1]['time']
    except Exception as e_data: logger.error(f"Error preparando datos velas chart: {e_data}", exc_info=True); return {}, []

    # --- Opciones Velas ---
    candlestick_options = { "upColor": '#26a69a', "downColor": '#ef5350', "borderVisible": False,
                            "wickUpColor": '#26a69a', "wickDownColor": '#ef5350',
                            "priceFormat": {"type": 'price', "precision": 4, "minMove": 0.0001,} }

    # --- Marcadores ---
    formatted_markers = []
    if markers and candlestick_data and min_ts_unix_for_levels is not None:
        for idx, m in enumerate(markers):
            try:
                ts_orig = m.get('time')
                if not isinstance(ts_orig, (pd.Timestamp, datetime.datetime)): ts_orig = pd.Timestamp(ts_orig)

                ts_aware = ensure_timezone_utc(ts_orig)
                if ts_aware is None:
                    continue

                ts_aware_local = ts_aware.tz_convert(local_tz)
                naive_dt = ts_aware_local.replace(tzinfo=None)
                utc_dt_for_unix = pytz.utc.localize(naive_dt)
                ts_unix_for_chart = utc_dt_for_unix.timestamp()

                if min_ts_unix_for_levels <= ts_unix_for_chart <= max_ts_unix_for_levels:
                    text = str(m.get('text',''))[:20]
                    formatted_markers.append({ "time": ts_unix_for_chart, "position": m.get('position', 'belowBar'),
                                               "color": m.get('color', 'blue'), "shape": m.get('shape', 'arrowUp'), "text": text })
            except Exception as e_mform:
                logger.warning(f"Error procesando marcador {idx}: {e_mform} - Data: {m}", exc_info=False)

    # --- Lista de Series ---
    series_data_list = []
    if candlestick_data:
        series_data_list.append({ "type": "Candlestick", "data": candlestick_data,
                                  "options": candlestick_options, "markers": formatted_markers })
    else: logger.warning("No hay datos de velas para añadir a series_data_list.")

    # --- Series de Línea para Niveles y Rangos ---
    if static_levels and candlestick_data and min_ts_unix_for_levels is not None:
        level_styles = { 'PDH': {'c': '#E57373', 'ls': 2, 't': 'PDH', 'rc': '#E573734D'}, 'PDL': {'c': '#81C784', 'ls': 2, 't': 'PDL', 'rc': '#81C7844D'},
                         'PMH': {'c': '#FFB74D', 'ls': 1, 't': 'PMH', 'rc': '#FFB74D4D'}, 'PML': {'c': '#64B5F6', 'ls': 1, 't': 'PML', 'rc': '#64B5F64D'},
                         'ORH': {'c': '#FFF176', 'ls': 0, 't': 'ORH', 'rc': '#FFF1764D'}, 'ORL': {'c': '#FFF176', 'ls': 0, 't': 'ORL', 'rc': '#FFF1764D'},
                         'SL': {'c': '#F48FB1', 'ls': 4, 't': 'SL', 'rc': '#F48FB14D'}, 'TP': {'c': '#A5D6A7', 'ls': 4, 't': 'TP', 'rc': '#A5D6A74D'}}
        for level_name, price in static_levels.items():
            if price is None or pd.isna(price): continue
            try: price_float = float(price)
            except (ValueError, TypeError): continue
            style = level_styles.get(level_name)
            if style:
                line_data = [{'time': min_ts_unix_for_levels, 'value': price_float}, {'time': max_ts_unix_for_levels, 'value': price_float}]
                line_opts = { "color": style['c'], "lineWidth": 1, "lineStyle": style['ls'], "priceLineVisible": False,
                              "lastValueVisible": False, "crosshairMarkerVisible": False, "title": f"{style['t']} {price_float:.2f}" }
                series_data_list.append({"type": "Line", "data": line_data, "options": line_opts})
                
                range_val = level_ranges.get(level_name) if level_ranges else None
                if range_val is not None and range_val > 0 and level_name not in ['SL', 'TP']:
                     try:
                         range_f = float(range_val)
                         line_opts_r = {"lineWidth": 1, "lineStyle": 3, "priceLineVisible": False, "lastValueVisible": False, "crosshairMarkerVisible": False}
                         up_data = [{'time': min_ts_unix_for_levels, 'value': price_float + range_f}, {'time': max_ts_unix_for_levels, 'value': price_float + range_f}]
                         series_data_list.append({"type":"Line", "data":up_data, "options": {**line_opts_r, "color": style['rc'], "title": f"{style['t']}+"}})
                         lo_data = [{'time': min_ts_unix_for_levels, 'value': price_float - range_f}, {'time': max_ts_unix_for_levels, 'value': price_float - range_f}]
                         series_data_list.append({"type":"Line", "data":lo_data, "options": {**line_opts_r, "color": style['rc'], "title": f"{style['t']}-"}})
                     except (ValueError, TypeError): pass
            else: logger.warning(f"Estilo no encontrado para nivel: {level_name}")

    return chart_options, series_data_list
