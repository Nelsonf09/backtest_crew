# ui/app.py
"""
Aplicación Streamlit para el Backtesting Player Visual.
FASE 1: Refactorizado para ser controlado por un Agent Core.
- La lógica de la estrategia ha sido eliminada de este archivo.
- La generación de señales ahora es manejada por el `agent_core`.
- Todas las importaciones han sido actualizadas a la nueva estructura de carpetas.
"""

# --- INICIO PARCHE ASYNCIO ---
import asyncio
import nest_asyncio
nest_asyncio.apply()
# --- FIN PARCHE ASYNCIO ---

import logging
LOG_LEVEL = logging.INFO
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s - %(levelname)-7s - %(name)-25s:%(lineno)3d - %(message)s',
                        datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import datetime
from decimal import Decimal, InvalidOperation
import traceback
import pytz
import math
import sys

# <--- CAMBIO CLAVE: Añadimos la ruta raíz del proyecto al path de Python
# Esto permite que app.py (dentro de /ui) encuentre los módulos en /agent_core y /shared
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
    from agent_core.main import handle_signal_request
    from agent_core.data_manager import DataManager
    from ui.streamlit_chart import prepare_chart_data_and_options
    from streamlit_lightweight_charts import renderLightweightCharts
    from agent_core.execution import ExecutionSimulator
    from agent_core.metrics import calculate_performance_metrics
    # <--- CAMBIO CLAVE: Importamos desde la nueva carpeta 'shared'
    from shared.timezone_handler import TimezoneHandler, apply_timezone_fixes
except ImportError as e:
    st.error(f"Error importando módulos necesarios: {e}. Asegúrate de que la estructura de carpetas es correcta y todos los archivos __init__.py existen.")
    logger.error(f"ImportError crítico: {e}", exc_info=True)
    traceback.print_exc()
    st.stop()
except Exception as e_import:
     st.error(f"Error inesperado durante la importación de módulos: {e_import}")
     logger.error(f"Error genérico en importación: {e_import}", exc_info=True)
     traceback.print_exc()
     st.stop()

# --- Constantes y Configuraciones de la App (SIN CAMBIOS) ---
STRATEGY_OBR_NAME = "Opening Break & Retest"
BR_STRATEGY_PARAMS = {
    "max_retest_candles": getattr(config, 'BR_MAX_RETEST_CANDLES', 20),
    "risk_reward_ratio": getattr(config, 'BR_RISK_REWARD_RATIO', 2.0),
    "sl_method": getattr(config, 'BR_SL_METHOD', 'LOOKBACK_MIN_MAX'),
    "sl_lookback": getattr(config, 'BR_SL_LOOKBACK', 2),
    "ema_periods": getattr(config, 'BR_EMA_PERIODS', [9, 21, 50]),
    "ignore_premarket_previousday_breaks_minutes": getattr(config, 'BR_IGNORE_PM_PD_BREAKS_MINUTES', 5),
    "ema_filter_mode": "Desactivado"
}
ACTIVE_STRATEGY_DEFAULT = STRATEGY_OBR_NAME
LEVEL_RANGES = getattr(config, 'LEVEL_RANGES', {
    'PDH': 0.5, 'PDL': 0.5, 'PMH': 0.3, 'PML': 0.3, 'ORH': 0.2, 'ORL': 0.2
})
INITIAL_CAPITAL = getattr(config, 'INITIAL_CAPITAL', 1000.0)
COMMISSION_PER_TRADE = getattr(config, 'COMMISSION_PER_TRADE', 0.85)
SLIPPAGE_POINTS = getattr(config, 'SLIPPAGE_POINTS', 0.0)
DEFAULT_LEVERAGE = getattr(config, 'DEFAULT_LEVERAGE', 5)
VELAS_HACIA_ATRAS_DEFAULT = getattr(config, 'VELAS_HACIA_ATRAS', 17)
VELAS_HACIA_ADELANTE = 0

st.set_page_config(
    layout="wide",
    page_title="Scalping Backtester Visual",
    initial_sidebar_state="expanded"
)

# --- Inicialización del Estado de Sesión (st.session_state) (SIN CAMBIOS) ---
if 'app_initialized' not in st.session_state:
    _today = datetime.date.today()
    _default_download_start = _today - datetime.timedelta(days=getattr(config, 'DEFAULT_DOWNLOAD_RANGE_DAYS', 5))
    _default_download_end = _today
    _default_start_date_obj_candidate = _today - datetime.timedelta(days=getattr(config, 'DEFAULT_START_DATE_OFFSET_DAYS', 1))

    if _default_start_date_obj_candidate < _default_download_start:
        _initial_start_date_obj = _default_download_start
    elif _default_start_date_obj_candidate > _default_download_end:
        _initial_start_date_obj = _default_download_end
    else:
        _initial_start_date_obj = _default_start_date_obj_candidate

    default_session_state_values = {
        'replay_started': False, 'all_data_utc': None, 'df_context_display': None,
        'df_replay_display': None, 'current_index': 0,
        'display_tz': getattr(config, 'DEFAULT_DISPLAY_TZ', 'America/New_York'),
        'processed_tz': None, 'download_start_date': _default_download_start,
        'download_end_date': _default_download_end, 'start_date_obj': _initial_start_date_obj,
        'processed_start_date': None, 'symbol_processed': None,
        'strategy_name': ACTIVE_STRATEGY_DEFAULT, 'selected_leverage': DEFAULT_LEVERAGE,
        'executor': None, 'markers': [],
        'closed_pnl': Decimal('0.0'), 'last_equity': Decimal(str(INITIAL_CAPITAL)),
        'symbol_loaded': config.DEFAULT_SYMBOL,
        'timeframe_loaded': config.DEFAULT_TIMEFRAME,
        'equity_history': [], 'performance_metrics': {},
        'static_levels': {}, 'opening_levels': {}, 'tz_handler': None,
        'mostrar_todo': False, 'velas_atras': VELAS_HACIA_ATRAS_DEFAULT,
        'last_processed_day': None,
        'last_trade_sl_display': None,
        'last_trade_tp_display': None,
        'autoplaying': False,
        'autoplay_speed': 0.5,
        'app_initialized': True,
        'sidebar_sec_type': getattr(config, 'DEFAULT_SEC_TYPE', 'STK'),
        'sidebar_exchange': getattr(config, 'DEFAULT_EXCHANGE', 'SMART'),
        'sidebar_currency': getattr(config, 'DEFAULT_CURRENCY', 'USD'),
        'sidebar_primary_exchange': getattr(config, 'DEFAULT_PRIMARY_EXCHANGE', 'ISLAND'),
        'sidebar_use_cache': getattr(config, 'ENABLE_CACHING', True),
        'obr_ema_filter_mode': BR_STRATEGY_PARAMS.get("ema_filter_mode", "Desactivado")
    }
    for key, value in default_session_state_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
else:
    if 'download_start_date_widget' in st.session_state and \
       st.session_state.download_start_date != st.session_state.download_start_date_widget:
         st.session_state.download_start_date = st.session_state.download_start_date_widget

    if 'download_end_date_widget' in st.session_state and \
       st.session_state.download_end_date != st.session_state.download_end_date_widget:
         st.session_state.download_end_date = st.session_state.download_end_date_widget

if st.session_state.tz_handler is None:
    try:
        st.session_state.tz_handler = TimezoneHandler(default_display_tz_str=st.session_state.display_tz)
    except Exception as e_tz_init:
         st.error(f"Error inicializando TimezoneHandler: {e_tz_init}")
         logger.error(f"Error TimezoneHandler init: {e_tz_init}", exc_info=True)
         st.stop()

# --- Funciones de Callback para Widgets (SIN CAMBIOS) ---
def prev_candle():
    if st.session_state.current_index > 0 and not st.session_state.get('autoplaying', False):
        st.session_state.current_index -= 1
    logger.debug(f"Callback: prev_candle. Nuevo índice replay: {st.session_state.current_index}")

def next_candle():
    if st.session_state.df_replay_display is not None and \
       st.session_state.current_index < len(st.session_state.df_replay_display) - 1 and \
       not st.session_state.get('autoplaying', False):
        st.session_state.current_index += 1
    elif not st.session_state.get('autoplaying', False):
        st.toast("Fin de los datos para este día.")
        logger.warning("Callback: next_candle. No se puede avanzar más (fin del día o datos no cargados).")
    logger.debug(f"Callback: next_candle. Nuevo índice replay: {st.session_state.current_index}")

def actualizar_mostrar_todo():
    st.session_state.mostrar_todo = not st.session_state.get('mostrar_todo', False)
    logger.debug(f"Callback: actualizar_mostrar_todo. Nuevo estado: {st.session_state.mostrar_todo}")

def actualizar_velas_atras(input_key='velas_atras_input_widget'):
    velas_input_value = st.session_state.get(input_key, VELAS_HACIA_ATRAS_DEFAULT)
    try:
        new_value = int(velas_input_value)
        if new_value > 0 and new_value != st.session_state.velas_atras:
            st.session_state.velas_atras = new_value
            logger.debug(f"Callback: actualizar_velas_atras. Nuevo valor: {new_value}")
    except (ValueError, TypeError):
        logger.warning(f"Callback: actualizar_velas_atras. Valor inválido: {velas_input_value}.")
        st.toast(f"Número de velas inválido: {velas_input_value}")

def go_to_next_day():
    if st.session_state.get('autoplaying', False):
        st.toast("Pausa la reproducción automática para cambiar de día.")
        return

    if not st.session_state.replay_started or st.session_state.all_data_utc is None or st.session_state.all_data_utc.empty:
        st.warning("Carga primero los datos antes de avanzar de día.")
        return

    logger.info("Intentando avanzar al siguiente día...")
    current_date_of_replay = st.session_state.start_date_obj

    if current_date_of_replay >= st.session_state.download_end_date:
        st.warning(f"No se puede avanzar más allá de la fecha final de descarga: {st.session_state.download_end_date.strftime('%Y-%m-%d')}")
        return

    try:
        if not isinstance(st.session_state.all_data_utc.index, pd.DatetimeIndex) or \
           st.session_state.all_data_utc.index.tz is None:
             st.error("Error interno: Formato de datos UTC incorrecto (índice o timezone).")
             logger.error("go_to_next_day: all_data_utc no tiene índice DatetimeIndex UTC.")
             return

        all_available_dates_in_display_tz = sorted(list(set(
            st.session_state.all_data_utc.index.tz_convert(st.session_state.display_tz).date
        )))
        
        next_day_date_to_set = None
        for available_date in all_available_dates_in_display_tz:
            if available_date > current_date_of_replay and available_date <= st.session_state.download_end_date:
                next_day_date_to_set = available_date
                break
        
        if not next_day_date_to_set:
            st.warning(f"No hay más días con datos disponibles hasta {st.session_state.download_end_date.strftime('%Y-%m-%d')} en el rango cargado.")
            return

        logger.info(f"Avanzando de {current_date_of_replay.strftime('%Y-%m-%d')} al siguiente día con datos: {next_day_date_to_set.strftime('%Y-%m-%d')}")
        
        st.session_state.start_date_obj = next_day_date_to_set
        
        st.session_state.markers = []
        st.session_state.current_index = 0
    except Exception as e:
        st.error(f"Error al intentar avanzar al siguiente día: {e}")
        logger.error(f"Error en go_to_next_day: {e}", exc_info=True)

def close_position_manually():
    if st.session_state.get('autoplaying', False):
        st.toast("Pausa la reproducción automática para cerrar la posición manualmente.")
        return

    logger.info("Intento de cierre manual de posición.")
    if st.session_state.executor and st.session_state.executor.position != 'FLAT' and \
       st.session_state.df_replay_display is not None and not st.session_state.df_replay_display.empty:

        current_idx_replay = st.session_state.current_index
        if 0 <= current_idx_replay < len(st.session_state.df_replay_display):
            current_candle_for_close = st.session_state.df_replay_display.iloc[current_idx_replay]
            
            position_before_manual_close = st.session_state.executor.position
            sl_before_manual_close_dec = st.session_state.executor.sl_price
            tp_before_manual_close_dec = st.session_state.executor.tp_price

            manual_close_signal = 'SELL' if st.session_state.executor.position == 'LONG' else 'BUY'
            logger.info(f"Cerrando manually posición {st.session_state.executor.position} con señal {manual_close_signal} en vela: {current_candle_for_close.name}")

            execution_result = st.session_state.executor.process_signal(manual_close_signal, current_candle_for_close)

            if execution_result and 'marker' in execution_result and execution_result['marker']:
                manual_marker = execution_result['marker']
                original_pnl_text = manual_marker.get('text', '')
                pnl_value_part = original_pnl_text[original_pnl_text.find("("):] if '(' in original_pnl_text else ""
                manual_marker['text'] = f"MANUAL_CLOSE {pnl_value_part}".strip()
                manual_marker['color'] = 'orange'
                
                is_duplicate = False
                if st.session_state.markers:
                    last_m = st.session_state.markers[-1]
                    try:
                        if abs(pd.Timestamp(last_m.get('time')).timestamp() - pd.Timestamp(manual_marker.get('time')).timestamp()) < 1 and \
                           last_m.get('shape') == manual_marker.get('shape'):
                            is_duplicate = True
                            logger.debug("Marcador de cierre manual potencialmente duplicado, no se añade.")
                    except: pass
                
                if not is_duplicate:
                    st.session_state.markers.append(manual_marker)

            if position_before_manual_close != 'FLAT' and st.session_state.executor.position == 'FLAT':
                if not sl_before_manual_close_dec.is_nan():
                    try: st.session_state.last_trade_sl_display = float(sl_before_manual_close_dec)
                    except: st.session_state.last_trade_sl_display = None
                else: st.session_state.last_trade_sl_display = None

                if not tp_before_manual_close_dec.is_nan():
                    try: st.session_state.last_trade_tp_display = float(tp_before_manual_close_dec)
                    except: st.session_state.last_trade_tp_display = None
                else: st.session_state.last_trade_tp_display = None
                logger.debug(f"Operación cerrada manualmente. Guardando SL: {st.session_state.last_trade_sl_display}, TP: {st.session_state.last_trade_tp_display}")


            st.session_state.closed_pnl = st.session_state.executor.closed_pnl_total
            st.session_state.last_equity = st.session_state.executor.last_equity
            st.session_state.equity_history = st.session_state.executor.get_equity_history()
            st.toast(f"Posición cerrada manualmente.")
            st.rerun()
        else:
            st.warning("No se pudo obtener la vela actual para el cierre manual.")
            logger.warning("Cierre manual: Índice de vela actual fuera de rango o df_replay_display vacío.")
    else:
        st.toast("No hay posición abierta para cerrar manualmente.")
        logger.info("Cierre manual: No hay posición abierta o executor/datos no disponibles.")

def toggle_autoplay():
    st.session_state.autoplaying = not st.session_state.get('autoplaying', False)
    if not st.session_state.autoplaying:
        logger.info("Autoplay pausado.")
    else:
        logger.info(f"Autoplay iniciado a {st.session_state.get('autoplay_speed', 0.5)} s/vela.")
# --- FIN FUNCIONES DE CALLBACK ---

# --- Sidebar y Botón de Carga (SIN CAMBIOS LÓGICOS IMPORTANTES) ---
with st.sidebar:
    st.title("Configuración Backtest")

    with st.expander("1. Instrumento y Rango de Datos", expanded=True):
        available_symbols = config.STOCK_SYMBOLS_LIST
        default_symbol_on_load = st.session_state.get('symbol_loaded', config.DEFAULT_SYMBOL)
        try:
            default_symbol_index = available_symbols.index(default_symbol_on_load)
        except ValueError:
            default_symbol_index = 0
            st.session_state.symbol_loaded = available_symbols[0] if available_symbols else 'AAPL'

        symbol = st.selectbox("Símbolo", options=available_symbols, index=default_symbol_index, key="symbol_select")
        
        sec_type_options = ['STK', 'FUT', 'IND', 'FOREX', 'CFD', 'CRYPTO']
        sec_type_default = getattr(config, 'DEFAULT_SEC_TYPE', 'STK')
        sec_type_index = sec_type_options.index(sec_type_default) if sec_type_default in sec_type_options else 0
        sec_type = st.selectbox("Tipo de Activo", sec_type_options, index=sec_type_index, key="sidebar_sec_type_widget")

        exchange = st.text_input("Exchange", getattr(config, 'DEFAULT_EXCHANGE', 'SMART'), key="sidebar_exchange_widget")
        currency = st.text_input("Moneda", getattr(config, 'DEFAULT_CURRENCY', 'USD'), key="sidebar_currency_widget")
        
        primary_exchange_options = ['', 'NASDAQ', 'NYSE', 'ARCA', 'AMEX', 'ISLAND', 'BATS', 'IEX', 'EDGEA', 'EDGEX']
        default_primary_exchange = getattr(config, 'DEFAULT_PRIMARY_EXCHANGE', 'ISLAND')
        primary_exchange_index = primary_exchange_options.index(default_primary_exchange) if default_primary_exchange in primary_exchange_options else 0
        primary_exchange_st = st.selectbox("Primary Exch. (para STK)", primary_exchange_options, index=primary_exchange_index, help="Necesario para desambiguar acciones en SMART.", key="sidebar_primary_exchange_widget")

        timeframe_options = ['1 sec', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins']
        timeframe_default = st.session_state.get('timeframe_loaded', config.DEFAULT_TIMEFRAME)
        if timeframe_default not in timeframe_options: timeframe_default = '1 min'
        timeframe = st.selectbox("Timeframe (Principal)", timeframe_options, index=timeframe_options.index(timeframe_default))

        st.markdown("**Rango de Descarga de Datos**")
        
        _ds = st.date_input(
            "Fecha Inicio Descarga",
            value=st.session_state.download_start_date,
            key="download_start_date_widget",
            help="Primer día para el cual se descargarán datos de IB."
        )
        if _ds != st.session_state.download_start_date:
            st.session_state.download_start_date = _ds
            if st.session_state.start_date_obj < st.session_state.download_start_date:
                st.session_state.start_date_obj = st.session_state.download_start_date
                logger.debug(f"Ajustado start_date_obj a {st.session_state.start_date_obj} debido a cambio en download_start_date.")

        _de = st.date_input(
            "Fecha Fin Descarga",
            value=st.session_state.download_end_date,
            min_value=st.session_state.download_start_date,
            key="download_end_date_widget",
            help="Último día para el cual se descargarán datos de IB."
        )
        if _de != st.session_state.download_end_date:
            st.session_state.download_end_date = _de
            if st.session_state.start_date_obj > st.session_state.download_end_date:
                st.session_state.start_date_obj = st.session_state.download_end_date
                logger.debug(f"Ajustado start_date_obj a {st.session_state.start_date_obj} debido a cambio en download_end_date.")

        st.markdown("**Inicio del Replay Visual**")
        
        if st.session_state.start_date_obj < st.session_state.download_start_date:
            st.session_state.start_date_obj = st.session_state.download_start_date
            logger.debug(f"Pre-widget: Ajustado start_date_obj a min_download: {st.session_state.start_date_obj}")
        if st.session_state.start_date_obj > st.session_state.download_end_date:
            st.session_state.start_date_obj = st.session_state.download_end_date
            logger.debug(f"Pre-widget: Ajustado start_date_obj a max_download: {st.session_state.start_date_obj}")

        _sdo = st.date_input(
            "Fecha Inicio Replay",
            value=st.session_state.start_date_obj,
            min_value=st.session_state.download_start_date,
            max_value=st.session_state.download_end_date,
            key="date_input_replay_widget",
            help="Día en que comenzará la simulación vela a vela."
        )
        if _sdo != st.session_state.start_date_obj:
             st.session_state.start_date_obj = _sdo

        use_rth_default = getattr(config, 'USE_RTH', True)
        use_rth = st.toggle("Usar solo RTH para datos principales", value=use_rth_default, key="rth_toggle_main_data", help="Aplica a la descarga de datos principales. Los niveles PDL/PMH pueden usar otras reglas.")
        
        what_to_show_options = ['TRADES', 'MIDPOINT', 'BID', 'ASK']
        what_to_show_default = getattr(config, 'WHAT_TO_SHOW', 'TRADES')
        what_to_show_idx = what_to_show_options.index(what_to_show_default) if what_to_show_default in what_to_show_options else 0
        what_to_show = st.selectbox("Fuente de Datos Velas (IB)", what_to_show_options, index=what_to_show_idx)
        
        use_cache = st.toggle("Usar Caché de Datos", getattr(config, 'ENABLE_CACHING', True), key="sidebar_use_cache_widget")

    with st.expander("2. Estrategia y Ejecución", expanded=True):
        strategy_options = [STRATEGY_OBR_NAME]
        selected_strategy_name = st.selectbox(
            "Estrategia",
            strategy_options,
            index=(strategy_options.index(st.session_state.strategy_name) if st.session_state.strategy_name in strategy_options else 0)
        )
        
        ema_filter_options = ["Desactivado", "Moderado", "Fuerte", "Cualquiera"]
        default_obr_ema_filter_mode = st.session_state.get('obr_ema_filter_mode', BR_STRATEGY_PARAMS.get("ema_filter_mode", "Desactivado"))
        
        selected_obr_ema_filter_mode = st.selectbox(
            "Filtro Tendencia EMA (OBR)",
            ema_filter_options,
            index=ema_filter_options.index(default_obr_ema_filter_mode),
            key="obr_ema_filter_mode_widget", 
            help="Filtro de tendencia EMA para la estrategia Opening Break & Retest."
        )

        leverage_options = [1, 2, 5, 10, 20, 30, 50, 100]
        selected_leverage = st.selectbox(
            "Apalancamiento (Leverage)",
            leverage_options,
            index=(leverage_options.index(st.session_state.selected_leverage) if st.session_state.selected_leverage in leverage_options else leverage_options.index(DEFAULT_LEVERAGE)),
            key="leverage_select"
        )

        display_tz_options = pytz.common_timezones
        display_tz_default = st.session_state.display_tz
        try:
            display_tz_idx = display_tz_options.index(display_tz_default)
        except ValueError:
            display_tz_default = 'America/New_York'
            st.session_state.display_tz = display_tz_default
            display_tz_idx = display_tz_options.index(display_tz_default)

        display_tz_selected = st.selectbox("Timezone Gráfico/Tabla", display_tz_options, index=display_tz_idx)

    with st.expander("3. Controles de Reproducción", expanded=True):
        st.number_input(
            "Velocidad Autoplay (s/vela)", 
            min_value=0.1, 
            max_value=5.0, 
            value=st.session_state.get('autoplay_speed', 0.5), 
            step=0.1, 
            format="%.1f",
            key="autoplay_speed_selector_widget",
            on_change=lambda: setattr(st.session_state, 'autoplay_speed', st.session_state.autoplay_speed_selector_widget),
            disabled=st.session_state.get('autoplaying', False)
        )

    st.divider()
    load_button_text = "Cargar Datos e Iniciar Replay" if not st.session_state.get('replay_started', False) else "Recargar Datos y Reiniciar Replay"
    load_button = st.button(load_button_text, key="load_button", use_container_width=True, type="primary")

# --- Lógica Principal (con cambios clave) ---
if load_button:
    st.session_state.obr_ema_filter_mode = st.session_state.obr_ema_filter_mode_widget

    st.session_state.update({
        'replay_started': False, 'all_data_utc': None, 'df_context_display': None,
        'df_replay_display': None, 'current_index': 0,
        'display_tz': display_tz_selected,
        'processed_tz': None, 'processed_start_date': None, 'symbol_processed': None,
        'strategy_name': selected_strategy_name,
        'selected_leverage': selected_leverage,
        'executor': None, 'markers': [],
        'closed_pnl': Decimal('0.0'),
        'last_equity': Decimal(str(INITIAL_CAPITAL)),
        'symbol_loaded': symbol,
        'timeframe_loaded': timeframe,
        'equity_history': [],
        'performance_metrics': {},
        'static_levels': {}, 'opening_levels': {},
        'last_processed_day': None,
        'last_trade_sl_display': None,
        'last_trade_tp_display': None,
        'autoplaying': False,
        'sidebar_sec_type': st.session_state.sidebar_sec_type_widget,
        'sidebar_exchange': st.session_state.sidebar_exchange_widget,
        'sidebar_currency': st.session_state.sidebar_currency_widget,
        'sidebar_primary_exchange': st.session_state.sidebar_primary_exchange_widget,
        'sidebar_use_cache': st.session_state.sidebar_use_cache_widget,
    })
    
    handle_signal_request(historical_data=None, current_levels=None, reset_strategy=True)

    if st.session_state.download_start_date > st.session_state.download_end_date:
        st.error("La 'Fecha Inicio Descarga' no puede ser posterior a la 'Fecha Fin Descarga'.")
        st.stop()

    if st.session_state.start_date_obj < st.session_state.download_start_date:
        st.session_state.start_date_obj = st.session_state.download_start_date
        logger.warning(f"Ajustando Fecha Inicio Replay a {st.session_state.start_date_obj.strftime('%Y-%m-%d')} para coincidir con el inicio del rango de descarga.")
    if st.session_state.start_date_obj > st.session_state.download_end_date:
        st.session_state.start_date_obj = st.session_state.download_end_date
        logger.warning(f"Ajustando Fecha Inicio Replay a {st.session_state.start_date_obj.strftime('%Y-%m-%d')} para coincidir con el fin del rango de descarga.")
    
    try:
        st.session_state.tz_handler = TimezoneHandler(default_display_tz_str=st.session_state.display_tz)
    except Exception as e_tz_reset:
        st.error(f"Error reinicializando TimezoneHandler con {st.session_state.display_tz}: {e_tz_reset}")
        logger.error(f"Error TimezoneHandler reset en carga: {e_tz_reset}", exc_info=True)
        st.stop()

    logger.info(f"--- Iniciando Carga/Reinicio: Símbolo={st.session_state.symbol_loaded} | TF={st.session_state.timeframe_loaded} ---")
    logger.info(f"Rango Descarga: {st.session_state.download_start_date.strftime('%Y-%m-%d')} a {st.session_state.download_end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Fecha Inicio Replay: {st.session_state.start_date_obj.strftime('%Y-%m-%d')}")
    logger.info(f"Estrategia: {st.session_state.strategy_name} | Apalancamiento: {st.session_state.selected_leverage}:1 | Filtro EMA OBR: {st.session_state.obr_ema_filter_mode}")

    data_manager_instance = None
    ib_connection_successful = False
    try:
        data_manager_instance = DataManager()
        with st.spinner("Conectando a Interactive Brokers..."):
            if not data_manager_instance.connect_ib():
                st.error("Fallo la conexión a Interactive Brokers. Verifica que TWS/Gateway esté ejecutándose y configurado para API.")
                st.stop()
            ib_connection_successful = True
        
        progress_bar = st.progress(0, text="Descargando datos principales...")
        with st.spinner(f"Cargando datos para {st.session_state.symbol_loaded} ({st.session_state.timeframe_loaded}) desde {st.session_state.download_start_date.strftime('%Y-%m-%d')} hasta {st.session_state.download_end_date.strftime('%Y-%m-%d')}..."):
            contract_details_for_main_data = {
                'primary_exchange': st.session_state.sidebar_primary_exchange if st.session_state.sidebar_primary_exchange and st.session_state.sidebar_sec_type == 'STK' else None
            }
            rth_value_for_main_fetch = st.session_state.get('rth_toggle_main_data', use_rth_default)
            
            df_total_utc_data = data_manager_instance.get_main_data(
                symbol=st.session_state.symbol_loaded,
                timeframe=st.session_state.timeframe_loaded,
                sec_type=st.session_state.sidebar_sec_type,
                exchange=st.session_state.sidebar_exchange,
                currency=st.session_state.sidebar_currency,
                rth=rth_value_for_main_fetch,
                what_to_show=what_to_show,
                download_start_date=st.session_state.download_start_date,
                download_end_date=st.session_state.download_end_date,
                use_cache=st.session_state.sidebar_use_cache,
                **contract_details_for_main_data
            )
        progress_bar.progress(50, text="Datos principales descargados. Calculando niveles...")

        if df_total_utc_data is None or df_total_utc_data.empty:
            st.error(f"No se pudieron obtener datos principales para {st.session_state.symbol_loaded}. Verifica los parámetros y la conexión.")
            st.stop()
        st.session_state.all_data_utc = df_total_utc_data

        target_date_for_initial_levels = st.session_state.start_date_obj
        with st.spinner(f"Calculando niveles iniciales para {target_date_for_initial_levels.strftime('%Y-%m-%d')}..."):
            contract_details_for_levels = {
                 'primary_exchange': st.session_state.sidebar_primary_exchange if st.session_state.sidebar_primary_exchange and st.session_state.sidebar_sec_type == 'STK' else None
            }
            needs_reconnect_for_levels = False
            if not data_manager_instance.ib.isConnected():
                logger.warning("IB no conectado para get_levels_data, intentando reconectar...")
                if data_manager_instance.connect_ib():
                    needs_reconnect_for_levels = True
                else:
                     st.error("Fallo la reconexión a IB para calcular niveles. Usando datos de caché si es posible.")
            
            df_prev_day_for_levels, df_premarket_for_levels = data_manager_instance.get_levels_data(
                target_date=target_date_for_initial_levels,
                symbol=st.session_state.symbol_loaded,
                sec_type=st.session_state.sidebar_sec_type, 
                exchange=st.session_state.sidebar_exchange,
                currency=st.session_state.sidebar_currency,
                use_cache=st.session_state.sidebar_use_cache,
                **contract_details_for_levels
            )
            
            static_lvls = {
                **data_manager_instance.calculate_pdh_pdl(df_prev_day_for_levels),
                **data_manager_instance.calculate_pmh_pml(df_premarket_for_levels)
            }
            st.session_state['static_levels'] = {k: v for k, v in static_lvls.items() if v is not None and pd.notna(v)}
            
            if needs_reconnect_for_levels:
                data_manager_instance.disconnect_ib()

        progress_bar.progress(100, text="Carga de datos y niveles completada.")
        time.sleep(0.5)
        progress_bar.empty()

    except Exception as e_load_data:
        st.error(f"Error durante la carga de datos o cálculo de niveles: {e_load_data}")
        logger.error(f"Error en bloque de carga de datos: {e_load_data}", exc_info=True)
        st.session_state.replay_started = False
        if data_manager_instance and ib_connection_successful:
            data_manager_instance.disconnect_ib()
        st.stop()

    try:
        st.session_state.executor = ExecutionSimulator(
            initial_capital=INITIAL_CAPITAL,
            commission_per_trade=COMMISSION_PER_TRADE,
            slippage_points=SLIPPAGE_POINTS,
            leverage=st.session_state.selected_leverage
        )
        
        initial_equity_timestamp = None
        if st.session_state.all_data_utc is not None and not st.session_state.all_data_utc.empty:
            start_day_for_equity_utc = st.session_state.tz_handler.market_tz.localize(
                datetime.datetime.combine(st.session_state.download_start_date, datetime.time.min)
            ).astimezone(pytz.utc)
            first_valid_ts_in_data = st.session_state.all_data_utc[st.session_state.all_data_utc.index >= start_day_for_equity_utc].index.min()
            if pd.notna(first_valid_ts_in_data):
                 initial_equity_timestamp = first_valid_ts_in_data
            else:
                 initial_equity_timestamp = st.session_state.all_data_utc.index.min()
        
        if initial_equity_timestamp is None:
            initial_equity_timestamp = pd.Timestamp.now(tz='UTC')
            logger.warning("No se pudieron determinar datos para el timestamp inicial del equity, usando now().")

        initial_equity_point = (initial_equity_timestamp, st.session_state.executor.initial_capital)
        st.session_state.equity_history = [initial_equity_point]
        st.session_state.executor.equity_history = [initial_equity_point]

        st.session_state.replay_started = True

    except Exception as e_init_strat_exec:
        st.error(f"Error inicializando el Simulador de Ejecución: {e_init_strat_exec}")
        logger.error(f"Error en inicialización de ejecutor: {e_init_strat_exec}", exc_info=True)
        st.session_state.replay_started = False
    finally:
        if data_manager_instance and ib_connection_successful:
            data_manager_instance.disconnect_ib()
        
        if st.session_state.replay_started:
            logger.info("--- Carga/Reinicio Completado Exitosamente ---")
            st.success(f"Backtest listo para: {st.session_state.symbol_loaded} | Replay desde: {st.session_state.start_date_obj.strftime('%Y-%m-%d')}")
            st.rerun()
        else:
            logger.warning("--- Carga/Reinicio Falló ---")

# --- Bucle de Reproducción (con la llamada al Agent Core) ---
if st.session_state.replay_started:
    data_needs_reprocessing = (
        st.session_state.get('processed_tz') != st.session_state.display_tz or
        st.session_state.get('last_processed_day') != st.session_state.start_date_obj or
        st.session_state.get('symbol_processed') != st.session_state.symbol_loaded or
        st.session_state.df_replay_display is None
    )

    if data_needs_reprocessing:
        logger.info(f"Reprocesando datos para fecha de replay: {st.session_state.start_date_obj.strftime('%Y-%m-%d')} y Timezone Display: {st.session_state.display_tz}...")
        
        handle_signal_request(historical_data=None, current_levels=None, reset_strategy=True)

        st.session_state.processed_tz = st.session_state.display_tz
        st.session_state.last_processed_day = st.session_state.start_date_obj
        st.session_state.symbol_processed = st.session_state.symbol_loaded
        st.session_state.current_index = 0
        st.session_state.markers = []
        st.session_state.last_trade_sl_display = None
        st.session_state.last_trade_tp_display = None
        st.session_state.autoplaying = False
        
        if st.session_state.executor:
            if st.session_state.executor.position != 'FLAT':
                 logger.warning(f"Reprocesando datos con posición {st.session_state.executor.position} aún abierta.")
            st.session_state.executor.position = 'FLAT'
            st.session_state.executor.current_size = 0
            st.session_state.executor.entry_price = Decimal('NaN')
            st.session_state.executor.entry_time = None
            st.session_state.executor.has_sl_tp = False
            st.session_state.executor.sl_price = Decimal('NaN')
            st.session_state.executor.tp_price = Decimal('NaN')
            st.session_state.executor.candles_in_current_trade = 0

        dm_for_levels_recalc = None
        try:
            logger.info(f"Recalculando static_levels para el nuevo día: {st.session_state.start_date_obj}")
            dm_for_levels_recalc = DataManager()
            contract_details_for_lvl_recalc = {
                'primary_exchange': st.session_state.sidebar_primary_exchange if st.session_state.sidebar_primary_exchange and st.session_state.sidebar_sec_type == 'STK' else None
            }
            
            df_prev_day_recalc, df_premarket_recalc = dm_for_levels_recalc.get_levels_data(
                target_date=st.session_state.start_date_obj,
                symbol=st.session_state.symbol_loaded, 
                sec_type=st.session_state.sidebar_sec_type,
                exchange=st.session_state.sidebar_exchange,
                currency=st.session_state.sidebar_currency,
                use_cache=st.session_state.sidebar_use_cache, 
                **contract_details_for_lvl_recalc
            )
            new_static_lvls = {
                **dm_for_levels_recalc.calculate_pdh_pdl(df_prev_day_recalc),
                **dm_for_levels_recalc.calculate_pmh_pml(df_premarket_recalc)
            }
            st.session_state['static_levels'] = {k: v for k, v in new_static_lvls.items() if v is not None and pd.notna(v)}
            logger.info(f"Nuevos static_levels para {st.session_state.start_date_obj}: {st.session_state['static_levels']}")

        except Exception as e_recalc_levels:
            st.error(f"Error recalculando niveles estáticos para el nuevo día: {e_recalc_levels}")
            logger.error(f"Error recalculando static_levels: {e_recalc_levels}", exc_info=True)
        finally:
            if dm_for_levels_recalc and dm_for_levels_recalc.ib.isConnected():
                dm_for_levels_recalc.disconnect_ib()


        df_total_utc_for_processing = st.session_state.all_data_utc
        target_display_tz_str = st.session_state.display_tz
        date_to_replay = st.session_state.start_date_obj

        if df_total_utc_for_processing is None or df_total_utc_for_processing.empty:
            st.error("Error Interno: No hay datos base (all_data_utc) para procesar.")
            st.session_state.replay_started = False
            st.stop()
        
        try:
            st.session_state.tz_handler.set_display_timezone(target_display_tz_str)

            start_replay_dt_display_tz = st.session_state.tz_handler.display_tz.localize(
                datetime.datetime.combine(date_to_replay, datetime.time.min)
            )
            end_replay_dt_display_tz = st.session_state.tz_handler.display_tz.localize(
                datetime.datetime.combine(date_to_replay, datetime.time.max)
            )
            start_replay_dt_utc = start_replay_dt_display_tz.astimezone(pytz.utc)
            end_replay_dt_utc = end_replay_dt_display_tz.astimezone(pytz.utc)

            df_context_utc_slice = df_total_utc_for_processing[df_total_utc_for_processing.index < start_replay_dt_utc]
            df_replay_utc_slice = df_total_utc_for_processing[
                (df_total_utc_for_processing.index >= start_replay_dt_utc) &
                (df_total_utc_for_processing.index <= end_replay_dt_utc)
            ]
            
            if df_replay_utc_slice.empty:
                st.warning(f"No hay datos para la fecha de replay seleccionada: {date_to_replay.strftime('%Y-%m-%d')}.")
                empty_cols = ['open', 'high', 'low', 'close', 'volume']
                empty_idx_display_tz = pd.to_datetime([]).tz_localize('UTC').tz_convert(target_display_tz_str)
                st.session_state.df_replay_display = pd.DataFrame(columns=empty_cols, index=empty_idx_display_tz)
                if st.session_state.df_context_display is None:
                    st.session_state.df_context_display = pd.DataFrame(columns=empty_cols, index=empty_idx_display_tz)
                st.session_state['opening_levels'] = {}
            else:
                rth_for_display = st.session_state.get('rth_toggle_main_data', use_rth_default)
                
                if not df_context_utc_slice.empty:
                    st.session_state.df_context_display = apply_timezone_fixes(
                        st.session_state, df_context_utc_slice, target_display_tz_str, use_rth=rth_for_display
                    )
                else:
                    empty_cols = ['open','high','low','close','volume']
                    empty_idx_display_tz = pd.to_datetime([]).tz_localize('UTC').tz_convert(target_display_tz_str)
                    st.session_state.df_context_display = pd.DataFrame(columns=empty_cols, index=empty_idx_display_tz)

                st.session_state.df_replay_display = apply_timezone_fixes(
                    st.session_state, df_replay_utc_slice, target_display_tz_str, use_rth=rth_for_display
                )

                st.session_state['opening_levels'] = {}
                if st.session_state.df_replay_display is not None and not st.session_state.df_replay_display.empty:
                    try:
                        market_open_time_obj = st.session_state.tz_handler.market_open_time
                        or_duration_config_minutes = 5 
                        
                        first_timestamp_of_replay_day = st.session_state.df_replay_display.index[0]
                        current_date_for_or_calc = first_timestamp_of_replay_day.date()
                        current_display_tz_for_or = st.session_state.tz_handler.display_tz

                        or_start_dt_local = current_display_tz_for_or.localize(
                            datetime.datetime.combine(current_date_for_or_calc, market_open_time_obj)
                        )
                        or_end_dt_local = or_start_dt_local + datetime.timedelta(minutes=or_duration_config_minutes)
                        
                        opening_range_candles = st.session_state.df_replay_display[
                            (st.session_state.df_replay_display.index >= or_start_dt_local) &
                            (st.session_state.df_replay_display.index < or_end_dt_local)
                        ]
                        
                        if not opening_range_candles.empty:
                            or_high = pd.to_numeric(opening_range_candles['high'], errors='coerce').max()
                            or_low = pd.to_numeric(opening_range_candles['low'], errors='coerce').min()
                            if pd.notna(or_high) and pd.notna(or_low):
                                st.session_state['opening_levels'] = {'ORH': or_high, 'ORL': or_low}
                                logger.info(f"ORH/ORL para {current_date_for_or_calc.strftime('%Y-%m-%d')}: H={or_high:.4f}, L={or_low:.4f}")
                        else:
                            logger.warning(f"No se encontraron velas para calcular ORH/ORL para el día {current_date_for_or_calc.strftime('%Y-%m-%d')}")
                    except Exception as e_or_calc:
                        logger.error(f"Error recalculando ORH/ORL para el día de replay: {e_or_calc}", exc_info=True)
            
            shape_ctx = st.session_state.df_context_display.shape if st.session_state.df_context_display is not None else "None"
            shape_rpl = st.session_state.df_replay_display.shape if st.session_state.df_replay_display is not None else "None"
            logger.debug(f"df_context_display shape: {shape_ctx}, df_replay_display shape: {shape_rpl}")

        except Exception as e_process_daily_data:
            st.error(f"Error procesando datos diarios para replay: {e_process_daily_data}")
            logger.error(f"Error crítico en reprocesamiento de datos diarios: {e_process_daily_data}", exc_info=True)
            st.session_state.replay_started = False
            st.stop()
        st.rerun()
    df_context_for_display = st.session_state.get('df_context_display')
    df_replay_for_display = st.session_state.get('df_replay_display')

    if df_replay_for_display is None:
        st.warning("Esperando datos de replay para procesar...")
    elif df_replay_for_display.empty:
        st.info(f"No hay datos para mostrar para la fecha {st.session_state.start_date_obj.strftime('%Y-%m-%d')}.")
        if st.session_state.start_date_obj < st.session_state.download_end_date:
            if st.button(">>> Avanzar al Siguiente Día con Datos >>>", key="btn_next_day_main_empty", use_container_width=True, disabled=st.session_state.get('autoplaying', False)):
                go_to_next_day()
                st.rerun() 
        else:
             st.success(f"Fin del rango de backtest: Se alcanzó la fecha final de descarga ({st.session_state.download_end_date.strftime('%Y-%m-%d')}).")
    else:
        current_idx_replay = st.session_state.current_index
        current_candle_for_info = None
        historical_data_for_strategy_signal = pd.DataFrame()

        if 0 <= current_idx_replay < len(df_replay_for_display):
            current_candle_for_info = df_replay_for_display.iloc[current_idx_replay]
            current_ts_for_info = current_candle_for_info.name

            try:
                current_ts_for_info_utc = current_ts_for_info.tz_convert(pytz.utc)
                if st.session_state.all_data_utc is not None:
                    historical_data_for_strategy_signal = st.session_state.all_data_utc[
                        st.session_state.all_data_utc.index <= current_ts_for_info_utc
                    ].copy()
                else:
                    historical_data_for_strategy_signal = pd.DataFrame()
            except Exception as e_hist_utc_build:
                logger.error(f"Error construyendo el histórico UTC para la estrategia en TS {current_ts_for_info}: {e_hist_utc_build}", exc_info=True)
                historical_data_for_strategy_signal = pd.DataFrame()
            
            vela_num_str_log = f"Vela Replay {current_idx_replay + 1}/{len(df_replay_for_display)}"
            try:
                time_str_log_display = st.session_state.tz_handler.format_timestamp_for_display(current_ts_for_info, include_date=True)
            except: time_str_log_display = "ErrorFormatoTiempo"
            try:
                o_log=current_candle_for_info.get('open','N/A'); h_log=current_candle_for_info.get('high','N/A'); l_log=current_candle_for_info.get('low','N/A'); c_log=current_candle_for_info.get('close','N/A')
                o_str_log=f"{float(o_log):.4f}" if isinstance(o_log,(int,float,Decimal)) and pd.notna(o_log) else str(o_log)
                h_str_log=f"{float(h_log):.4f}" if isinstance(h_log,(int,float,Decimal)) and pd.notna(h_log) else str(h_log)
                l_str_log=f"{float(l_log):.4f}" if isinstance(l_log,(int,float,Decimal)) and pd.notna(l_log) else str(l_log)
                c_str_log=f"{float(c_log):.4f}" if isinstance(c_log,(int,float,Decimal)) and pd.notna(c_log) else str(c_log)
                ohlc_str_log=f"O:{o_str_log} H:{h_str_log} L:{l_str_log} C:{c_str_log}"
            except Exception as e_ohlc_log_format: ohlc_str_log=f"ErrorFormatoOHLC: {e_ohlc_log_format}"
            
            abs_idx_log = len(historical_data_for_strategy_signal) - 1 if not historical_data_for_strategy_signal.empty else -1
            if not st.session_state.get('autoplaying', False) or (st.session_state.get('autoplaying', False) and current_idx_replay % 5 == 0) :
                logger.info(f"--- {vela_num_str_log} ({time_str_log_display}) | AbsIdxUTC:{abs_idx_log} | {ohlc_str_log} ---")

            signal_output = 'HOLD'
            if not historical_data_for_strategy_signal.empty:
                 try:
                     levels_for_signal_call = {
                         **st.session_state.get('static_levels', {}),
                         **st.session_state.get('opening_levels', {})
                     }
                     levels_for_signal_call_cleaned = {
                         k: v for k, v in levels_for_signal_call.items() if v is not None and pd.notna(v)
                     }
                     
                     signal_output = handle_signal_request(
                         historical_data=historical_data_for_strategy_signal,
                         current_levels=levels_for_signal_call_cleaned
                     )
                     
                     if isinstance(signal_output, dict):
                         log_emas = signal_output.get('emas',{})
                         emas_str_log = ", ".join([f"{k.split('_')[1]}={v:.4f}" for k,v in log_emas.items() if pd.notna(v)]) if log_emas else "NoEMAs"
                         logger.info(f"Señal del Agente: {signal_output.get('type','?')} (Nivel:{signal_output.get('level','')}, SL:{signal_output.get('sl_price',0):.4f}, TP:{signal_output.get('tp1_price',0):.4f}, EMAs:[{emas_str_log}])")
                     elif signal_output != 'HOLD':
                         logger.info(f"Señal del Agente: {signal_output}")
                 except Exception as e_strat_call:
                     logger.error(f"Error al llamar al Agent Core 'handle_signal_request': {e_strat_call}", exc_info=True)
                     signal_output = 'HOLD'
            
            execution_result_event = None
            if st.session_state.executor:
                 try:
                    position_before_signal = st.session_state.executor.position
                    sl_before_signal_dec = st.session_state.executor.sl_price
                    tp_before_signal_dec = st.session_state.executor.tp_price

                    execution_result_event = st.session_state.executor.process_signal(signal_output, current_candle_for_info)
                    
                    if execution_result_event and 'marker' in execution_result_event and execution_result_event['marker']:
                        new_trade_marker = execution_result_event['marker']
                        is_duplicate_marker = False
                        if st.session_state.markers:
                            last_m = st.session_state.markers[-1]
                            try:
                                if abs(pd.Timestamp(last_m.get('time')).timestamp() - pd.Timestamp(new_trade_marker.get('time')).timestamp()) < 1 and \
                                     last_m.get('shape') == new_trade_marker.get('shape') and \
                                    last_m.get('text') == new_trade_marker.get('text'):
                                    is_duplicate_marker = True
                            except Exception as e_dup_check: logger.warning(f"Error comparando marcadores para duplicados: {e_dup_check}")
                        
                        if not is_duplicate_marker:
                            st.session_state.markers.append(new_trade_marker)
                    
                    if position_before_signal != 'FLAT' and st.session_state.executor.position == 'FLAT':
                        # --- INICIO CORRECCIÓN: Lógica para pausar Autoplay ---
                        if st.session_state.get('autoplaying', False):
                            st.session_state.autoplaying = False
                            st.toast("Autoplay pausado por cierre de operación.")
                            logger.info("Autoplay pausado por cierre de operación.")
                        # --- FIN CORRECCIÓN ---

                        if not sl_before_signal_dec.is_nan():
                            try: st.session_state.last_trade_sl_display = float(sl_before_signal_dec)
                            except: st.session_state.last_trade_sl_display = None
                        else: st.session_state.last_trade_sl_display = None

                        if not tp_before_signal_dec.is_nan():
                            try: st.session_state.last_trade_tp_display = float(tp_before_signal_dec)
                            except: st.session_state.last_trade_tp_display = None
                        else: st.session_state.last_trade_tp_display = None
                        logger.debug(f"Operación cerrada. Guardando SL para display: {st.session_state.last_trade_sl_display}, TP: {st.session_state.last_trade_tp_display}")


                    if position_before_signal == 'FLAT' and st.session_state.executor.position != 'FLAT':
                        logger.debug("Nueva operación abierta. Limpiando SL/TP de la última operación cerrada para display.")
                        st.session_state.last_trade_sl_display = None
                        st.session_state.last_trade_tp_display = None

                    st.session_state.closed_pnl = st.session_state.executor.closed_pnl_total
                    st.session_state.last_equity = st.session_state.executor.last_equity
                    st.session_state.equity_history = st.session_state.executor.get_equity_history()
                    
                    if not st.session_state.get('autoplaying', False) or (st.session_state.get('autoplaying', False) and current_idx_replay % 5 == 0) :
                        logger.info(f"Estado Cuenta Post-Exec: Equity=${st.session_state.last_equity:,.2f} | PNL Cerrado=${st.session_state.closed_pnl:,.2f} | Posición Actual={st.session_state.executor.position} | Velas en Trade: {st.session_state.executor.candles_in_current_trade}")

                 except Exception as e_exec_call:
                     logger.error(f"Error durante process_signal de ExecutionSimulator: {e_exec_call}", exc_info=True)
                     st.error(f"Error en Simulación de Ejecución: {e_exec_call}")
            else:
                 st.error("Error Crítico: Simulador de ejecución (Executor) no encontrado en session_state.")
        
        elif current_idx_replay >= len(df_replay_for_display) and len(df_replay_for_display) > 0 :
            logger.info(f"Fin de datos para el día actual: {st.session_state.start_date_obj.strftime('%Y-%m-%d')}.")
            if st.session_state.get('autoplaying', False): 
                st.session_state.autoplaying = False
                st.toast("Autoplay: Fin de los datos para este día.")
                logger.info("Autoplay detenido al final del día.")
            
            if st.session_state.start_date_obj >= st.session_state.download_end_date:
                st.success(f"Fin del rango de backtest: Se alcanzó la fecha final de descarga ({st.session_state.download_end_date.strftime('%Y-%m-%d')}).")
            else:
                st.info(f"Fin del día {st.session_state.start_date_obj.strftime('%Y-%m-%d')}. Presiona 'Next Day' o selecciona otra fecha.")
            st.session_state.current_index = len(df_replay_for_display) - 1
        else:
             logger.warning(f"Índice de replay ({current_idx_replay}) inválido o df_replay_for_display vacío. Resetando a 0.")
             st.session_state.current_index = 0
        
        # --- El resto del código para mostrar métricas y el gráfico no necesita cambios ---
        # ... (pegar el resto del archivo app.py desde la línea `col_metrics_area, col_chart_area = st.columns([1, 3])` hasta el final)
        col_metrics_area, col_chart_area = st.columns([1, 3])

        with col_metrics_area:
            st.subheader("Estado de Cuenta")
            if st.session_state.executor:
                executor_instance = st.session_state.executor
                current_equity_display = st.session_state.last_equity
                closed_pnl_display = st.session_state.closed_pnl
                
                st.metric("Capital Inicial", f"${INITIAL_CAPITAL:,.2f}")
                delta_equity = current_equity_display - Decimal(str(INITIAL_CAPITAL))
                st.metric("Equity Actual", f"${current_equity_display:,.2f}", delta=f"{delta_equity:,.2f}", delta_color="normal")
                
                position_status_display = executor_instance.position
                entry_info_display = ""
                if executor_instance.position != 'FLAT' and not executor_instance.entry_price.is_nan():
                    try:
                        entry_info_display = f"{executor_instance.current_size} @ {float(executor_instance.entry_price):.4f}"
                    except: entry_info_display = f"{executor_instance.current_size} @ ?"
                st.metric("Posición Actual", f"{position_status_display} {entry_info_display}".strip())
                
                st.metric("PNL Cerrado Acum.", f"${closed_pnl_display:,.2f}")
                try:
                    open_pnl_display = current_equity_display - Decimal(str(INITIAL_CAPITAL)) - closed_pnl_display
                    st.metric("PNL Abierto Actual", f"${open_pnl_display:,.2f}")
                except Exception: st.metric("PNL Abierto Actual", "Error Cálculo")

                if executor_instance.has_sl_tp and executor_instance.position != 'FLAT':
                    st.markdown("**Órdenes Activas en Posición**")
                    try:
                        sl_f_disp = float(executor_instance.sl_price) if not executor_instance.sl_price.is_nan() else None
                        tp_f_disp = float(executor_instance.tp_price) if not executor_instance.tp_price.is_nan() else None
                        if sl_f_disp is not None and abs(sl_f_disp) > 1e-9: st.metric("Stop Loss Activo", f"${sl_f_disp:.4f}")
                        if tp_f_disp is not None and abs(tp_f_disp) > 1e-9: st.metric("Take Profit Activo", f"${tp_f_disp:.4f}")
                    except Exception: st.text("SL/TP Inválidos para mostrar")
            
            st.divider()
            st.subheader("Métricas Totales del Backtest")
            if st.session_state.executor:
                closed_trades_for_metrics = st.session_state.executor.get_closed_trades()
                equity_history_for_metrics = st.session_state.get('equity_history', [])
                
                if closed_trades_for_metrics or len(equity_history_for_metrics) > 1:
                    metrics_calculated = calculate_performance_metrics(
                        closed_trades_for_metrics,
                        float(INITIAL_CAPITAL),
                        equity_history_for_metrics
                    )
                    st.session_state.performance_metrics = metrics_calculated

                    m_total_trades = metrics_calculated.get("Total Trades", 0)
                    m_win_rate = metrics_calculated.get('Win Rate (%)', 0.0)
                    m_profit_factor = metrics_calculated.get('Profit Factor', 'N/A')
                    m_max_drawdown = metrics_calculated.get('Max Drawdown (%)', None)

                    st.metric("Trades Totales Cerrados", f"{m_total_trades}")
                    st.metric("Win Rate (%)", f"{m_win_rate:.2f}%")
                    
                    pf_display_val = "N/A"
                    if isinstance(m_profit_factor, (int, float, Decimal)) and not pd.isna(m_profit_factor):
                        pf_display_val = "Inf" if m_profit_factor == float('inf') else f"{float(m_profit_factor):.2f}"
                    st.metric("Profit Factor", pf_display_val)
                    
                    st.metric("Max Drawdown (%)", f"{m_max_drawdown:.2f}%" if m_max_drawdown is not None else "N/A")
                else:
                    st.text("No hay trades cerrados o historial de equity suficiente para calcular métricas.")
            else: st.text("Simulador no disponible para métricas.")


        with col_chart_area:
            df_for_chart_concat_list = []
            if df_context_for_display is not None and not df_context_for_display.empty:
                df_for_chart_concat_list.append(df_context_for_display)
            
            if df_replay_for_display is not None and not df_replay_for_display.empty:
                end_slice_idx_chart = min(current_idx_replay + 1, len(df_replay_for_display))
                if end_slice_idx_chart > 0:
                    df_for_chart_concat_list.append(df_replay_for_display.iloc[0:end_slice_idx_chart])
            
            full_slice_data_for_chart = pd.DataFrame()
            if df_for_chart_concat_list:
                try:
                    full_slice_data_for_chart = pd.concat(df_for_chart_concat_list)
                    if not full_slice_data_for_chart.index.is_unique:
                        full_slice_data_for_chart = full_slice_data_for_chart[~full_slice_data_for_chart.index.duplicated(keep='last')]
                        full_slice_data_for_chart = full_slice_data_for_chart.sort_index()
                except Exception as e_concat_chart:
                    st.error(f"Error concatenando DataFrames para el gráfico: {e_concat_chart}")
                    logger.error(f"Error concat DFs chart: {e_concat_chart}", exc_info=True)
                    full_slice_data_for_chart = pd.DataFrame()

            df_to_render_on_chart = pd.DataFrame()
            if not full_slice_data_for_chart.empty:
                sub_cols_chart_view = st.columns([1, 3])
                with sub_cols_chart_view[0]:
                    mostrar_todo_chart = st.checkbox(
                        "Mostrar todo el historial cargado",
                        value=st.session_state.mostrar_todo,
                        key="toggle_chart_view_mode",
                        on_change=actualizar_mostrar_todo,
                        disabled=st.session_state.get('autoplaying', False)
                    )
                with sub_cols_chart_view[1]:
                    if not mostrar_todo_chart:
                        velas_visibles_input = st.number_input(
                            "Velas Visibles Atrás:",
                            min_value=5, max_value=5000,
                            value=st.session_state.velas_atras,
                            step=10,
                            key="velas_atras_input_widget",
                            on_change=actualizar_velas_atras, args=('velas_atras_input_widget',),
                            disabled=st.session_state.get('autoplaying', False)
                        )
                
                if df_replay_for_display is not None and not df_replay_for_display.empty:
                    max_idx_replay_day = len(df_replay_for_display) - 1
                    
                    nav_cols = st.columns([2, 2, 3, 2, 2]) 
                    
                    with nav_cols[0]: 
                        st.button("<< Vela Anterior", key="btn_prev_candle_top", use_container_width=True,
                                  disabled=(current_idx_replay <= 0 or st.session_state.get('autoplaying', False)), 
                                  on_click=prev_candle)
                    with nav_cols[1]: 
                        st.button("Cerrar Posición", key="btn_manual_close", use_container_width=True,
                                  on_click=close_position_manually,
                                  disabled=(st.session_state.executor is None or st.session_state.executor.position == 'FLAT' or st.session_state.get('autoplaying', False)),
                                  help="Cierra la posición abierta actual al precio de cierre de esta vela.")
                    
                    with nav_cols[2]: 
                        time_str_nav_display = "N/A"
                        if current_candle_for_info is not None:
                            try:
                                time_str_nav_display = st.session_state.tz_handler.format_timestamp_for_display(
                                    current_candle_for_info.name, include_date=False
                                )
                            except: time_str_nav_display = "ErrTiempo"
                        st.caption(f"<div style='text-align:center; width:100%; margin-top: 5px;'>Vela {current_idx_replay + 1}/{max_idx_replay_day + 1}<br>{time_str_nav_display}</div>", unsafe_allow_html=True)

                    with nav_cols[3]: 
                         st.button("Vela Siguiente >>", key="btn_next_candle_top_moved", use_container_width=True,
                                  disabled=(current_idx_replay >= max_idx_replay_day or st.session_state.get('autoplaying', False)), 
                                  on_click=next_candle)
                    
                    with nav_cols[4]: 
                        if st.session_state.get('autoplaying', False):
                            st.button("❚❚ Pausar", key="btn_pause_autoplay", use_container_width=True, on_click=toggle_autoplay, type="primary")
                        else:
                            st.button("▶ Play", key="btn_play_autoplay", use_container_width=True, on_click=toggle_autoplay,
                                      disabled=(current_idx_replay >= max_idx_replay_day))


                if mostrar_todo_chart:
                    df_to_render_on_chart = full_slice_data_for_chart
                else:
                    num_velas_para_slice_render = st.session_state.velas_atras + VELAS_HACIA_ADELANTE + 1
                    df_to_render_on_chart = full_slice_data_for_chart.iloc[-num_velas_para_slice_render:]
            
                if df_to_render_on_chart is not None and not df_to_render_on_chart.empty:
                    levels_to_draw_on_chart = {
                        **st.session_state.get('static_levels',{}),
                        **st.session_state.get('opening_levels',{})
                    }
                    levels_to_draw_on_chart = {k:v for k,v in levels_to_draw_on_chart.items() if v is not None and pd.notna(v)}

                    active_sl_tp_lines = {}
                    persistent_sl_tp_lines = {}

                    if st.session_state.executor:
                        if st.session_state.executor.position != 'FLAT' and st.session_state.executor.has_sl_tp:
                            try:
                                sl_p_active = float(st.session_state.executor.sl_price) if not st.session_state.executor.sl_price.is_nan() else None
                                tp_p_active = float(st.session_state.executor.tp_price) if not st.session_state.executor.tp_price.is_nan() else None
                                if sl_p_active is not None and abs(sl_p_active) > 1e-9: active_sl_tp_lines['SL'] = sl_p_active
                                if tp_p_active is not None and abs(tp_p_active) > 1e-9: active_sl_tp_lines['TP'] = tp_p_active
                                logger.debug(f"Operación activa. SL chart: {sl_p_active}, TP chart: {tp_p_active}")
                            except Exception as e:
                                logger.warning(f"Error al obtener SL/TP de operación activa para el gráfico: {e}")
                        elif st.session_state.executor.position == 'FLAT':
                            if st.session_state.last_trade_sl_display is not None:
                                persistent_sl_tp_lines['SL'] = st.session_state.last_trade_sl_display
                            if st.session_state.last_trade_tp_display is not None:
                                persistent_sl_tp_lines['TP'] = st.session_state.last_trade_tp_display
                            if persistent_sl_tp_lines:
                                 logger.debug(f"Sin operación activa. Mostrando SL_Last: {persistent_sl_tp_lines.get('SL')}, TP_Last: {persistent_sl_tp_lines.get('TP')}")

                    final_sl_tp_for_chart = active_sl_tp_lines if active_sl_tp_lines else persistent_sl_tp_lines
                    combined_levels_for_chart_render = {**levels_to_draw_on_chart, **final_sl_tp_for_chart}
                    chart_render_key = f"trading_chart_{st.session_state.symbol_loaded}_{st.session_state.timeframe_loaded}_{current_idx_replay}_{len(df_to_render_on_chart)}_{st.session_state.mostrar_todo}_{st.session_state.start_date_obj.strftime('%Y%m%d')}_{bool(final_sl_tp_for_chart)}"


                    if 'prepare_chart_data_and_options' in globals() and 'renderLightweightCharts' in globals():
                        try:
                            chart_options_dict, series_data_list = prepare_chart_data_and_options(
                                df_display=df_to_render_on_chart,
                                markers=st.session_state.get('markers',[]),
                                chart_height=650, 
                                static_levels=combined_levels_for_chart_render,
                                timezone_str=st.session_state.display_tz,
                                fit_content=True,
                                level_ranges=LEVEL_RANGES.copy()
                            )
                            if series_data_list:
                                with st.container():
                                    renderLightweightCharts(
                                        charts=[{"chart": chart_options_dict, "series": series_data_list}],
                                        key=chart_render_key
                                    )
                            else:
                                logger.warning("prepare_chart_data_and_options devolvió series_data vacía. No se renderizará el gráfico.")
                                st.warning("No hay datos de series para mostrar en el gráfico en este momento.")
            
                        except Exception as e_render_chart:
                            st.error(f"Error al renderizar el gráfico: {e_render_chart}")
                            logger.error(f"Error renderizado gráfico: {e_render_chart}", exc_info=True)
                else:
                    st.info("No hay datos de velas para mostrar en el gráfico en este momento.")


            next_day_button_disabled = st.session_state.start_date_obj >= st.session_state.download_end_date or st.session_state.get('autoplaying', False)
            if st.button(
                ">>> Avanzar al Siguiente Día con Datos >>>",
                key="btn_next_day_bottom",
                use_container_width=True,
                on_click=go_to_next_day,
                disabled=next_day_button_disabled
            ):
                st.rerun()

    if st.session_state.replay_started:
        st.divider()
        st.subheader("Historial Total de Trades Cerrados del Backtest")
        if st.session_state.executor:
            all_closed_trades = st.session_state.executor.get_closed_trades()
            if all_closed_trades:
                try:
                    trades_df_for_table = pd.DataFrame(all_closed_trades)
                    trades_df_display_table = trades_df_for_table.copy()

                    for time_col in ['entry_time', 'exit_time']:
                         if time_col in trades_df_display_table.columns:
                            trades_df_display_table[time_col] = trades_df_display_table[time_col].apply(
                                lambda ts: st.session_state.tz_handler.format_timestamp_for_display(ts, include_date=True) if pd.notna(ts) else 'N/A'
                            )
                    
                    if 'direction' not in trades_df_display_table.columns: trades_df_display_table['direction'] = 'N/A'
                    if 'size' not in trades_df_display_table.columns: trades_df_display_table['size'] = 0
                    if 'exit_reason' not in trades_df_display_table.columns: trades_df_display_table['exit_reason'] = 'N/A'
                    if 'level_triggered' not in trades_df_display_table.columns:
                        trades_df_display_table['level_triggered'] = 'N/A'
                    else:
                        trades_df_display_table['level_triggered'] = trades_df_display_table['level_triggered'].fillna('N/A')


                    ema_cols_to_add_for_display = []
                    if 'entry_emas' in trades_df_display_table.columns:
                        valid_emas_list = [x if isinstance(x, dict) else {} for x in trades_df_display_table['entry_emas']]
                        if any(valid_emas_list):
                            emas_normalized_df = pd.json_normalize(valid_emas_list).fillna(np.nan)
                            emas_normalized_df.columns = [f'{col}_entryEMA' for col in emas_normalized_df.columns]
                            
                            trades_df_display_table = pd.concat(
                                [trades_df_display_table.drop(columns=['entry_emas'], errors='ignore'), emas_normalized_df],
                                axis=1
                            )
                            ema_cols_to_add_for_display = list(emas_normalized_df.columns)
                            logger.debug(f"Columnas de EMA añadidas para visualización en tabla: {ema_cols_to_add_for_display}")
                    
                    cols_base_order = ['entry_time', 'exit_time', 'direction', 'size', 'entry_price', 'exit_price', 'level_triggered', 'exit_reason']
                    cols_sl_tp_at_entry = ['sl_at_entry', 'tp_at_entry']
                    cols_financial_order = ['pnl_net', 'commission', 'pnl_gross']
                    
                    final_cols_for_table_display = cols_base_order + \
                                                   cols_sl_tp_at_entry + \
                                                   ema_cols_to_add_for_display + \
                                                   cols_financial_order
                    
                    actual_cols_to_show_in_table = [col for col in final_cols_for_table_display if col in trades_df_display_table.columns]

                    dynamic_column_config = {
                        "entry_time": st.column_config.TextColumn("Hora Entrada"),
                        "exit_time": st.column_config.TextColumn("Hora Salida"),
                        "direction": st.column_config.TextColumn("Dirección"),
                        "entry_price": st.column_config.NumberColumn("Entrada $", format="%.4f"),
                        "exit_price": st.column_config.NumberColumn("Salida $", format="%.4f"),
                        "pnl_net": st.column_config.NumberColumn("PNL Neto ($)", format="%.2f"),
                        "commission": st.column_config.NumberColumn("Comisión ($)", format="%.2f"),
                        "pnl_gross": st.column_config.NumberColumn("PNL Bruto ($)", format="%.2f"),
                        "size": st.column_config.NumberColumn("Tamaño", format="%d"),
                        "exit_reason": st.column_config.TextColumn("Razón Salida"),
                        "sl_at_entry": st.column_config.NumberColumn("SL Entrada $", format="%.4f"),
                        "tp_at_entry": st.column_config.NumberColumn("TP Entrada $", format="%.4f"),
                        "level_triggered": st.column_config.TextColumn("Nivel Activado")
                    }
                    for ema_col_name_table in ema_cols_to_add_for_display:
                        if ema_col_name_table in trades_df_display_table.columns:
                            readable_ema_label = ema_col_name_table.replace('_entryEMA', '').replace('EMA_', 'EMA ') + " (Entrada)"
                            dynamic_column_config[ema_col_name_table] = st.column_config.NumberColumn(readable_ema_label, format="%.4f")
                    
                    st.dataframe(
                        trades_df_display_table[actual_cols_to_show_in_table],
                        use_container_width=True,
                        height=300,
                        column_config=dynamic_column_config,
                        hide_index=True
                    )
                except Exception as e_table_render:
                    st.error(f"Error mostrando la tabla de trades cerrados: {e_table_render}")
                    logger.error(f"Error generando/renderizando DataFrame de trades para visualización: {e_table_render}", exc_info=True)
            else:
                st.text("Aún no hay trades cerrados en el historial de este backtest.")
        else:
            st.text("Simulador de ejecución no inicializado. No se pueden mostrar trades.")

    # --- LÓGICA DE REPRODUCCIÓN AUTOMÁTICA ---
    if st.session_state.get('autoplaying', False) and \
       st.session_state.df_replay_display is not None and not st.session_state.df_replay_display.empty:
        
        max_idx_replay_day = len(st.session_state.df_replay_display) - 1
        if st.session_state.current_index < max_idx_replay_day:
            time.sleep(st.session_state.get('autoplay_speed', 0.5)) 
            st.session_state.current_index += 1
            st.rerun()
        else:
            st.session_state.autoplaying = False
            st.toast("Autoplay: Fin de los datos para este día.")
            logger.info("Autoplay detenido al final del día.")
            st.rerun() 
    # --- FIN LÓGICA DE REPRODUCCIÓN AUTOMÁTICA ---


elif not st.session_state.get('replay_started', False):
    st.info("Bienvenido al Backtester Visual. Configura los parámetros en la barra lateral y presiona 'Cargar Datos e Iniciar Replay'.")
