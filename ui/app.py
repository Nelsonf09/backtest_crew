# ui/app.py
"""
Aplicación Streamlit para el Backtesting Player Visual.
Versión optimizada con cálculo de métricas eficiente y uso de st.empty()
para una actualización fluida de la interfaz. CORREGIDO: Lógica de actualización de métricas y apalancamiento.
"""
# --- PARCHE ASYNCIO (Necesario para crewai en Streamlit) ---
import nest_asyncio
nest_asyncio.apply()
# --- FIN PARCHE ASYNCIO ---

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-7s - %(name)-25s:%(lineno)3d - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

import streamlit as st
import pandas as pd
import time
from pathlib import Path
import datetime
from decimal import Decimal
import sys
import pytz

# Añadir la ruta raíz del proyecto para permitir importaciones de otros módulos
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Importaciones de módulos del proyecto
import config
from agent_core.main import handle_signal_request
from agent_core.data_manager import DataManager
from ui.streamlit_chart import prepare_chart_data_and_options
from streamlit_lightweight_charts import renderLightweightCharts
from agent_core.execution import ExecutionSimulator, TradeState
from agent_core.metrics import calculate_performance_metrics
from shared.timezone_handler import TimezoneHandler, apply_timezone_fixes
from shared.fsm import FSM, AppState

# --- CONFIGURACIÓN DE PÁGINA Y ESTADO DE SESIÓN ---
st.set_page_config(layout="wide", page_title="Backtester Visual (FSM)", initial_sidebar_state="expanded")

def initialize_session_state():
    """Inicializa todos los valores necesarios en st.session_state si no existen."""
    if 'app_fsm' in st.session_state:
        return

    logger.info("--- INICIALIZANDO ESTADO DE SESIÓN POR PRIMERA VEZ ---")
    _today = datetime.date.today()
    _default_download_start = _today - datetime.timedelta(days=getattr(config, 'DEFAULT_DOWNLOAD_RANGE_DAYS', 5))
    _default_start_date_obj_candidate = _today - datetime.timedelta(days=getattr(config, 'DEFAULT_START_DATE_OFFSET_DAYS', 1))
    _initial_start_date_obj = max(_default_download_start, min(_default_start_date_obj_candidate, _today))

    # FSM y datos principales
    st.session_state.app_fsm = FSM(AppState.CONFIGURING)
    st.session_state.all_data_utc = pd.DataFrame()
    st.session_state.df_context_display = pd.DataFrame()
    st.session_state.df_replay_display = pd.DataFrame()
    st.session_state.current_index = 0
    st.session_state.markers = []
    st.session_state.equity_history = []
    st.session_state.performance_metrics = {}
    st.session_state.static_levels = {}
    st.session_state.opening_levels = {}
    st.session_state.session_trades = []
    st.session_state.last_closed_trade_levels = {}
    
    # Parámetros de la UI
    st.session_state.ui_symbol = config.DEFAULT_SYMBOL
    st.session_state.ui_timeframe = config.DEFAULT_TIMEFRAME
    st.session_state.ui_download_start = _default_download_start
    st.session_state.ui_download_end = _today
    st.session_state.ui_replay_start_date = _initial_start_date_obj
    st.session_state.ui_display_tz = config.DEFAULT_DISPLAY_TZ
    st.session_state.ui_leverage = getattr(config, 'DEFAULT_LEVERAGE', 5)
    st.session_state.ui_ema_filter = "Desactivado"
    st.session_state.ui_velas_atras = 100
    st.session_state.ui_mostrar_todo = False
    st.session_state.ui_autoplay_speed = 0.5
    st.session_state.ui_sec_type = config.DEFAULT_SEC_TYPE
    st.session_state.ui_exchange = config.DEFAULT_EXCHANGE
    st.session_state.ui_currency = config.DEFAULT_CURRENCY
    st.session_state.ui_primary_exchange = config.DEFAULT_PRIMARY_EXCHANGE
    st.session_state.ui_use_rth = config.USE_RTH
    st.session_state.ui_what_to_show = config.WHAT_TO_SHOW
    st.session_state.ui_use_cache = config.ENABLE_CACHING
    
    # Instancias de clases
    st.session_state.tz_handler = TimezoneHandler(default_display_tz_str=config.DEFAULT_DISPLAY_TZ)
    st.session_state.data_manager = DataManager()
    st.session_state.executor = ExecutionSimulator(initial_capital=config.INITIAL_CAPITAL, leverage=st.session_state.ui_leverage)

initialize_session_state()

# --- FUNCIONES DE LÓGICA Y CONTROL ---

def load_data_and_start_replay():
    """Inicia la transición para cargar datos."""
    st.session_state.app_fsm.transition_to(AppState.LOADING_DATA)

def process_loading_state():
    """Ejecuta la lógica de carga de datos."""
    st.session_state.data_manager = DataManager()
    dm = st.session_state.data_manager

    st.session_state.session_trades = []
    st.session_state.current_index = 0
    st.session_state.markers = []
    st.session_state.equity_history = []
    st.session_state.last_closed_trade_levels = {}
    st.session_state.performance_metrics = {}
    handle_signal_request(None, None, reset_strategy=True)
    st.session_state.executor = ExecutionSimulator(
        initial_capital=config.INITIAL_CAPITAL,
        commission_per_trade=config.COMMISSION_PER_TRADE,
        slippage_points=config.SLIPPAGE_POINTS,
        leverage=st.session_state.ui_leverage
    )

    with st.spinner("Conectando a Interactive Brokers..."):
        if not dm.connect_ib():
            st.error("Fallo la conexión a Interactive Brokers. Verifica que TWS/Gateway esté activo.")
            st.session_state.app_fsm.transition_to(AppState.ERROR)
            return

    try:
        with st.spinner(f"Cargando datos para {st.session_state.ui_symbol}..."):
            st.session_state.all_data_utc = dm.get_main_data(
                symbol=st.session_state.ui_symbol, timeframe=st.session_state.ui_timeframe,
                sec_type=st.session_state.ui_sec_type, exchange=st.session_state.ui_exchange,
                currency=st.session_state.ui_currency, rth=st.session_state.ui_use_rth,
                what_to_show=st.session_state.ui_what_to_show,
                download_start_date=st.session_state.ui_download_start,
                download_end_date=st.session_state.ui_download_end,
                use_cache=st.session_state.ui_use_cache,
                primary_exchange=st.session_state.ui_primary_exchange
            )
        if st.session_state.all_data_utc.empty:
            st.warning("No se pudieron obtener datos. Verifique los parámetros o el rango de fechas.")
            st.session_state.app_fsm.transition_to(AppState.CONFIGURING)
            return

        st.session_state.app_fsm.transition_to(AppState.READY)
    except Exception as e:
        st.error(f"Error durante la carga de datos: {e}")
        st.session_state.app_fsm.transition_to(AppState.ERROR)
    finally:
        dm.disconnect_ib()

def process_and_prepare_daily_data():
    """Filtra y prepara los datos para el día de replay seleccionado."""
    date_to_replay = st.session_state.ui_replay_start_date
    df_total_utc = st.session_state.all_data_utc
    
    handle_signal_request(None, None, reset_strategy=True)
    
    dm = st.session_state.data_manager
    df_prev, df_pm = dm.get_levels_data(
        target_date=date_to_replay, symbol=st.session_state.ui_symbol,
        sec_type=st.session_state.ui_sec_type, exchange=st.session_state.ui_exchange,
        currency=st.session_state.ui_currency, use_cache=True,
        primary_exchange=st.session_state.ui_primary_exchange
    )
    st.session_state.static_levels = {**dm.calculate_pdh_pdl(df_prev), **dm.calculate_pmh_pml(df_pm)}
    
    tz_handler = st.session_state.tz_handler
    tz_handler.set_display_timezone(st.session_state.ui_display_tz)
    start_utc = tz_handler.display_tz.localize(datetime.datetime.combine(date_to_replay, datetime.time.min)).astimezone(pytz.utc)
    end_utc = tz_handler.display_tz.localize(datetime.datetime.combine(date_to_replay, datetime.time.max)).astimezone(pytz.utc)

    df_context_utc = df_total_utc[df_total_utc.index < start_utc]
    df_replay_utc = df_total_utc[(df_total_utc.index >= start_utc) & (df_total_utc.index <= end_utc)]
    
    st.session_state.df_context_display = apply_timezone_fixes(st.session_state, df_context_utc, st.session_state.ui_display_tz, st.session_state.ui_use_rth)
    st.session_state.df_replay_display = apply_timezone_fixes(st.session_state, df_replay_utc, st.session_state.ui_display_tz, st.session_state.ui_use_rth)

    if not st.session_state.df_replay_display.empty:
        try:
            or_start_time = st.session_state.tz_handler.market_open_time
            or_start_dt = st.session_state.df_replay_display.index[0].replace(hour=or_start_time.hour, minute=or_start_time.minute, second=0, microsecond=0)
            or_end_dt = or_start_dt + datetime.timedelta(minutes=5)
            or_candles = st.session_state.df_replay_display[(st.session_state.df_replay_display.index >= or_start_dt) & (st.session_state.df_replay_display.index < or_end_dt)]
            if not or_candles.empty:
                st.session_state.opening_levels = {'ORH': or_candles['high'].max(), 'ORL': or_candles['low'].min()}
        except Exception as e:
            logger.error(f"Error calculando ORH/ORL: {e}")
            st.session_state.opening_levels = {}
    
    st.session_state.current_index = 0
    st.session_state.markers = []
    st.session_state.executor.closed_trades = []
    st.session_state.last_closed_trade_levels = {}
    st.session_state.performance_metrics = {}
    st.session_state.app_fsm.transition_to(AppState.PAUSED)

def close_position_manually():
    """Cierra una posición manualmente y recalcula las métricas."""
    executor = st.session_state.executor
    if executor.account_fsm.is_in_state(TradeState.ACTIVE):
        idx = st.session_state.current_index
        current_candle = st.session_state.df_replay_display.iloc[idx]
        
        close_signal = 'SELL' if executor.current_trade.direction == 'LONG' else 'BUY'
        result = executor.process_signal(close_signal, current_candle)
        
        if result and 'marker' in result:
            result['marker']['text'] = f"MANUAL_CLOSE ({result['marker']['text']})"
            result['marker']['color'] = 'orange'
            st.session_state.markers.append(result['marker'])
            if 'trade' in result:
                st.session_state.session_trades.append(result['trade'])
                trade_info = result['trade']
                st.session_state.last_closed_trade_levels = {
                    'SL': trade_info.get('sl_at_entry'),
                    'TP': trade_info.get('tp_at_entry')
                }
                st.session_state.performance_metrics = calculate_performance_metrics(
                    st.session_state.session_trades,
                    config.INITIAL_CAPITAL,
                    st.session_state.executor.get_equity_history()
                )
        
        st.toast("Posición cerrada manualmente.")

def go_to_next_day():
    """Encuentra el siguiente día con datos y cambia el replay a esa fecha."""
    current_date = st.session_state.ui_replay_start_date
    all_dates = sorted(list(set(st.session_state.all_data_utc.index.tz_convert(st.session_state.ui_display_tz).date)))
    
    next_day = None
    for date in all_dates:
        if date > current_date:
            next_day = date
            break
            
    if next_day:
        st.session_state.ui_replay_start_date = next_day
        st.session_state.app_fsm.transition_to(AppState.READY)
    else:
        st.toast("No hay más días con datos en el rango cargado.")

# --- RENDERIZADO DE LA UI ---

if st.session_state.ui_replay_start_date < st.session_state.ui_download_start:
    st.session_state.ui_replay_start_date = st.session_state.ui_download_start
if st.session_state.ui_replay_start_date > st.session_state.ui_download_end:
    st.session_state.ui_replay_start_date = st.session_state.ui_download_end

with st.sidebar:
    st.title("Configuración")
    
    with st.expander("1. Instrumento y Rango", expanded=True):
        st.selectbox("Símbolo", options=config.STOCK_SYMBOLS_LIST, key="ui_symbol")
        st.selectbox("Tipo de Activo", options=['STK', 'FUT', 'IND', 'FOREX', 'CFD', 'CRYPTO'], key="ui_sec_type")
        st.text_input("Exchange", key="ui_exchange")
        st.text_input("Moneda", key="ui_currency")
        
        primary_exchange_options = ['', 'NASDAQ', 'NYSE', 'ARCA', 'AMEX', 'ISLAND', 'BATS', 'IEX']
        st.selectbox("Primary Exch.", options=primary_exchange_options, key="ui_primary_exchange", help="Necesario para desambiguar acciones en SMART.")
        
        st.selectbox("Timeframe", options=['1 min', '3 mins', '5 mins', '15 mins'], key="ui_timeframe")
        st.date_input("Inicio Descarga", key="ui_download_start")
        st.date_input("Fin Descarga", key="ui_download_end")
        st.date_input("Inicio Replay", key="ui_replay_start_date", min_value=st.session_state.ui_download_start, max_value=st.session_state.ui_download_end)
        st.toggle("Usar solo RTH", key="ui_use_rth")
        st.selectbox("Fuente de Datos Velas", options=['TRADES', 'MIDPOINT', 'BID', 'ASK'], key="ui_what_to_show")
        st.toggle("Usar Caché de Datos", key="ui_use_cache")

    with st.expander("2. Estrategia y Ejecución", expanded=True):
        st.selectbox("Filtro EMA (OBR)", options=["Desactivado", "Moderado", "Fuerte"], key="ui_ema_filter")
        st.select_slider("Apalancamiento", options=[1, 5, 10, 20, 50, 100], key="ui_leverage")
        st.selectbox("Timezone Gráfico", options=pytz.common_timezones, key="ui_display_tz")

    st.button("Cargar / Recargar Datos", on_click=load_data_and_start_replay, use_container_width=True, type="primary")
    st.divider()
    st.subheader("Controles de Replay")
    st.number_input("Velocidad Autoplay (s)", min_value=0.1, max_value=5.0, step=0.1, key="ui_autoplay_speed")

if st.session_state.app_fsm.is_in_state(AppState.CONFIGURING):
    st.info("Ajusta los parámetros en la barra lateral y presiona 'Cargar / Recargar Datos'.")

elif st.session_state.app_fsm.is_in_state(AppState.LOADING_DATA):
    process_loading_state()
    st.rerun()

elif st.session_state.app_fsm.is_in_state(AppState.ERROR):
    st.error("Ha ocurrido un error. Por favor, revisa la configuración y vuelve a intentarlo.")
    if st.button("Reconfigurar"):
        st.session_state.app_fsm.transition_to(AppState.CONFIGURING)
        st.rerun()

elif st.session_state.app_fsm.is_in_state(AppState.READY):
    process_and_prepare_daily_data()
    st.rerun()

elif st.session_state.app_fsm.state in [AppState.PAUSED, AppState.REPLAYING, AppState.FINISHED]:
    df_replay = st.session_state.df_replay_display
    if df_replay.empty:
        st.warning(f"No hay datos de replay para {st.session_state.ui_replay_start_date.strftime('%Y-%m-%d')}.")
        st.button(">>> Avanzar al Siguiente Día con Datos >>>", use_container_width=True, on_click=go_to_next_day)
        st.stop()

    idx = st.session_state.current_index
    current_candle = df_replay.iloc[idx]
    executor = st.session_state.executor
    
    if not st.session_state.performance_metrics:
        st.session_state.performance_metrics = calculate_performance_metrics(
            st.session_state.session_trades, 
            config.INITIAL_CAPITAL, 
            executor.get_equity_history()
        )

    if not st.session_state.app_fsm.is_in_state(AppState.FINISHED):
        historical_data_utc = st.session_state.all_data_utc[st.session_state.all_data_utc.index <= current_candle.name.tz_convert(pytz.utc)]
        levels = {**st.session_state.static_levels, **st.session_state.opening_levels}
        signal = handle_signal_request(historical_data_utc, {k:v for k,v in levels.items() if pd.notna(v)})

        # --- LÓGICA DE APALANCAMIENTO MEJORADA ---
        # Actualizar el apalancamiento solo si hay una señal de apertura y ha cambiado.
        if isinstance(signal, dict) and signal.get('type') in ['BUY', 'SELL']:
            if executor.leverage != st.session_state.ui_leverage:
                executor.set_leverage(st.session_state.ui_leverage)
                st.toast(f"Apalancamiento ajustado a {st.session_state.ui_leverage}x para la nueva operación.", icon="⚙️")
        # --- FIN DE LA MEJORA ---

        result = executor.process_signal(signal, current_candle)
        
        if result and 'marker' in result:
            st.session_state.markers.append(result['marker'])
            
            if 'trade' in result:
                st.session_state.session_trades.append(result['trade'])
                trade_info = result['trade']
                st.session_state.last_closed_trade_levels = {
                    'SL': trade_info.get('sl_at_entry'),
                    'TP': trade_info.get('tp_at_entry')
                }
                st.session_state.performance_metrics = calculate_performance_metrics(
                    st.session_state.session_trades,
                    config.INITIAL_CAPITAL,
                    executor.get_equity_history()
                )
                if st.session_state.app_fsm.is_in_state(AppState.REPLAYING):
                    st.session_state.app_fsm.transition_to(AppState.PAUSED)
                    st.toast("Autoplay pausado por cierre de operación.")
            
            if executor.account_fsm.state == TradeState.ACTIVE and 'trade' not in result:
                st.session_state.last_closed_trade_levels = {}

    col_metrics, col_chart = st.columns([1, 3])
    
    with col_metrics:
        account_status_placeholder = st.empty()
        performance_metrics_placeholder = st.empty()

    with account_status_placeholder.container():
        st.subheader("Estado de Cuenta")
        equity = executor.get_equity(float(current_candle['close']))
        initial_capital_dec = Decimal(str(config.INITIAL_CAPITAL))
        st.metric("Equity Actual", f"${equity:,.2f}", delta=f"{(equity - initial_capital_dec):,.2f}")
        st.metric("Posición", f"{executor.account_fsm.state}")
        if executor.current_trade:
            trade = executor.current_trade
            st.metric("Entrada", f"{trade.direction} {trade.size} @ {trade.entry_price:.4f}")
            st.metric("SL Activo", f"{trade.current_sl_price:.4f}")
            st.metric("TP Activo", f"{trade.current_tp_price:.4f}")

    with performance_metrics_placeholder.container():
        st.divider()
        st.subheader("Métricas del Backtest")
        metrics = st.session_state.performance_metrics
        st.metric("Trades Totales", metrics.get("Total Trades", 0))
        st.metric("Win Rate (%)", f"{metrics.get('Win Rate (%)', 0.0):.2f}%")
        pf_value = metrics.get('Profit Factor', 'N/A')
        st.metric("Profit Factor", f"{pf_value}")
        max_dd = metrics.get('Max Drawdown (%)', 0.0)
        st.metric("Max Drawdown (%)", f"{max_dd:.2f}%")
        
    with col_chart:
        is_playing = st.session_state.app_fsm.is_in_state(AppState.REPLAYING)
        is_finished = st.session_state.app_fsm.is_in_state(AppState.FINISHED) or idx >= len(df_replay) - 1
        is_trade_active = executor.account_fsm.is_in_state(TradeState.ACTIVE)
        
        view_cols = st.columns([2,3])
        view_cols[0].checkbox("Mostrar todo el historial", key="ui_mostrar_todo", disabled=is_playing)
        view_cols[1].number_input("Velas Visibles Atrás:", min_value=10, max_value=5000, step=10, key="ui_velas_atras", disabled=is_playing or st.session_state.ui_mostrar_todo)

        nav_cols = st.columns([2, 2, 3, 2, 2, 2])
        nav_cols[0].button("<< Prev", use_container_width=True, disabled=is_playing or idx <= 0, on_click=lambda: setattr(st.session_state, 'current_index', st.session_state.current_index - 1))
        
        play_button_text = "Play ▶" if not is_playing else "Pause ❚❚"
        nav_cols[1].button(play_button_text, use_container_width=True, disabled=is_finished, on_click=lambda: st.session_state.app_fsm.transition_to(AppState.REPLAYING if not is_playing else AppState.PAUSED))
        
        time_str_nav_display = "N/A"
        if current_candle is not None:
            try:
                time_str_nav_display = st.session_state.tz_handler.format_timestamp_for_display(current_candle.name, include_date=False)
            except: time_str_nav_display = "ErrTiempo"
        nav_cols[2].caption(f"<div style='text-align:center; width:100%; margin-top: 5px;'>Vela {idx + 1}/{len(df_replay)}<br>{time_str_nav_display}</div>", unsafe_allow_html=True)

        nav_cols[3].button("Next >>", use_container_width=True, disabled=is_playing or is_finished, on_click=lambda: setattr(st.session_state, 'current_index', st.session_state.current_index + 1))
        nav_cols[4].button("Cerrar Pos.", use_container_width=True, disabled=is_playing or not is_trade_active, on_click=close_position_manually)
        nav_cols[5].button("Reset Day", use_container_width=True, disabled=is_playing, on_click=lambda: st.session_state.app_fsm.transition_to(AppState.READY))

        if st.session_state.ui_mostrar_todo:
            df_to_render = pd.concat([st.session_state.df_context_display, df_replay.iloc[:idx+1]])
        else:
            full_slice = pd.concat([st.session_state.df_context_display, df_replay.iloc[:idx+1]])
            df_to_render = full_slice.iloc[-st.session_state.ui_velas_atras:]
            
        levels_for_chart = {**levels}
        if executor.current_trade:
            levels_for_chart['SL'] = float(executor.current_trade.current_sl_price)
            levels_for_chart['TP'] = float(executor.current_trade.current_tp_price)
        else:
            levels_for_chart.update(st.session_state.last_closed_trade_levels)
            
        chart_options, series_data = prepare_chart_data_and_options(
            df_display=df_to_render, markers=st.session_state.markers,
            static_levels=levels_for_chart,
            timezone_str=st.session_state.ui_display_tz,
            level_ranges=config.LEVEL_RANGES
        )
        if series_data:
            renderLightweightCharts([{"chart": chart_options, "series": series_data}], key=f"chart_{idx}_{st.session_state.ui_mostrar_todo}")

        if not st.session_state.app_fsm.is_in_state(AppState.REPLAYING):
            st.button(">>> Avanzar al Siguiente Día con Datos >>>", use_container_width=True, on_click=go_to_next_day)

    # --- LÓGICA DE TRANSICIÓN DE ESTADO Y TABLA DE TRADES ---
    
    if st.session_state.app_fsm.is_in_state(AppState.PAUSED) and idx >= len(df_replay) - 1:
        st.session_state.app_fsm.transition_to(AppState.FINISHED)
        st.rerun()

    if st.session_state.app_fsm.is_in_state(AppState.REPLAYING):
        if idx < len(df_replay) - 1:
            time.sleep(st.session_state.ui_autoplay_speed)
            st.session_state.current_index += 1
            st.rerun()
        else:
            st.session_state.app_fsm.transition_to(AppState.FINISHED)
            st.rerun()

    if st.session_state.app_fsm.is_in_state(AppState.FINISHED):
        st.success("Replay del día finalizado.")
    
    st.divider()
    st.subheader("Historial de Trades de la Sesión")
    if st.session_state.session_trades:
        trades_df = pd.DataFrame(st.session_state.session_trades)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(trades_df[['entry_time', 'exit_time', 'direction', 'size', 'entry_price', 'exit_price', 'pnl_net', 'exit_reason', 'level_triggered']], use_container_width=True)
    else:
        st.text("No hay operaciones cerradas en esta sesión.")
