# ui/app.py
"""
Aplicación Streamlit para el Backtesting Player Visual.
Versión COMPLETA con MODO DUAL: Replay visual paso a paso y backtest global rápido con Numba.
"""
import nest_asyncio
nest_asyncio.apply()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-7s - %(name)-25s:%(lineno)3d - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import datetime
from decimal import Decimal
import sys
import pytz

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import config
from agent_core.main import handle_signal_request
from agent_core.data_manager import DataManager
from ui.streamlit_chart import prepare_chart_data_and_options
from streamlit_lightweight_charts import renderLightweightCharts
from agent_core.execution import ExecutionSimulator, TradeState
from agent_core.metrics import calculate_performance_metrics
from shared.timezone_handler import TimezoneHandler, apply_timezone_fixes
from shared.fsm import FSM, AppState
from strategies.vectorized_obr import run_fast_backtest

st.set_page_config(layout="wide", page_title="Backtester Híbrido (IA)", initial_sidebar_state="expanded")

def initialize_session_state():
    if 'app_fsm' in st.session_state: return

    logger.info("--- INICIALIZANDO ESTADO DE SESIÓN ---")
    _today = datetime.date.today()
    _start = _today - datetime.timedelta(days=getattr(config, 'DEFAULT_DOWNLOAD_RANGE_DAYS', 5))
    
    st.session_state.app_fsm = FSM(AppState.CONFIGURING)
    st.session_state.all_data_utc = pd.DataFrame()
    st.session_state.session_trades = []
    st.session_state.performance_metrics = {}
    st.session_state.global_equity_history = pd.DataFrame()
    st.session_state.df_context_display = pd.DataFrame()
    st.session_state.df_replay_display = pd.DataFrame()
    st.session_state.current_index = 0
    st.session_state.markers = []
    st.session_state.static_levels = {}
    st.session_state.opening_levels = {}
    st.session_state.last_closed_trade_levels = {}

    st.session_state.ui_symbol = config.DEFAULT_SYMBOL
    st.session_state.ui_timeframe = config.DEFAULT_TIMEFRAME
    st.session_state.ui_download_start = _start
    st.session_state.ui_download_end = _today
    st.session_state.ui_replay_start_date = max(_start, _today - datetime.timedelta(days=1))
    st.session_state.ui_display_tz = config.DEFAULT_DISPLAY_TZ
    st.session_state.ui_leverage = getattr(config, 'DEFAULT_LEVERAGE', 5)
    st.session_state.ui_backtest_mode = "Visual (Paso a Paso)"
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
    
    st.session_state.data_manager = DataManager()
    st.session_state.executor = ExecutionSimulator(initial_capital=config.INITIAL_CAPITAL, leverage=st.session_state.ui_leverage)
    st.session_state.tz_handler = TimezoneHandler(default_display_tz_str=st.session_state.ui_display_tz)


initialize_session_state()

def execute_backtest():
    if st.session_state.ui_backtest_mode == "Visual (Paso a Paso)":
        st.session_state.app_fsm.transition_to(AppState.LOADING_DATA)
    else:
        st.session_state.app_fsm.transition_to(AppState.GLOBAL_BACKTESTING)

def process_global_backtesting():
    dm = st.session_state.data_manager 
    tz_handler = TimezoneHandler(default_display_tz_str=st.session_state.ui_display_tz)
    
    with st.spinner("Conectando a IB..."):
        if not dm.connect_ib():
            st.error("Fallo la conexión a IB."); st.session_state.app_fsm.transition_to(AppState.ERROR); return

    try:
        with st.spinner(f"Cargando datos para {st.session_state.ui_symbol}..."):
            df_full = dm.get_main_data(
                symbol=st.session_state.ui_symbol, timeframe='1 min', 
                sec_type=st.session_state.ui_sec_type, exchange=st.session_state.ui_exchange, 
                currency=st.session_state.ui_currency, rth=st.session_state.ui_use_rth, 
                what_to_show=st.session_state.ui_what_to_show,
                download_start_date=st.session_state.ui_download_start,
                download_end_date=st.session_state.ui_download_end,
                use_cache=st.session_state.ui_use_cache, primary_exchange=st.session_state.ui_primary_exchange
            )
        if df_full.empty:
            st.warning("No se obtuvieron datos."); st.session_state.app_fsm.transition_to(AppState.CONFIGURING); return
        
        df_full = df_full.tz_convert(tz_handler.display_tz)
        all_trades, full_equity_history = [], []
        unique_dates = sorted(df_full.index.normalize().unique())
        progress_bar = st.progress(0, text="Procesando días...")

        for i, date in enumerate(unique_dates):
            date_obj = date.date()
            progress_bar.progress((i + 1) / len(unique_dates), text=f"Procesando {date_obj.strftime('%Y-%m-%d')}...")

            df_prev, df_pm = dm.get_levels_data(
                target_date=date_obj, symbol=st.session_state.ui_symbol, 
                sec_type=st.session_state.ui_sec_type, exchange=st.session_state.ui_exchange, 
                currency=st.session_state.ui_currency, use_cache=st.session_state.ui_use_cache, 
                primary_exchange=st.session_state.ui_primary_exchange
            )
            levels = {**dm.calculate_pdh_pdl(df_prev), **dm.calculate_pmh_pml(df_pm)}
            df_day = df_full[df_full.index.date == date_obj]
            if df_day.empty: continue

            orh, orl = np.nan, np.nan
            try:
                or_start_time = tz_handler.market_open_time
                or_start_dt = df_day.index[0].replace(hour=or_start_time.hour, minute=or_start_time.minute, second=0, microsecond=0)
                or_end_dt = or_start_dt + datetime.timedelta(minutes=5)
                or_candles = df_day[(df_day.index >= or_start_dt) & (df_day.index < or_end_dt)]
                if not or_candles.empty:
                    orh, orl = or_candles['high'].max(), or_candles['low'].min()
                    levels['ORH'], levels['ORL'] = orh, orl
            except Exception as e:
                logger.warning(f"No se pudo calcular ORH/ORL para {date_obj}: {e}")

            levels_data = [(k, v) for k, v in levels.items() if pd.notna(v)]
            if not levels_data: continue
            
            level_name_map = {'ORH': 1, 'ORL': 2, 'PDH': 3, 'PDL': 4, 'PMH': 5, 'PML': 6}
            level_names_numeric = np.array([level_name_map.get(item[0], 0) for item in levels_data], dtype=np.int32)
            
            level_prices = np.array([item[1] for item in levels_data], dtype=np.float64)
            level_dirs = np.array([1 if 'H' in name else -1 for name, _ in levels_data], dtype=np.int8)
            level_ranges = np.array([config.LEVEL_RANGES.get(name, 0.0) for name, _ in levels_data], dtype=np.float64)

            ema_mode_map = {"Desactivado": 0, "Moderado": 1, "Fuerte": 2}
            ema_filter_mode = ema_mode_map.get(st.session_state.ui_ema_filter, 0)

            trades, equity_hist = run_fast_backtest(
                df_day['open'].to_numpy(dtype=np.float64), df_day['high'].to_numpy(dtype=np.float64),
                df_day['low'].to_numpy(dtype=np.float64), df_day['close'].to_numpy(dtype=np.float64),
                df_day.index.values.astype(np.int64) // 10**9, 
                df_day.index.hour.to_numpy(dtype=np.int8),
                df_day.index.minute.to_numpy(dtype=np.int8),
                level_names_numeric, level_prices, level_dirs,
                max_retest_candles=15, risk_reward_ratio=2.0,
                sl_default_lookback=3, sl_reduced_lookback=2,
                level_ranges=level_ranges, orh_price=orh, orl_price=orl,
                initial_capital=config.INITIAL_CAPITAL, commission_per_side=config.COMMISSION_PER_TRADE,
                leverage=float(st.session_state.ui_leverage),
                ema_filter_mode=ema_filter_mode,
                ema_periods=np.array([9, 21, 50], dtype=np.float64)
            )
            
            if trades.shape[0] > 0: all_trades.append(pd.DataFrame(trades, columns=["entry_time", "exit_time", "direction", "size", "entry_price", "exit_price", "pnl_net", "exit_reason"]))
            if equity_hist.shape[0] > 0: full_equity_history.append(pd.DataFrame(equity_hist, columns=["time", "equity"]))

        progress_bar.empty()
        st.session_state.session_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        st.session_state.global_equity_history = pd.concat(full_equity_history, ignore_index=True) if full_equity_history else pd.DataFrame()
        st.session_state.app_fsm.transition_to(AppState.SHOWING_RESULTS)
    except Exception as e:
        st.error(f"Error en backtest global: {e}"); st.session_state.app_fsm.transition_to(AppState.ERROR)
    finally:
        dm.disconnect_ib()

def render_global_results():
    st.header("Resultados del Backtest Global")
    trades_df = st.session_state.session_trades
    equity_df = st.session_state.global_equity_history
    
    if trades_df.empty:
        st.warning("La estrategia no generó ninguna operación en el período seleccionado."); return

    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'], unit='s', utc=True).dt.tz_convert(st.session_state.ui_display_tz).dt.strftime('%Y-%m-%d %H:%M:%S')
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'], unit='s', utc=True).dt.tz_convert(st.session_state.ui_display_tz).dt.strftime('%Y-%m-%d %H:%M:%S')

    metrics = calculate_performance_metrics(trades_df.to_dict('records'), config.INITIAL_CAPITAL, equity_df.values.tolist())
    cols = st.columns(4)
    ganancia_neta_val = metrics['Ganancia Neta Total ($)']
    cols[0].metric("Ganancia Neta Total", f"${ganancia_neta_val:,.2f}", f"{metrics['Ganancia Neta Total (%)']:.2f}%")
    cols[1].metric("Trades Totales", metrics['Total Trades'])
    cols[2].metric("Win Rate", f"{metrics['Win Rate (%)']:.2f}%")
    cols[3].metric("Profit Factor", f"{metrics['Profit Factor']}")
    
    st.subheader("Curva de Equity")
    if not equity_df.empty:
        equity_df['time'] = pd.to_datetime(equity_df['time'], unit='s', utc=True)
        equity_df = equity_df.set_index('time')
        st.line_chart(equity_df['equity'])
    
    st.subheader("Historial de Operaciones")
    st.dataframe(trades_df)

def process_loading_state_visual():
    dm = st.session_state.data_manager
    st.session_state.session_trades, st.session_state.performance_metrics = [], {}
    st.session_state.executor = ExecutionSimulator(initial_capital=config.INITIAL_CAPITAL, leverage=st.session_state.ui_leverage)
    with st.spinner("Conectando a IB..."):
        if not dm.connect_ib(): st.error("Fallo la conexión a IB."); st.session_state.app_fsm.transition_to(AppState.ERROR); return
    try:
        lookback_days = datetime.timedelta(days=7)
        adjusted_download_start = st.session_state.ui_download_start - lookback_days
        
        with st.spinner(f"Cargando datos para {st.session_state.ui_symbol}..."):
            st.session_state.all_data_utc = dm.get_main_data(
                symbol=st.session_state.ui_symbol, timeframe=st.session_state.ui_timeframe,
                sec_type=st.session_state.ui_sec_type, exchange=st.session_state.ui_exchange,
                currency=st.session_state.ui_currency, rth=st.session_state.ui_use_rth,
                what_to_show=st.session_state.ui_what_to_show,
                download_start_date=adjusted_download_start,
                download_end_date=st.session_state.ui_download_end,
                use_cache=st.session_state.ui_use_cache, primary_exchange=st.session_state.ui_primary_exchange
            )
        if st.session_state.all_data_utc.empty: st.warning("No se obtuvieron datos."); st.session_state.app_fsm.transition_to(AppState.CONFIGURING); return
        st.session_state.app_fsm.transition_to(AppState.READY)
    except Exception as e: st.error(f"Error cargando datos: {e}"); st.session_state.app_fsm.transition_to(AppState.ERROR)

def process_and_prepare_daily_data_visual():
    date_to_replay = st.session_state.ui_replay_start_date
    handle_signal_request(None, None, reset_strategy=True)
    dm = st.session_state.data_manager
    
    with st.spinner("Obteniendo niveles del día..."):
        if not dm.connect_ib():
            st.error("Fallo la conexión a IB para obtener los niveles del día.")
            st.session_state.app_fsm.transition_to(AppState.ERROR)
            return
        try:
            df_prev, df_pm = dm.get_levels_data(
                target_date=date_to_replay, symbol=st.session_state.ui_symbol, 
                sec_type=st.session_state.ui_sec_type, exchange=st.session_state.ui_exchange, 
                currency=st.session_state.ui_currency, use_cache=st.session_state.ui_use_cache, 
                primary_exchange=st.session_state.ui_primary_exchange
            )
            st.session_state.static_levels = {**dm.calculate_pdh_pdl(df_prev), **dm.calculate_pmh_pml(df_pm)}
        finally:
            dm.disconnect_ib()

    st.session_state.tz_handler.set_display_timezone(st.session_state.ui_display_tz)
    start_utc = st.session_state.tz_handler.display_tz.localize(datetime.datetime.combine(date_to_replay, datetime.time.min)).astimezone(pytz.utc)
    end_utc = st.session_state.tz_handler.display_tz.localize(datetime.datetime.combine(date_to_replay, datetime.time.max)).astimezone(pytz.utc)
    
    df_context_utc = st.session_state.all_data_utc[st.session_state.all_data_utc.index < start_utc]
    df_replay_utc = st.session_state.all_data_utc[(st.session_state.all_data_utc.index >= start_utc) & (st.session_state.all_data_utc.index <= end_utc)]
    
    st.session_state.df_context_display = apply_timezone_fixes(st.session_state, df_context_utc, st.session_state.ui_display_tz, st.session_state.ui_use_rth)
    st.session_state.df_replay_display = apply_timezone_fixes(st.session_state, df_replay_utc, st.session_state.ui_display_tz, st.session_state.ui_use_rth)
    
    if not st.session_state.df_replay_display.empty:
        try:
            or_start_time = st.session_state.tz_handler.market_open_time
            or_start_dt = st.session_state.df_replay_display.index[0].replace(hour=or_start_time.hour, minute=or_start_time.minute, second=0, microsecond=0)
            or_end_dt = or_start_dt + datetime.timedelta(minutes=5)
            or_candles = st.session_state.df_replay_display[(st.session_state.df_replay_display.index >= or_start_dt) & (st.session_state.df_replay_display.index < or_end_dt)]
            if not or_candles.empty: st.session_state.opening_levels = {'ORH': or_candles['high'].max(), 'ORL': or_candles['low'].min()}
            else: st.session_state.opening_levels = {}
        except Exception as e: logger.error(f"Error calculando ORH/ORL: {e}"); st.session_state.opening_levels = {}
    
    st.session_state.current_index = 0
    st.session_state.markers, st.session_state.executor.closed_trades, st.session_state.last_closed_trade_levels = [], [], {}
    st.session_state.app_fsm.transition_to(AppState.PAUSED)


def close_position_manually_visual():
    executor = st.session_state.executor
    if executor.account_fsm.is_in_state(TradeState.ACTIVE):
        idx, current_candle = st.session_state.current_index, st.session_state.df_replay_display.iloc[st.session_state.current_index]
        result = executor.process_signal('SELL' if executor.current_trade.direction == 'LONG' else 'BUY', current_candle)
        if result and 'marker' in result:
            result['marker']['text'] = f"MANUAL_CLOSE ({result['marker']['text']})"
            st.session_state.markers.append(result['marker'])
            if 'trade' in result:
                trade_info = result['trade']
                st.session_state.session_trades.append(trade_info)
                st.session_state.last_closed_trade_levels = {
                    'SL': trade_info.get('sl_at_entry'),
                    'TP': trade_info.get('tp_at_entry')
                }
                st.session_state.performance_metrics = calculate_performance_metrics(st.session_state.session_trades, config.INITIAL_CAPITAL, executor.get_equity_history())
        st.toast("Posición cerrada manualmente.")


def go_to_next_day_visual():
    current_date = st.session_state.ui_replay_start_date
    all_dates = sorted(list(set(st.session_state.all_data_utc.index.tz_convert(st.session_state.ui_display_tz).date)))
    next_day = next((d for d in all_dates if d > current_date), None)
    if next_day: st.session_state.ui_replay_start_date = next_day; st.session_state.app_fsm.transition_to(AppState.READY)
    else: st.toast("No hay más días con datos.")

# --- BLOQUE DE CÓDIGO PROBLEMÁTICO ELIMINADO DE AQUÍ ---

with st.sidebar:
    st.title("Configuración")
    with st.expander("1. Instrumento y Rango", expanded=True):
        st.selectbox("Símbolo", options=config.STOCK_SYMBOLS_LIST, key="ui_symbol")
        st.selectbox("Tipo de Activo", options=['STK', 'FUT', 'IND', 'FOREX', 'CFD', 'CRYPTO'], key="ui_sec_type")
        st.text_input("Exchange", key="ui_exchange")
        st.text_input("Moneda", key="ui_currency")
        primary_exchange_options = ['', 'NASDAQ', 'NYSE', 'ARCA', 'AMEX', 'ISLAND', 'BATS', 'IEX']
        st.selectbox("Primary Exch.", options=primary_exchange_options, key="ui_primary_exchange", help="Necesario para desambiguar acciones en SMART.")
        st.date_input("Inicio Descarga", key="ui_download_start")
        st.date_input("Fin Descarga", key="ui_download_end")
        st.toggle("Usar solo RTH", key="ui_use_rth")
        st.selectbox("Fuente de Datos Velas", options=['TRADES', 'MIDPOINT', 'BID', 'ASK'], key="ui_what_to_show")
        st.toggle("Usar Caché de Datos", key="ui_use_cache")

    with st.expander("2. Estrategia y Ejecución", expanded=True):
        st.radio("Modo de Backtest", ["Visual (Paso a Paso)", "Rápido (Global)"], key="ui_backtest_mode")
        st.selectbox("Filtro EMA (OBR)", options=["Desactivado", "Moderado", "Fuerte"], key="ui_ema_filter")
        st.select_slider("Apalancamiento", options=[1, 5, 10, 20, 50, 100], key="ui_leverage")
        st.selectbox("Timezone Gráfico", options=pytz.common_timezones, key="ui_display_tz")

    if st.session_state.ui_backtest_mode == "Visual (Paso a Paso)":
        st.selectbox("Timeframe", options=['1 min', '3 mins', '5 mins'], key="ui_timeframe")
        st.date_input("Día de Replay", key="ui_replay_start_date", min_value=st.session_state.ui_download_start, max_value=st.session_state.ui_download_end)
        
        # --- INICIO DE LA CORRECCIÓN ---
        # La validación de la fecha de replay se hace aquí, DESPUÉS de que el widget ha sido creado.
        if st.session_state.ui_replay_start_date < st.session_state.ui_download_start:
            st.session_state.ui_replay_start_date = st.session_state.ui_download_start
        if st.session_state.ui_replay_start_date > st.session_state.ui_download_end:
            st.session_state.ui_replay_start_date = st.session_state.ui_download_end
        # --- FIN DE LA CORRECCIÓN ---

        st.divider()
        st.subheader("Controles de Replay")
        st.number_input("Velocidad Autoplay (s)", 0.1, 5.0, 0.5, 0.1, key="ui_autoplay_speed")

    button_text = "Ejecutar Backtest" if st.session_state.ui_backtest_mode == "Rápido (Global)" else "Cargar Datos del Día"
    st.button(button_text, on_click=execute_backtest, use_container_width=True, type="primary")

fsm = st.session_state.app_fsm

if fsm.is_in_state(AppState.CONFIGURING): st.info("Ajusta los parámetros y ejecuta el backtest.")
elif fsm.is_in_state(AppState.LOADING_DATA): process_loading_state_visual(); st.rerun()
elif fsm.is_in_state(AppState.READY): process_and_prepare_daily_data_visual(); st.rerun()
elif fsm.is_in_state(AppState.GLOBAL_BACKTESTING): process_global_backtesting(); st.rerun()
elif fsm.is_in_state(AppState.SHOWING_RESULTS): render_global_results()
elif fsm.state in [AppState.PAUSED, AppState.REPLAYING, AppState.FINISHED]:
    df_replay = st.session_state.df_replay_display
    if df_replay.empty:
        st.warning(f"No hay datos para {st.session_state.ui_replay_start_date.strftime('%Y-%m-%d')}.")
        st.button(">>> Avanzar al Siguiente Día >>>", on_click=go_to_next_day_visual, use_container_width=True)
        st.stop()

    idx, current_candle, executor = st.session_state.current_index, df_replay.iloc[st.session_state.current_index], st.session_state.executor
    if executor.leverage != st.session_state.ui_leverage: executor.set_leverage(st.session_state.ui_leverage)
    if not st.session_state.performance_metrics: st.session_state.performance_metrics = calculate_performance_metrics(st.session_state.session_trades, config.INITIAL_CAPITAL, executor.get_equity_history())

    if not fsm.is_in_state(AppState.FINISHED):
        dfs_to_concat_hist = [df for df in [st.session_state.df_context_display, df_replay.iloc[:idx+1]] if not df.empty]
        df_hist_context = pd.concat(dfs_to_concat_hist) if dfs_to_concat_hist else pd.DataFrame()

        historical_data_utc = df_hist_context.tz_convert(pytz.utc)

        levels = {**st.session_state.static_levels, **st.session_state.opening_levels}
        
        signal = handle_signal_request(
            historical_data=historical_data_utc, 
            current_levels={k:v for k,v in levels.items() if pd.notna(v)}, 
            ema_filter_mode=st.session_state.ui_ema_filter,
            daily_candle_index=idx
        )
        
        result = executor.process_signal(signal, current_candle)
        
        if result and 'marker' in result:
            st.session_state.markers.append(result['marker'])
            if 'trade' in result:
                trade_info = result['trade']
                st.session_state.session_trades.append(trade_info)
                st.session_state.last_closed_trade_levels = {
                    'SL': trade_info.get('sl_at_entry'),
                    'TP': trade_info.get('tp_at_entry')
                }
                st.session_state.performance_metrics = calculate_performance_metrics(st.session_state.session_trades, config.INITIAL_CAPITAL, executor.get_equity_history())
                if fsm.is_in_state(AppState.REPLAYING): fsm.transition_to(AppState.PAUSED); st.toast("Autoplay pausado.")
            else:
                st.session_state.last_closed_trade_levels = {}

    col_metrics, col_chart = st.columns([1, 3])
    with col_metrics:
        st.subheader("Estado de Cuenta")
        equity = executor.get_equity(float(current_candle['close']))
        st.metric("Equity Actual", f"${equity:,.2f}", f"{(equity - Decimal(str(config.INITIAL_CAPITAL))):,.2f}")
        st.metric("Posición", f"{executor.account_fsm.state}")
        if executor.current_trade:
            trade = executor.current_trade
            st.metric("Entrada", f"{trade.direction} {trade.size} @ {trade.entry_price:.4f}")
            st.metric("SL Activo", f"{trade.current_sl_price:.4f}")
            st.metric("TP Activo", f"{trade.current_tp_price:.4f}")
        st.divider()
        st.subheader("Métricas del Backtest")
        metrics = st.session_state.performance_metrics
        st.metric("Trades Totales", metrics.get("Total Trades", 0))
        st.metric("Win Rate (%)", f"{metrics.get('Win Rate (%)', 0.0):.2f}%")
        st.metric("Profit Factor", f"{metrics.get('Profit Factor', 'N/A')}")
        st.metric("Max Drawdown (%)", f"{metrics.get('Max Drawdown (%)', 0.0):.2f}%")

    with col_chart:
        is_playing, is_finished, is_trade_active = fsm.is_in_state(AppState.REPLAYING), fsm.is_in_state(AppState.FINISHED) or idx >= len(df_replay) - 1, executor.account_fsm.is_in_state(TradeState.ACTIVE)
        view_cols = st.columns([2,3]); view_cols[0].checkbox("Mostrar todo el historial", key="ui_mostrar_todo", disabled=is_playing); view_cols[1].number_input("Velas Visibles Atrás:", 10, 5000, key="ui_velas_atras", disabled=is_playing or st.session_state.ui_mostrar_todo)
        nav_cols = st.columns([2, 2, 3, 2, 2, 2])
        nav_cols[0].button("<< Prev", use_container_width=True, disabled=is_playing or idx <= 0, on_click=lambda: setattr(st.session_state, 'current_index', st.session_state.current_index - 1))
        nav_cols[1].button("Play ▶" if not is_playing else "Pause ❚❚", use_container_width=True, disabled=is_finished, on_click=lambda: fsm.transition_to(AppState.REPLAYING if not is_playing else AppState.PAUSED))
        time_str = st.session_state.tz_handler.format_timestamp_for_display(current_candle.name, include_date=True)
        nav_cols[2].caption(f"<div style='text-align:center; width:100%; margin-top: 5px;'>Vela {idx + 1}/{len(df_replay)}<br>{time_str}</div>", unsafe_allow_html=True)
        nav_cols[3].button("Next >>", use_container_width=True, disabled=is_playing or is_finished, on_click=lambda: setattr(st.session_state, 'current_index', st.session_state.current_index + 1))
        nav_cols[4].button("Cerrar Pos.", use_container_width=True, disabled=is_playing or not is_trade_active, on_click=close_position_manually_visual)
        nav_cols[5].button("Reset Day", use_container_width=True, disabled=is_playing, on_click=lambda: fsm.transition_to(AppState.READY))
        
        hover_info_placeholder = st.empty()

        dfs_to_concat = [df for df in [st.session_state.df_context_display, df_replay.iloc[:idx+1]] if not df.empty]
        if dfs_to_concat:
            df_combined = pd.concat(dfs_to_concat)
        else:
            df_combined = pd.DataFrame(columns=df_replay.columns)

        df_to_render = df_combined if st.session_state.ui_mostrar_todo else df_combined.iloc[-st.session_state.ui_velas_atras:]
        
        levels_for_chart = {**st.session_state.static_levels, **st.session_state.opening_levels}
        if executor.current_trade:
            levels_for_chart['SL'] = float(executor.current_trade.current_sl_price)
            levels_for_chart['TP'] = float(executor.current_trade.current_tp_price)
        else:
            levels_for_chart.update(st.session_state.last_closed_trade_levels)
        
        chart_options, series_data = prepare_chart_data_and_options(df_display=df_to_render, markers=st.session_state.markers, static_levels=levels_for_chart, timezone_str=st.session_state.ui_display_tz, level_ranges=config.LEVEL_RANGES)
        
        chart_return_data = None
        if series_data:
            chart_return_data = renderLightweightCharts([{"chart": chart_options, "series": series_data}], key=f"chart_{idx}_{st.session_state.ui_mostrar_todo}")

        if chart_return_data and chart_return_data[0] and 'crosshair' in chart_return_data[0] and chart_return_data[0]['crosshair']:
            hovered_timestamp = chart_return_data[0]['crosshair'].get('time')
            if hovered_timestamp:
                try:
                    hovered_dt = pd.to_datetime(hovered_timestamp, unit='s', utc=True).tz_convert(st.session_state.ui_display_tz)
                    actual_ts = df_to_render.index[df_to_render.index.get_indexer([hovered_dt], method='nearest')[0]]
                    hovered_candle_index_in_day = df_replay.index.get_loc(actual_ts)
                    hover_daily_idx = hovered_candle_index_in_day + 1
                    hover_info_placeholder.caption(f"<div style='text-align:center; font-style: italic;'>Vela bajo cursor: #{hover_daily_idx}</div>", unsafe_allow_html=True)
                except Exception:
                    pass

        if not fsm.is_in_state(AppState.REPLAYING): st.button(">>> Avanzar al Siguiente Día >>>", on_click=go_to_next_day_visual, use_container_width=True)

    if fsm.is_in_state(AppState.REPLAYING) and idx < len(df_replay) - 1:
        time.sleep(st.session_state.ui_autoplay_speed); st.session_state.current_index += 1; st.rerun()
    elif idx >= len(df_replay) - 1 and not fsm.is_in_state(AppState.FINISHED): fsm.transition_to(AppState.FINISHED); st.rerun()
    if fsm.is_in_state(AppState.FINISHED): st.success("Replay del día finalizado.")
    
    st.divider()
    st.subheader("Historial de Trades de la Sesión")
    if st.session_state.session_trades:
        trades_df_display = pd.DataFrame(st.session_state.session_trades)
        trades_df_display['entry_time'] = pd.to_datetime(trades_df_display['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        trades_df_display['exit_time'] = pd.to_datetime(trades_df_display['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(trades_df_display)
    else: st.text("No hay operaciones cerradas en esta sesión.")
elif fsm.is_in_state(AppState.ERROR): st.error("Ha ocurrido un error. Revisa la configuración y los logs.")
