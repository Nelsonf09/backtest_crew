# ui/app.py
"""
Aplicación Streamlit para el Backtesting Player Visual.
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
import re

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import config
from agent_core.main import handle_signal_request
from agent_core.data_manager import DataManager
from agent_core.technical_analyzer import add_technical_indicators
from ui.streamlit_chart import prepare_chart_data_and_options
from ui.results_renderer import render_global_results
from ui.comparison_renderer import render_comparison_dashboard
from streamlit_lightweight_charts import renderLightweightCharts
from agent_core.execution import ExecutionSimulator, TradeState
from agent_core.metrics import calculate_performance_metrics
from shared.timezone_handler import TimezoneHandler, apply_timezone_fixes
from shared.fsm import FSM, AppState
from strategies.vectorized_obr_exact import run_fast_backtest_exact

st.set_page_config(layout="wide", page_title="Backtester Híbrido (IA)", initial_sidebar_state="expanded")


def ema_tf_value_to_label(value: str) -> str:
    """Devuelve la etiqueta asociada a un valor de timeframe del filtro EMA."""
    return next(
        (opt["label"] for opt in config.EMA_FILTER_TIMEFRAME_OPTIONS if opt["value"] == value),
        value,
    )


def ema_tf_label_to_value(label: str) -> str:
    """Devuelve el valor asociado a una etiqueta de timeframe del filtro EMA."""
    return next(
        (opt["value"] for opt in config.EMA_FILTER_TIMEFRAME_OPTIONS if opt["label"] == label),
        label,
    )

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
    st.session_state.comparison_results = {}
    st.session_state.comparison_type = None # Para saber qué se está comparando
    st.session_state.df_context_display = pd.DataFrame()
    st.session_state.df_replay_display = pd.DataFrame()
    st.session_state.current_index = 0
    st.session_state.markers = []
    st.session_state.static_levels = {}
    st.session_state.opening_levels = {}
    st.session_state.last_closed_trade_levels = {}
    st.session_state.calendar_month_offset = 0

    st.session_state.ui_symbol = config.DEFAULT_SYMBOL
    st.session_state.ui_timeframe = '1 min'
    st.session_state.ui_exec_timeframe = '1 min'
    st.session_state.ema_filter_timeframe = getattr(
        config,
        "DEFAULT_EMA_FILTER_TIMEFRAME",
        next(
            (opt["value"] for opt in getattr(config, "EMA_FILTER_TIMEFRAME_OPTIONS", []) if opt.get("value")),
            "5min",
        ),
    )
    st.session_state.ema_tf_select = ema_tf_value_to_label(st.session_state.ema_filter_timeframe)
    st.session_state.ui_download_start = _start
    st.session_state.ui_download_end = _today
    st.session_state.ui_replay_start_date = max(_start, _today - datetime.timedelta(days=1))
    st.session_state.ui_display_tz = config.DEFAULT_DISPLAY_TZ
    st.session_state.ui_initial_capital = config.INITIAL_CAPITAL
    st.session_state.ui_leverage = getattr(config, 'DEFAULT_LEVERAGE', 5)
    st.session_state.ui_backtest_mode = "Rápido (Global)" # Cambiado para que por defecto esté el modo rápido
    st.session_state.ui_ema_filter = "Desactivado"
    st.session_state.ui_velas_atras = 100
    st.session_state.ui_mostrar_todo = False
    st.session_state.ui_autoplay_speed = 0.5
    st.session_state.ui_market = "stocks"  # MARKET-AWARE UI
    st.session_state.ui_sec_type = config.DEFAULT_SEC_TYPE
    st.session_state.ui_exchange = config.DEFAULT_EXCHANGE
    st.session_state.ui_currency = config.DEFAULT_CURRENCY
    st.session_state.ui_primary_exchange = config.DEFAULT_PRIMARY_EXCHANGE
    st.session_state.ui_use_rth = config.USE_RTH
    st.session_state.ui_what_to_show = config.WHAT_TO_SHOW
    st.session_state.ui_use_cache = config.ENABLE_CACHING
    
    st.session_state.ui_first_trade_loss_stop_pct = 6.0
    
    st.session_state.data_manager = DataManager()
    st.session_state.executor = ExecutionSimulator(initial_capital=st.session_state.ui_initial_capital, leverage=st.session_state.ui_leverage)
    st.session_state.tz_handler = TimezoneHandler(default_display_tz_str=st.session_state.ui_display_tz)


initialize_session_state()

def execute_backtest():
    st.session_state.calendar_month_offset = 0
    if st.session_state.ui_backtest_mode == "Visual (Paso a Paso)":
        st.session_state.app_fsm.transition_to(AppState.LOADING_DATA)
    else:
        st.session_state.app_fsm.transition_to(AppState.GLOBAL_BACKTESTING)

def run_single_backtest_iteration(df_enriched, tz_handler, ema_filter_mode):
    all_trades, full_equity_history = [], []
    unique_dates = df_enriched[df_enriched.index.date >= st.session_state.ui_download_start].index.normalize().unique()
    
    current_capital = st.session_state.ui_initial_capital
    first_trade_loss_stop_amount = -abs((st.session_state.ui_initial_capital * st.session_state.ui_first_trade_loss_stop_pct) / 100.0)
    last_day_closing_equity = st.session_state.ui_initial_capital

    dm = st.session_state.data_manager
    for date in unique_dates:
        date_obj = date.date()
        df_prev, df_pm = dm.get_levels_data(
            target_date=date_obj, symbol=st.session_state.ui_symbol, 
            sec_type=st.session_state.ui_sec_type, exchange=st.session_state.ui_exchange, 
            currency=st.session_state.ui_currency, use_cache=st.session_state.ui_use_cache, 
            primary_exchange=st.session_state.ui_primary_exchange
        )
        levels = {**dm.calculate_pdh_pdl(df_prev), **dm.calculate_pmh_pml(df_pm)}
        df_day = df_enriched[df_enriched.index.date == date_obj]
        if df_day.empty: continue

        try:
            or_start_time = tz_handler.market_open_time
            or_start_dt = df_day.index[0].replace(hour=or_start_time.hour, minute=or_start_time.minute, second=0, microsecond=0)
            or_end_dt = or_start_dt + datetime.timedelta(minutes=5)
            or_candles = df_day[(df_day.index >= or_start_dt) & (df_day.index < or_end_dt)]
            if not or_candles.empty:
                levels['ORH'], levels['ORL'] = or_candles['high'].max(), or_candles['low'].min()
        except Exception as e:
            logger.warning(f"No se pudo calcular ORH/ORL para {date_obj}: {e}")

        lookback_candles = 60
        previous_day_data = df_enriched[df_enriched.index.date < date_obj]
        df_lookback = previous_day_data.tail(lookback_candles)
        df_combined_for_day = pd.concat([df_lookback, df_day])
        day_start_index = len(df_lookback)

        market_type = "forex" if st.session_state.ui_sec_type == "FOREX" else "stocks"
        trades, equity_hist_array = run_fast_backtest_exact(
            df_day_with_context=df_combined_for_day,
            day_start_index=day_start_index,
            day_levels={k: v for k, v in levels.items() if pd.notna(v)},
            ema_filter_mode=ema_filter_mode,
            level_ranges=config.LEVEL_RANGES,
            initial_capital=current_capital,
            commission_per_side=config.COMMISSION_PER_TRADE,
            leverage=float(st.session_state.ui_leverage),
             market=market_type,
             symbol=st.session_state.ui_symbol,
            stop_after_first_win=True,
            first_trade_loss_stop=first_trade_loss_stop_amount,
            max_trades_per_day=2,
        )
        
        if trades.shape[0] > 0: all_trades.append(pd.DataFrame(trades, columns=["entry_time", "exit_time", "direction", "size", "entry_price", "exit_price", "pnl_net", "exit_reason"]))
        
        if equity_hist_array.shape[0] > 0:
            equity_hist_df = pd.DataFrame(equity_hist_array, columns=["time", "equity"])
            daily_starting_equity = equity_hist_df['equity'].iloc[0]
            equity_adjustment = last_day_closing_equity - daily_starting_equity
            equity_hist_df['equity'] += equity_adjustment
            full_equity_history.append(equity_hist_df)
            current_capital = equity_hist_df['equity'].iloc[-1]
            last_day_closing_equity = current_capital
        elif not df_day.empty:
            end_of_day_ts = df_day.index[-1].timestamp()
            no_trade_equity_df = pd.DataFrame([{'time': end_of_day_ts, 'equity': last_day_closing_equity}])
            full_equity_history.append(no_trade_equity_df)

    final_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    final_equity = pd.concat(full_equity_history, ignore_index=True) if full_equity_history else pd.DataFrame()
    return final_trades, final_equity

def load_data_for_backtest(dm: DataManager, exec_tf: str, filter_tf: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datos de ejecución y filtro para un backtest."""
    warmup_period = datetime.timedelta(days=30)
    adjusted_download_start = st.session_state.ui_download_start - warmup_period

    market = st.session_state.ui_market
    if market == "forex":
        if st.session_state.ui_primary_exchange:
            st.info("Primary Exchange no aplica en Forex y será ignorado.")
        symbol_list = getattr(config, "FOREX_SYMBOLS", [])
        exchange_list = getattr(config, "FOREX_EXCHANGES", [])
        if st.session_state.ui_symbol not in symbol_list:
            st.session_state.ui_symbol = symbol_list[0] if symbol_list else st.session_state.ui_symbol
        if st.session_state.ui_exchange not in exchange_list:
            st.session_state.ui_exchange = exchange_list[0] if exchange_list else st.session_state.ui_exchange

        common_params = {
            "symbol": st.session_state.ui_symbol,
            "sec_type": "FOREX",
            "exchange": st.session_state.ui_exchange or "IDEALPRO",
            "currency": st.session_state.ui_currency or "USD",
            "rth": st.session_state.ui_use_rth,
            "what_to_show": st.session_state.ui_what_to_show,
            "download_start_date": adjusted_download_start,
            "download_end_date": st.session_state.ui_download_end,
            "use_cache": st.session_state.ui_use_cache,
            "primary_exchange": None,
            "market": "forex",
        }
    else:
        symbol_list = getattr(config, "STOCK_SYMBOLS_LIST", [])
        exchange_list = getattr(config, "STOCKS_EXCHANGES", [])
        if st.session_state.ui_symbol not in symbol_list:
            st.session_state.ui_symbol = symbol_list[0] if symbol_list else st.session_state.ui_symbol
        if st.session_state.ui_exchange not in exchange_list:
            st.session_state.ui_exchange = exchange_list[0] if exchange_list else st.session_state.ui_exchange

        common_params = {
            "symbol": st.session_state.ui_symbol,
            "sec_type": st.session_state.ui_sec_type,
            "exchange": st.session_state.ui_exchange,
            "currency": st.session_state.ui_currency,
            "rth": st.session_state.ui_use_rth,
            "what_to_show": st.session_state.ui_what_to_show,
            "download_start_date": adjusted_download_start,
            "download_end_date": st.session_state.ui_download_end,
            "use_cache": st.session_state.ui_use_cache,
            "primary_exchange": st.session_state.ui_primary_exchange or None,
            "market": "stocks",
        }

    with st.spinner(f"Cargando datos (Ejecución: {exec_tf}, Filtro: {filter_tf})..."):
        df_exec_raw = dm.get_main_data(timeframe=exec_tf, **common_params)

        # Evitamos duplicar descargas cuando ambos timeframes coinciden
        df_filter_raw = (
            dm.get_main_data(timeframe=filter_tf, **common_params)
            if filter_tf != exec_tf
            else df_exec_raw.copy()
        )

    return df_exec_raw, df_filter_raw

def process_global_backtesting():
    dm = st.session_state.data_manager
    tz_handler = TimezoneHandler(default_display_tz_str=st.session_state.ui_display_tz)

    market = st.session_state.ui_market
    if market == "forex" and st.session_state.ui_primary_exchange:
        st.info("Primary Exchange no aplica en Forex y será ignorado.")
        st.session_state.ui_primary_exchange = ""
    symbol_list = config.FOREX_SYMBOLS if market == "forex" else config.STOCK_SYMBOLS_LIST
    if st.session_state.ui_symbol not in symbol_list:
        st.info(f"Símbolo inválido para {market}, ajustado a {symbol_list[0]}.")
        st.session_state.ui_symbol = symbol_list[0]
    exchange_list = config.FOREX_EXCHANGES if market == "forex" else config.STOCKS_EXCHANGES
    if st.session_state.ui_exchange not in exchange_list:
        st.session_state.ui_exchange = exchange_list[0]

    with st.spinner("Conectando a IB..."):
        if not dm.connect_ib():
            st.error("Fallo la conexión a IB."); st.session_state.app_fsm.transition_to(AppState.ERROR); return

    try:
        is_filter_comparison = st.session_state.ui_ema_filter == "Comparar Filtros"
        is_timeframe_comparison = st.session_state.ema_filter_timeframe == "compare"

        if is_filter_comparison and is_timeframe_comparison:
            st.error("No se pueden ejecutar ambas comparaciones a la vez. Por favor, elija solo una.")
            st.session_state.app_fsm.transition_to(AppState.CONFIGURING)
            return

        # Lógica de Comparación de Timeframes
        if is_timeframe_comparison:
            st.session_state.comparison_type = "Timeframes de Filtro"
            st.session_state.comparison_results = {}
            timeframes_to_compare = get_filter_timeframe_options(st.session_state.ui_market)
            progress_bar = st.progress(0, text="Iniciando comparación de timeframes...")
            fixed_ema_filter_mode = st.session_state.ui_ema_filter

            for i, tf_val in enumerate(timeframes_to_compare):
                tf_label = ema_tf_value_to_label(tf_val)
                text = f"Ejecutando para timeframe: {tf_label} (Filtro: {fixed_ema_filter_mode})"
                progress_bar.progress((i + 1) / len(timeframes_to_compare), text=text)

                df_exec_raw, df_filter_raw = load_data_for_backtest(
                    dm,
                    st.session_state.ui_exec_timeframe,
                    tf_label,
                )
                if df_exec_raw.empty:
                    logger.warning(f"No hay datos de ejecución, saltando timeframe {tf_label}.")
                    continue
                df_enriched = add_technical_indicators(df_exec_raw, df_filter_raw)
                df_enriched_local = df_enriched.tz_convert(tz_handler.display_tz)

                trades, equity = run_single_backtest_iteration(df_enriched_local, tz_handler, fixed_ema_filter_mode)
                st.session_state.comparison_results[tf_label] = {'trades': trades, 'equity': equity}

            progress_bar.empty()
            st.session_state.app_fsm.transition_to(AppState.SHOWING_COMPARISON)

        # Lógica de Comparación de Filtros EMA
        elif is_filter_comparison:
            st.session_state.comparison_type = "Modos de Filtro EMA"
            st.session_state.comparison_results = {}
            filter_modes_to_compare = ["Desactivado", "Moderado", "Fuerte"]
            progress_bar = st.progress(0, text="Iniciando comparación de filtros EMA...")
            fixed_filter_tf_val = st.session_state.ema_filter_timeframe
            fixed_filter_tf = ema_tf_value_to_label(fixed_filter_tf_val)

            df_exec_raw, df_filter_raw = load_data_for_backtest(dm, st.session_state.ui_exec_timeframe, fixed_filter_tf)
            if df_exec_raw.empty:
                st.warning("No se obtuvieron datos."); st.session_state.app_fsm.transition_to(AppState.CONFIGURING); return

            df_enriched = add_technical_indicators(df_exec_raw, df_filter_raw)
            df_enriched_local = df_enriched.tz_convert(tz_handler.display_tz)

            for i, mode in enumerate(filter_modes_to_compare):
                progress_bar.progress(
                    (i + 1) / len(filter_modes_to_compare),
                    text=f"Ejecutando para filtro: {mode} (Timeframe: {fixed_filter_tf})",
                )
                trades, equity = run_single_backtest_iteration(df_enriched_local.copy(), tz_handler, mode)
                st.session_state.comparison_results[mode] = {'trades': trades, 'equity': equity}
            
            progress_bar.empty()
            st.session_state.app_fsm.transition_to(AppState.SHOWING_COMPARISON)
        
        # Lógica de Backtest Único
        else:
            filter_tf = ema_tf_value_to_label(st.session_state.ema_filter_timeframe)
            df_exec_raw, df_filter_raw = load_data_for_backtest(
                dm,
                st.session_state.ui_exec_timeframe,
                filter_tf,
            )
            if df_exec_raw.empty:
                st.warning("No se obtuvieron datos."); st.session_state.app_fsm.transition_to(AppState.CONFIGURING); return

            df_enriched = add_technical_indicators(df_exec_raw, df_filter_raw)
            df_enriched_local = df_enriched.tz_convert(tz_handler.display_tz)
            
            st.session_state.session_trades, st.session_state.global_equity_history = run_single_backtest_iteration(
                df_enriched_local.copy(), tz_handler, st.session_state.ui_ema_filter
            )
            st.session_state.app_fsm.transition_to(AppState.SHOWING_RESULTS)

    except Exception as e:
        logger.error(f"Error en backtest global: {e}", exc_info=True)
        st.error(f"Error en backtest global: {e}")
        st.session_state.app_fsm.transition_to(AppState.ERROR)
    finally:
        dm.disconnect_ib()

def process_loading_state_visual():
    dm = st.session_state.data_manager
    st.session_state.session_trades, st.session_state.performance_metrics = [], {}
    st.session_state.executor = ExecutionSimulator(initial_capital=st.session_state.ui_initial_capital, leverage=st.session_state.ui_leverage)
    with st.spinner("Conectando a IB..."):
        if not dm.connect_ib(): st.error("Fallo la conexión a IB."); st.session_state.app_fsm.transition_to(AppState.ERROR); return
    try:
        lookback_days = datetime.timedelta(days=30)
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

    df_full_day_utc = pd.concat([df_context_utc, df_replay_utc])
    df_enriched = add_technical_indicators(df_full_day_utc, ema_periods=[9,21,50])
    
    df_context_enriched = df_enriched.loc[df_context_utc.index]
    df_replay_enriched = df_enriched.loc[df_replay_utc.index]
    
    st.session_state.df_context_display = apply_timezone_fixes(st.session_state, df_context_enriched, st.session_state.ui_display_tz, st.session_state.ui_use_rth)
    st.session_state.df_replay_display = apply_timezone_fixes(st.session_state, df_replay_enriched, st.session_state.ui_display_tz, st.session_state.ui_use_rth)
    
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
                st.session_state.performance_metrics = calculate_performance_metrics(st.session_state.session_trades, st.session_state.ui_initial_capital, executor.get_equity_history())
        st.toast("Posición cerrada manualmente.")

def go_to_next_day_visual():
    current_date = st.session_state.ui_replay_start_date
    all_dates = sorted(list(set(st.session_state.all_data_utc.index.tz_convert(st.session_state.ui_display_tz).date)))
    next_day = next((d for d in all_dates if d > current_date), None)
    if next_day: st.session_state.ui_replay_start_date = next_day; st.session_state.app_fsm.transition_to(AppState.READY)
    else: st.toast("No hay más días con datos.")


def timeframe_to_minutes(tf: str) -> int:
    """Convierte un timeframe (label o value) en minutos."""
    tf = tf.strip().lower()
    match = re.match(r"(\d+)\s*(\w+)?", tf)
    if not match:
        return 0
    value = int(match.group(1))
    unit = match.group(2) or "min"
    if unit.startswith("min"):
        return value
    if unit.startswith("hour") or unit == "h":
        return value * 60
    if unit.startswith("day") or unit == "d":
        return value * 60 * 24
    return value


def get_exec_timeframe_options(market: str) -> list[str]:
    """Devuelve opciones de timeframe de ejecución según el mercado."""
    return ["1 min", "5 mins"] if market == "forex" else ["1 min", "3 mins", "5 mins"]


def get_filter_timeframe_options(market: str) -> list[str]:
    """Devuelve opciones de timeframe (valores) para el filtro EMA según el mercado."""
    if market == "forex":
        return ["1min", "5min"]
    return [
        opt["value"]
        for opt in config.EMA_FILTER_TIMEFRAME_OPTIONS
        if opt["value"] != "compare"
    ]


def handle_market_change():
    """Reasigna selectores cuando cambia el mercado."""  # MARKET-AWARE UI
    market = st.session_state.ui_market
    if market == "forex":
        st.session_state.ui_symbol = config.FOREX_SYMBOLS[0]
        st.session_state.ui_exchange = config.FOREX_EXCHANGES[0]
        st.session_state.ui_primary_exchange = ""
        st.session_state.ui_sec_type = "FOREX"
    else:
        st.session_state.ui_symbol = config.STOCK_SYMBOLS_LIST[0]
        st.session_state.ui_exchange = config.STOCKS_EXCHANGES[0]
        st.session_state.ui_primary_exchange = (
            config.STOCKS_PRIMARY_EXCHANGES[0] if config.STOCKS_PRIMARY_EXCHANGES else ""
        )
        st.session_state.ui_sec_type = config.DEFAULT_SEC_TYPE

    exec_options = get_exec_timeframe_options(market)
    if st.session_state.ui_exec_timeframe not in exec_options:
        st.session_state.ui_exec_timeframe = exec_options[0]

    filter_options = get_filter_timeframe_options(market)
    if (
        st.session_state.ema_filter_timeframe not in filter_options
        and st.session_state.ema_filter_timeframe != "compare"
    ):
        st.session_state.ema_filter_timeframe = filter_options[0]
    st.session_state.ema_tf_select = ema_tf_value_to_label(st.session_state.ema_filter_timeframe)

with st.sidebar:
    st.title("Configuración")
    with st.expander("1. Instrumento y Rango", expanded=True):
        # MARKET-AWARE UI
        market_options = ["stocks", "forex"]
        if st.session_state.ui_market not in market_options:
            st.session_state.ui_market = "stocks"
        st.selectbox("Mercado", options=market_options, key="ui_market", on_change=handle_market_change)

        if st.session_state.ui_market == "forex":
            symbol_options = config.FOREX_SYMBOLS
            exchange_options = config.FOREX_EXCHANGES
            if st.session_state.ui_symbol not in symbol_options:
                st.session_state.ui_symbol = symbol_options[0]
            if st.session_state.ui_exchange not in exchange_options:
                st.session_state.ui_exchange = exchange_options[0]
            st.selectbox("Símbolo", options=symbol_options, key="ui_symbol")
            st.selectbox("Exchange", options=exchange_options, key="ui_exchange")
            st.caption("Forex usa IDEALPRO y no requiere Primary Exchange.")
            st.session_state.ui_primary_exchange = ""
            st.text_input("Primary Exch.", value="N/A", disabled=True)
        else:
            symbol_options = config.STOCK_SYMBOLS_LIST
            exchange_options = config.STOCKS_EXCHANGES
            if st.session_state.ui_symbol not in symbol_options:
                st.session_state.ui_symbol = symbol_options[0]
            if st.session_state.ui_exchange not in exchange_options:
                st.session_state.ui_exchange = exchange_options[0]
            st.selectbox("Símbolo", options=symbol_options, key="ui_symbol")
            st.selectbox("Exchange", options=exchange_options, key="ui_exchange")
            primary_exchange_options = [""] + config.STOCKS_PRIMARY_EXCHANGES
            if st.session_state.ui_primary_exchange not in primary_exchange_options:
                st.session_state.ui_primary_exchange = primary_exchange_options[0]
            st.selectbox("Primary Exch.", options=primary_exchange_options, key="ui_primary_exchange", help="Necesario para desambiguar acciones en SMART.")

        st.selectbox("Tipo de Activo", options=['STK', 'FUT', 'IND', 'FOREX', 'CFD', 'CRYPTO'], key="ui_sec_type")
        st.text_input("Moneda", key="ui_currency")
        st.date_input("Inicio Descarga", key="ui_download_start")
        st.date_input("Fin Descarga", key="ui_download_end")
        st.toggle("Usar solo RTH", key="ui_use_rth")
        st.session_state.ui_what_to_show = st.session_state.get(
            "ui_what_to_show", config.WHAT_TO_SHOW
        )
        st.selectbox(
            "Fuente de Datos Velas",
            options=['TRADES', 'MIDPOINT', 'BID', 'ASK'],
            key="ui_what_to_show",
            help="En Forex se usará MIDPOINT internamente, aunque aquí veas TRADES.",
        )
        st.toggle("Usar Caché de Datos", key="ui_use_cache")

    with st.expander("2. Estrategia y Ejecución", expanded=True):
        st.radio("Modo de Backtest", ["Visual (Paso a Paso)", "Rápido (Global)"], key="ui_backtest_mode")
        st.number_input("Capital Inicial ($)", min_value=100.0, step=100.0, key="ui_initial_capital")
        st.number_input(
            "Pérdida Máxima 1er Trade (%)", 
            min_value=0.1, max_value=100.0, 
            key="ui_first_trade_loss_stop_pct", 
            step=0.5,
            help="Si la primera operación del día pierde este % del capital, se detiene el trading para ese día."
        )
        exec_tf_options = get_exec_timeframe_options(st.session_state.ui_market)
        if st.session_state.ui_exec_timeframe not in exec_tf_options:
            st.session_state.ui_exec_timeframe = exec_tf_options[0]
        st.selectbox(
            "Timeframe de Ejecución",
            options=exec_tf_options,
            key="ui_exec_timeframe",
            help="Timeframe de Ejecución = timeframe base de las velas en las que se abren/cerran operaciones. El Filtro EMA puede usar un timeframe igual o superior.",
        )

        if st.session_state.ui_market == "forex" and st.session_state.ui_exec_timeframe == "1 min":
            st.caption("Para rangos largos usa 1–3 días; el sistema fragmenta las descargas para evitar timeouts de HMDS.")

        filter_tf_base_values = get_filter_timeframe_options(st.session_state.ui_market)
        if (
            st.session_state.ema_filter_timeframe not in filter_tf_base_values
            and st.session_state.ema_filter_timeframe != "compare"
        ):
            st.session_state.ema_filter_timeframe = filter_tf_base_values[0]

        options = filter_tf_base_values + ["compare"]
        labels = [ema_tf_value_to_label(val) for val in options]
        default_idx = next(
            (
                i
                for i, val in enumerate(options)
                if val == st.session_state.ema_filter_timeframe
                or val == config.DEFAULT_EMA_FILTER_TIMEFRAME
                or ema_tf_value_to_label(val) == config.DEFAULT_EMA_FILTER_TIMEFRAME
            ),
            0,
        )

        selected_label = st.selectbox(
            "Timeframe del Filtro EMA",
            labels,
            index=default_idx,
            key="ema_tf_select",
            help="Timeframe para calcular las EMAs. Puede ser igual o superior al de ejecución.",
        )

        selected_value = options[labels.index(selected_label)]
        st.session_state.ema_filter_timeframe = selected_value

        if st.session_state.ema_filter_timeframe != "compare":
            exec_minutes = timeframe_to_minutes(st.session_state.ui_exec_timeframe)
            filter_minutes = timeframe_to_minutes(st.session_state.ema_filter_timeframe)
            if filter_minutes < exec_minutes:
                st.warning(
                    "Se recomienda usar un timeframe del filtro EMA >= al de ejecución."
                )
        st.selectbox("Filtro EMA (OBR)", options=["Desactivado", "Moderado", "Fuerte", "Comparar Filtros"], key="ui_ema_filter")
        st.select_slider("Apalancamiento", options=[1, 5, 10, 20, 50, 100], key="ui_leverage")
        st.selectbox("Timezone Gráfico", options=pytz.common_timezones, key="ui_display_tz")

    if st.session_state.ui_backtest_mode == "Visual (Paso a Paso)":
        st.selectbox("Timeframe (Visual)", options=['1 min', '3 mins', '5 mins'], key="ui_timeframe")
        
        if st.session_state.ui_replay_start_date < st.session_state.ui_download_start:
            st.session_state.ui_replay_start_date = st.session_state.ui_download_start
        if st.session_state.ui_replay_start_date > st.session_state.ui_download_end:
            st.session_state.ui_replay_start_date = st.session_state.ui_download_end

        st.date_input("Día de Replay", key="ui_replay_start_date", 
                      min_value=st.session_state.ui_download_start, 
                      max_value=st.session_state.ui_download_end)
        
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
elif fsm.is_in_state(AppState.SHOWING_COMPARISON): render_comparison_dashboard()
elif fsm.state in [AppState.PAUSED, AppState.REPLAYING, AppState.FINISHED]:
    df_replay = st.session_state.df_replay_display
    if df_replay.empty:
        st.warning(f"No hay datos para {st.session_state.ui_replay_start_date.strftime('%Y-%m-%d')}.")
        st.button(">>> Avanzar al Siguiente Día >>>", on_click=go_to_next_day_visual, use_container_width=True)
        st.stop()

    idx, current_candle, executor = st.session_state.current_index, df_replay.iloc[st.session_state.current_index], st.session_state.executor
    if executor.leverage != st.session_state.ui_leverage: executor.set_leverage(st.session_state.ui_leverage)
    if not st.session_state.performance_metrics: st.session_state.performance_metrics = calculate_performance_metrics(st.session_state.session_trades, st.session_state.ui_initial_capital, executor.get_equity_history())

    if not fsm.is_in_state(AppState.FINISHED):
        dfs_to_concat_hist = [df for df in [st.session_state.df_context_display, df_replay.iloc[:idx+1]] if not df.empty]
        df_hist_context = pd.concat(dfs_to_concat_hist) if dfs_to_concat_hist else pd.DataFrame()

        levels = {**st.session_state.static_levels, **st.session_state.opening_levels}
        
        ema_filter = st.session_state.ui_ema_filter if st.session_state.ui_ema_filter != "Comparar Filtros" else "Desactivado"
        signal = handle_signal_request(
            historical_data=df_hist_context, 
            current_levels={k:v for k,v in levels.items() if pd.notna(v)}, 
            ema_filter_mode=ema_filter,
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
                st.session_state.performance_metrics = calculate_performance_metrics(st.session_state.session_trades, st.session_state.ui_initial_capital, executor.get_equity_history())
                if fsm.is_in_state(AppState.REPLAYING): fsm.transition_to(AppState.PAUSED); st.toast("Autoplay pausado.")
            else:
                st.session_state.last_closed_trade_levels = {}

    col_metrics, col_chart = st.columns([1, 3])
    with col_metrics:
        st.subheader("Estado de Cuenta")
        equity = executor.get_equity(float(current_candle['close']))
        st.metric("Equity Actual", f"${equity:,.2f}", f"{(equity - Decimal(str(st.session_state.ui_initial_capital))):,.2f}")
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
