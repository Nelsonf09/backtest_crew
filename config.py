# config.py
"""
Archivo de configuración central para el Scalping Backtester.
"""

import os
from pathlib import Path

# --- Configuración de Conexión a Interactive Brokers ---
IB_HOST = '127.0.0.1' # Host de TWS o IB Gateway
IB_PORT = 7497        # Puerto de TWS (por defecto). Gateway suele ser 4002.
IB_CLIENT_ID = 1      # ID de cliente único para esta conexión API.

# --- Configuración de Datos por Defecto ---
DEFAULT_SEC_TYPE = 'STK'
DEFAULT_EXCHANGE = 'SMART'
DEFAULT_CURRENCY = 'USD'
DEFAULT_PRIMARY_EXCHANGE = 'ISLAND'

# --- Lista Predefinida de Símbolos ---
STOCK_SYMBOLS_LIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'TSLA', 'META', 'SPY', 'QQQ', 'IWM',
    'JPM', 'JNJ', 'V', 'PG', 'MA', 'UNH', 'HD',
]
DEFAULT_SYMBOL = STOCK_SYMBOLS_LIST[0] if STOCK_SYMBOLS_LIST else 'AAPL'
DEFAULT_START_DATE_OFFSET_DAYS = 1
DEFAULT_DOWNLOAD_RANGE_DAYS = 5
DEFAULT_DISPLAY_TZ = 'America/New_York'

# --- Configuración de Timeframe y Rango de Datos ---
DEFAULT_TIMEFRAME = '1 min'
# --- INICIO DE LA MODIFICACIÓN ---
# Lista de timeframes para el filtro EMA, ahora centralizada.
EMA_FILTER_TIMEFRAMES = [
    '1 min', '5 mins', '15 mins'
]
# --- FIN DE LA MODIFICACIÓN ---
DEFAULT_DATA_DURATION = '5 D'
USE_RTH = False
WHAT_TO_SHOW = 'TRADES'

# --- Configuración de Niveles y Estrategia ---
LEVEL_RANGES = {
    'PDH': 0.5, 'PDL': 0.5, 'PMH': 0.3, 'PML': 0.3, 'ORH': 0.2, 'ORL': 0.2
}

# --- Configuración del Backtest / Replay ---
BACKTEST_START_DATETIME_STR = None
REPLAY_SPEED_SECONDS = 0.1

# --- Configuración de Caching ---
ENABLE_CACHING = True
PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = PROJECT_ROOT / 'cache'

# --- Configuración de Salida ---
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_CHART_FILENAME = 'trading_chart.html'

# --- Configuración de Ejecución (Valores por defecto) ---
INITIAL_CAPITAL = 1000.0
COMMISSION_PER_TRADE = 0.85
SLIPPAGE_POINTS = 0.0
DEFAULT_LEVERAGE = 5
