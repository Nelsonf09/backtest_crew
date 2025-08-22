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
# DEFAULT_SYMBOL = 'AAPL' # Ya no se usa como default principal aquí, se manejará en app.py
DEFAULT_SEC_TYPE = 'STK'  # Tipo de seguridad ('FOREX', 'FUT', 'STK', 'IND', etc.)
DEFAULT_EXCHANGE = 'SMART'# Exchange (ej: 'IDEALPRO', 'SMART', 'CME', 'NYSE')
DEFAULT_CURRENCY = 'USD'    # Moneda base del instrumento
# PRIMARY_EXCHANGE = 'ISLAND' # Opcional: Necesario para desambiguar acciones (ej: 'NASDAQ', 'NYSE')
DEFAULT_PRIMARY_EXCHANGE = 'ISLAND' # Opcional, puedes mantenerlo o ajustarlo

# --- Lista Predefinida de Símbolos ---
# Puedes agregar o quitar símbolos de esta lista según necesites
STOCK_SYMBOLS_LIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'TSLA', 'META', 'SPY', 'QQQ', 'IWM', # Añadidos ETFs comunes
    'JPM', 'JNJ', 'V', 'PG', 'MA', 'UNH', 'HD',
    # Agrega más aquí...
]
# Asegúrate de que el primer símbolo de la lista sea un buen default
DEFAULT_SYMBOL = STOCK_SYMBOLS_LIST[0] if STOCK_SYMBOLS_LIST else 'AAPL' # Fallback si la lista está vacía
DEFAULT_START_DATE_OFFSET_DAYS = 1 # Cuantos días atrás empezar por defecto
DEFAULT_DISPLAY_TZ = 'America/New_York' # Timezone por defecto para mostrar

# --- Configuración de Timeframe y Rango de Datos ---
# Timeframes válidos para IB API: '1 sec', '5 secs', '10 secs', '15 secs', '30 secs',
# '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins',
# '1 hour', '2 hours', '3 hours', '4 hours', '8 hours', '1 day', '1 week', '1 month'
DEFAULT_TIMEFRAME = '5 mins'

# Duración para descargar datos. Formato: "<int> S/D/W/M/Y" (segundos, días, semanas, meses, años)
# ¡Importante! IB tiene límites de duración según el timeframe solicitado.
# Ej: Para '1 min', el límite suele ser ~7 días ('7 D') en una sola petición.
#     Para '5 secs', el límite puede ser ~2-3 horas ('7200 S'). Consulta la doc de IB.
DEFAULT_DATA_DURATION = '5 D'

# Define si se deben usar solo datos dentro del Horario Regular de Trading (RTH)
# Relevante para Acciones, Futuros. Usualmente False para Forex/Indices Continuos.
USE_RTH = False

# What to show in chart bars (affect historical data request)
# 'TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK'
# 'MIDPOINT' or 'BID_ASK' often preferred for Forex backtesting
# 'TRades' for stocks/futures if volume is important
WHAT_TO_SHOW = 'TRADES'


# --- Configuración del Backtest / Replay ---
# Fecha y hora de inicio para la reproducción/backtest vela a vela.
# Formato: 'YYYY-MM-DD HH:MM:SS' o None para empezar desde el inicio de los datos descargados.
# Este valor será probablemente sobreescrito por el script principal basado en input del usuario.
BACKTEST_START_DATETIME_STR = None

# Velocidad de reproducción (segundos de pausa entre velas) - No implementado activamente en app.py actual
REPLAY_SPEED_SECONDS = 0.1

# --- Configuración de Caching ---
ENABLE_CACHING = True # True para guardar y cargar datos descargados localmente
# Obtiene la ruta del directorio actual del script config.py
# y luego sube un nivel para estar en la raíz del proyecto.
PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = PROJECT_ROOT / 'cache'

# --- Configuración de Salida ---
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_CHART_FILENAME = 'trading_chart.html'


# --- Configuración de Ejecución (Valores por defecto, pueden ser sobreescritos en app.py) ---
INITIAL_CAPITAL = 1000.0
COMMISSION_PER_TRADE = 0.85 # Ejemplo de comisión (ajustar a tu broker/activo)
SLIPPAGE_POINTS = 0.0 # Ejemplo de slippage (en puntos/pips, ajustar realisticamente)