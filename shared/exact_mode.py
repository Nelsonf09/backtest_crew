import pandas as pd
import pytz

# Parámetros 1:1 usados SOLO por el motor paso a paso
TRADING_WINDOW_START = '09:30'  # NY
TRADING_WINDOW_END   = '11:30'  # NY
FORCE_CLOSE_TIME     = '13:00'  # NY
SLIPPAGE_MODE        = 'none'   # 'none' | 'points' | 'percent'
SLIPPAGE_VALUE       = 0.0
ALLOW_FRACTIONAL_SIZE_CRYPTO = True
NY_TZ = pytz.timezone('America/New_York')


def to_ny(ts):
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize('UTC')
    return t.tz_convert(NY_TZ)
