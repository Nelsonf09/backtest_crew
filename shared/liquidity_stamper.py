
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import time
from .liquidity_profiles import LIQUIDITY_PROFILES


def _add_minutes_to_time(hh: int, mm: int, span: int) -> time:
    total = (hh * 60 + mm + int(span)) % 1440
    return time(total // 60, total % 60)


def stamp_liquidity_window(df: pd.DataFrame, market: str, window_key: str | None) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    m = (market or '').lower()
    profiles = LIQUIDITY_PROFILES.get(m, {})
    if not window_key or window_key not in profiles:
        window_key = {
            'stocks': 'us_equity_open',
            'forex': 'london_open',
            'crypto': 'us_equity_open'
        }.get(m, 'us_equity_open')

    cfg = profiles[window_key]
    tz = cfg['tz']
    hh, mm = map(int, cfg['start'].split(':'))
    span = int(cfg['minutes'])

    # Asegurar índice con tz y convertir a tz del perfil
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    local_idx = df.index.tz_convert(tz)

    # Campos estándar para niveles
    df['session_date'] = pd.to_datetime(local_idx.date)
    df['in_session'] = True  # 24/7; si necesitas RTH, filtra fuera de esta función

    # Construir máscara sin usar .dt sobre DatetimeIndex
    start_t = time(hh, mm)
    end_t = _add_minutes_to_time(hh, mm, span)

    # Vectorizar las horas del índice local
    times = np.array([t.time() for t in local_idx])  # array de datetime.time

    if end_t > start_t:
        in_open = (times >= start_t) & (times < end_t)
    else:
        # Ventana cruza medianoche: [start_t, 24h) U [0h, end_t)
        in_open = (times >= start_t) | (times < end_t)

    df['in_opening_window'] = in_open

    # Metadatos de diagnóstico
    df.attrs['liquidity_window'] = {
        'market': m,
        'window_key': window_key,
        'tz': tz,
        'start': cfg['start'],
        'minutes': span
    }
    return df
