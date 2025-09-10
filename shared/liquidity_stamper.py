# -*- coding: utf-8 -*-
import pandas as pd
from datetime import time, timedelta
from .liquidity_profiles import LIQUIDITY_PROFILES


def stamp_liquidity_window(df: pd.DataFrame, market: str, window_key: str | None) -> pd.DataFrame:
    if df.empty:
        return df
    m = (market or '').lower()
    profiles = LIQUIDITY_PROFILES.get(m, {})
    # Fallback sensato por mercado
    if not window_key or window_key not in profiles:
        window_key = {
            'stocks': 'us_equity_open',
            'forex':  'london_open',
            'crypto': 'us_equity_open'  # si quieres replicar impulso USA en cripto
        }.get(m, 'us_equity_open')
    cfg = profiles[window_key]
    tz = cfg['tz']
    hh, mm = map(int, cfg['start'].split(':'))
    span = int(cfg['minutes'])

    df = df.copy()
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    local = df.index.tz_convert(tz)

    df['session_date'] = pd.to_datetime(local.date)
    df['in_session'] = True

    start_t = time(hh, mm)
    # soportar saltos > 60 min de forma simple
    end_local = (local.floor('T').map(lambda ts: ts.replace(hour=hh, minute=mm, second=0, microsecond=0)) + pd.Timedelta(minutes=span)).dt.time
    df['in_opening_window'] = (local.time >= start_t) & (local.time < end_local)

    # Meta para diagnósticos
    df.attrs['liquidity_window'] = {'market': m, 'window_key': window_key, 'tz': tz, 'start': cfg['start'], 'minutes': span}
    return df
