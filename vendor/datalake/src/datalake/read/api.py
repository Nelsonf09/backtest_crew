from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

# Stub simple que lee el parquet mensual y filtra half-open [from,to)

def _load_month_partition(root: str, source: str, market: str, tf: str, symbol: str, year: int, month: int) -> pd.DataFrame:
    path = Path(root) / 'data' / f'source={source}' / f'market={market}' / f'timeframe={tf}' / f'symbol={symbol}' / f'year={year:04d}' / f'month={month:02d}' / f'part-{year:04d}-{month:02d}.parquet'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df


def read_range_df(lake_root: str, market: str, tf: str, symbol: str, date_from: str | None, date_to: str | None, source: str = 'binance') -> pd.DataFrame:
    if not lake_root:
        raise ValueError("lake_root requerido")
    if date_from is None or date_to is None:
        raise ValueError("date_from y date_to requeridos en este stub")
    start = pd.to_datetime(date_from, utc=True)
    end = pd.to_datetime(date_to, utc=True)
    if end <= start:
        return pd.DataFrame()

    # Iterar todas las particiones mensuales que intersectan [start, end)
    parts: list[pd.DataFrame] = []
    cur_year = start.year
    cur_month = start.month
    # Determinar el último mes a incluir: mes de (end - 1ns)
    last = (end - pd.Timedelta(nanoseconds=1))
    last_year = last.year
    last_month = last.month

    while (cur_year < last_year) or (cur_year == last_year and cur_month <= last_month):
        df_part = _load_month_partition(lake_root, source, market, tf, symbol, cur_year, cur_month)
        if df_part is not None and not df_part.empty:
            parts.append(df_part)
        # Avanzar un mes
        if cur_month == 12:
            cur_month = 1
            cur_year += 1
        else:
            cur_month += 1

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    if 'ts' not in df.columns:
        raise ValueError("Parquet sin columna ts")
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    # Filtrado defensivo half-open [start, end)
    mask = (df['ts'] >= start) & (df['ts'] < end)
    df = df.loc[mask].copy()
    if df.empty:
        return df
    # Orden y dedupe por seguridad
    df.sort_values('ts', inplace=True)
    df.drop_duplicates(subset='ts', keep='first', inplace=True)
    return df
