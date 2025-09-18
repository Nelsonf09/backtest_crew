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
    year = start.year
    month = start.month
    df = _load_month_partition(lake_root, source, market, tf, symbol, year, month)
    if df is None or df.empty:
        return pd.DataFrame()
    if 'ts' not in df.columns:
        raise ValueError("Parquet sin columna ts")
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    mask = (df['ts'] >= start) & (df['ts'] < end)
    return df.loc[mask].copy()
