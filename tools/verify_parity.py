#!/usr/bin/env python3
"""
Verificación de paridad: Motor Rápido vs Paso-a-Paso (Datalake, Crypto)

Uso:
  PYTHONPATH=/workspaces/backtest_crew:/workspaces/backtest_crew-datalake/src \
  LAKE_ROOT=/ruta/al/lake_root \
  python tools/verify_parity.py --symbol BTC-USD --date 2025-01-01 --exec-tf "1 min" --filter-tf "15 mins" --ema-mode Fuerte

Imprime la primera operación del motor rápido y evalúa la primera señal paso-a-paso
en la misma data (mostrando EMAs y estado del filtro) para diagnosticar diferencias.
"""
from __future__ import annotations
import argparse
import os
import sys
import datetime as dt
import pandas as pd
import pytz
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from backtest_crew.bridge import BridgeConfig, DatalakeFeed  # type: ignore
from agent_core.technical_analyzer import add_technical_indicators
from shared.liquidity_stamper import stamp_liquidity_window
from strategies.vectorized_obr_exact import run_fast_backtest_exact
from strategies.opening_br_strategy import OpeningBreakRetestStrategy


def map_tf_dl(label: str) -> str:
    s = (label or "").strip().lower().replace("mins", "min").replace(" ", "")
    return {"1min": "M1", "1m": "M1", "5min": "M5", "5m": "M5", "15min": "M15", "15m": "M15", "30min": "M30", "30m": "M30"}.get(s, "M1")


def build_day_levels(df_day_local: pd.DataFrame, df_prev_local: pd.DataFrame, market_key: str) -> dict:
    # PM: antes de 1ra vela de OR
    try:
        df_day_marked = stamp_liquidity_window(df_day_local.copy(), market_key, 'us_equity_open')
        if 'in_opening_window' in df_day_marked.columns and df_day_marked['in_opening_window'].any():
            first_or_idx = df_day_marked.index[df_day_marked['in_opening_window']].min()
            df_pm = df_day_marked[df_day_marked.index < first_or_idx]
        else:
            df_pm = df_day_marked.iloc[0:0]
    except Exception:
        df_pm = df_day_local.iloc[0:0]

    levels = {}
    if not df_prev_local.empty:
        levels["PDH"] = pd.to_numeric(df_prev_local["high"], errors="coerce").max()
        levels["PDL"] = pd.to_numeric(df_prev_local["low"], errors="coerce").min()
    else:
        levels.update({"PDH": None, "PDL": None})
    if not df_pm.empty:
        levels["PMH"] = pd.to_numeric(df_pm["high"], errors="coerce").max()
        levels["PML"] = pd.to_numeric(df_pm["low"], errors="coerce").min()
    else:
        levels.update({"PMH": None, "PML": None})

    # ORH/ORL desde ventana real
    try:
        df_day_or = stamp_liquidity_window(df_day_local.copy(), market_key, 'us_equity_open')
        or_candles = df_day_or[df_day_or['in_opening_window']]
        if not or_candles.empty:
            levels["ORH"] = or_candles["high"].max()
            levels["ORL"] = or_candles["low"].min()
    except Exception:
        pass
    return {k: v for k, v in levels.items() if pd.notna(v)}


def to_local(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize('UTC')
    return out.tz_convert(tz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake-root", default=os.getenv("LAKE_ROOT", ""))
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (día en timezone local de gráfico)")
    ap.add_argument("--exec-tf", default="1 min")
    ap.add_argument("--filter-tf", default="15 mins")
    ap.add_argument("--ema-mode", default="Fuerte", choices=["Desactivado","Moderado","Fuerte"]) 
    ap.add_argument("--display-tz", default="America/New_York")
    args = ap.parse_args()

    if not args.lake_root:
        ap.error("Requiere --lake-root o LAKE_ROOT")

    # Rango con warmup de 30 días
    date_obj = dt.datetime.fromisoformat(args.date).date()
    start_date = date_obj - dt.timedelta(days=30)
    end_date = date_obj
    date_from = f"{start_date.isoformat()}T00:00:00Z"
    date_to = f"{(end_date + dt.timedelta(days=1)).isoformat()}T00:00:00Z"

    dl_exec_tf = map_tf_dl(args.exec_tf)
    dl_filter_tf = map_tf_dl(args.filter_tf)

    feed = DatalakeFeed()
    cfg_exec = BridgeConfig(lake_root=args.lake_root, source="binance", symbol=args.symbol, tf=dl_exec_tf, date_from=date_from, date_to=date_to, mode="bulk")
    cfg_filt = BridgeConfig(lake_root=args.lake_root, source="binance", symbol=args.symbol, tf=dl_filter_tf, date_from=date_from, date_to=date_to, mode="bulk")

    df_exec = feed.load_df(cfg_exec)
    df_filt = feed.load_df(cfg_filt) if dl_filter_tf != dl_exec_tf else df_exec.copy()
    if df_exec.empty:
        print("[ERR] df_exec vacío")
        sys.exit(2)

    df_exec = df_exec.set_index('ts') if 'ts' in df_exec.columns else df_exec
    df_filt = df_filt.set_index('ts') if 'ts' in df_filt.columns else df_filt
    if df_exec.index.tz is None: df_exec.index = df_exec.index.tz_localize('UTC')
    if df_filt.index.tz is None: df_filt.index = df_filt.index.tz_localize('UTC')

    # Enriquecer sobre TODO el histórico, como en Visual/Global
    df_enriched_full = add_technical_indicators(df_exec, df_filt, market='crypto', or_window='us_equity_open')

    # Convertir a TZ local (para 09:30 ventana)
    df_local = to_local(df_enriched_full, args.display_tz)

    # Particiones por día
    start_local = pytz.timezone(args.display_tz).localize(dt.datetime.combine(date_obj, dt.time.min))
    end_local = pytz.timezone(args.display_tz).localize(dt.datetime.combine(date_obj, dt.time.max))
    df_day = df_local[(df_local.index >= start_local) & (df_local.index <= end_local)]
    if df_day.empty:
        print("[ERR] df_day vacío")
        sys.exit(3)
    prev_day = date_obj - dt.timedelta(days=1)
    prev_start = pytz.timezone(args.display_tz).localize(dt.datetime.combine(prev_day, dt.time.min))
    prev_end = pytz.timezone(args.display_tz).localize(dt.datetime.combine(prev_day, dt.time.max))
    df_prev = df_local[(df_local.index >= prev_start) & (df_local.index <= prev_end)]

    # Niveles del día
    levels = build_day_levels(df_day, df_prev, 'crypto')

    # Contexto 60 velas
    previous_data = df_local[df_local.index < start_local]
    df_lookback = previous_data.tail(60)
    df_day_with_ctx = pd.concat([df_lookback, df_day])
    day_start_index = len(df_lookback)

    # Motor rápido
    trades_fast, eq_fast = run_fast_backtest_exact(
        df_day_with_context=df_day_with_ctx,
        day_start_index=day_start_index,
        day_levels=levels,
        ema_filter_mode=args.ema_mode,
        level_ranges=config.LEVEL_RANGES,
        initial_capital=config.INITIAL_CAPITAL,
        commission_per_side=config.COMMISSION_PER_TRADE,
        leverage=float(getattr(config, 'DEFAULT_LEVERAGE', 5)),
        market='crypto',
        symbol=args.symbol,
        stop_after_first_win=True,
        first_trade_loss_stop=-(config.INITIAL_CAPITAL * 0.06),
        max_trades_per_day=2,
    )
    if trades_fast.shape[0] > 0:
        ft = trades_fast[0]
        entry_ts = pd.to_datetime(ft[0], unit='s', utc=True).tz_convert(args.display_tz)
        exit_ts = pd.to_datetime(ft[1], unit='s', utc=True).tz_convert(args.display_tz)
        print(f"FAST first trade: entry={entry_ts} exit={exit_ts} dir={int(ft[2])} entry_price={ft[4]:.2f} pnl={ft[6]:.2f}")
    else:
        print("FAST first trade: none")

    # Paso-a-paso: buscar primera señal con la misma estrategia
    strat = OpeningBreakRetestStrategy(
        max_retest_candles=15,
        level_ranges=config.LEVEL_RANGES,
        ema_periods=[9,21,50],
        ema_filter_mode=args.ema_mode,
        sl_method='LOOKBACK_MIN_MAX',
        sl_lookback=2,
        risk_reward_ratio=2.0,
    )
    strat.reset_for_new_day()

    first_signal = None
    first_idx = None
    # 09:30–11:30 ventana de trading
    def in_window(ts: pd.Timestamp) -> bool:
        h, m = ts.hour, ts.minute
        return (h == 9 and m >= 30) or (h == 10) or (h == 11 and m <= 30)

    for i in range(day_start_index, len(df_day_with_ctx)):
        ts = df_day_with_ctx.index[i]
        if not in_window(ts):
            continue
        daily_idx = i - day_start_index
        sig = strat.get_signal(df_day_with_ctx.iloc[: i + 1], current_day_levels=levels, daily_candle_index=daily_idx)
        if isinstance(sig, dict):
            first_signal = sig
            first_idx = i
            break

    if first_signal is None:
        print("STEP first signal: none")
    else:
        ts = df_day_with_ctx.index[first_idx]
        cndl = df_day_with_ctx.iloc[first_idx]
        print(f"STEP first signal: ts={ts} type={first_signal.get('type')} level={first_signal.get('level')} price={cndl['close']:.2f}")
        ema9, ema21, ema50 = cndl.get('EMA_9'), cndl.get('EMA_21'), cndl.get('EMA_50')
        print(f"EMAs at entry: EMA9={ema9:.2f} EMA21={ema21:.2f} EMA50={ema50:.2f}")

    # Diagnóstico si difiere del FAST
    if trades_fast.shape[0] > 0 and first_signal is not None:
        fast_entry_local = pd.to_datetime(trades_fast[0][0], unit='s', utc=True).tz_convert(args.display_tz)
        step_entry_local = df_day_with_ctx.index[first_idx]
        if abs((fast_entry_local - step_entry_local).total_seconds()) > 60:
            print("WARN: entry mismatch > 60s")


if __name__ == "__main__":
    main()
