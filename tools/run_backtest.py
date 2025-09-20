#!/usr/bin/env python3
"""CLI to execute fast backtests using IB data loader."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import os

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from agent_core.technical_analyzer import add_technical_indicators
from shared.timezone_handler import TimezoneHandler
from strategies.vectorized_obr_exact import run_fast_backtest_exact
from agent_core.utils.metrics import compute_global_metrics

# Bridge opcional
try:
    from backtest_crew.bridge import BridgeConfig, DatalakeFeed  # type: ignore
except Exception:  # pragma: no cover
    BridgeConfig = None  # type: ignore
    DatalakeFeed = None  # type: ignore


def _calc_pdh_pdl(df_previous_day: pd.DataFrame) -> dict:
    if df_previous_day is None or df_previous_day.empty:
        return {"PDH": None, "PDL": None}
    pdh = pd.to_numeric(df_previous_day["high"], errors="coerce").max()
    pdl = pd.to_numeric(df_previous_day["low"], errors="coerce").min()
    return {"PDH": pdh if pd.notna(pdh) else None, "PDL": pdl if pd.notna(pdl) else None}


def _calc_pmh_pml(df_premarket: pd.DataFrame) -> dict:
    if df_premarket is None or df_premarket.empty:
        return {"PMH": None, "PML": None}
    pmh = pd.to_numeric(df_premarket["high"], errors="coerce").max()
    pml = pd.to_numeric(df_premarket["low"], errors="coerce").min()
    return {"PMH": pmh if pd.notna(pmh) else None, "PML": pml if pd.notna(pml) else None}


def run_backtest(symbol: str, timeframe: str, market: str, limit_days: int, out_path: Path,
                 use_datalake: bool = False, lake_kwargs: dict | None = None) -> None:
    dm = None
    tz_handler = TimezoneHandler()

    today = dt.date.today()
    start_date = today - dt.timedelta(days=limit_days)

    if market == "stocks":
        sec_type = "STK"
        exchange = config.DEFAULT_EXCHANGE
    elif market == "forex":
        sec_type = "FOREX"
        exchange = "IDEALPRO"
    elif market == "crypto":
        sec_type = "CRYPTO"
        exchange = getattr(config, "IB_CRYPTO_EXCHANGE", "PAXOS")
    else:
        sec_type = "STK"
        exchange = config.DEFAULT_EXCHANGE
    currency = config.DEFAULT_CURRENCY

    if use_datalake and DatalakeFeed and BridgeConfig and lake_kwargs:
        # Interpretamos timeframe IB -> tf datalake si es posible (simple mapping de minutos).
        tf_map = {"1 min": "M1", "5 mins": "M5", "15 mins": "M15", "30 mins": "M30"}
        dl_tf = tf_map.get((timeframe or "").strip().lower(), "M1")
        # Determinamos rango: si se pasa 'date' en lake_kwargs se convierte en from/to.
        if 'date' in lake_kwargs and lake_kwargs['date']:
            date_str = lake_kwargs.pop('date')
            base_date = dt.datetime.fromisoformat(date_str).date()
            date_from = f"{base_date.isoformat()}T00:00:00Z"
            date_to = f"{(base_date + dt.timedelta(days=1)).isoformat()}T00:00:00Z"
            lake_kwargs['date_from'] = date_from
            lake_kwargs['date_to'] = date_to
        cfg = BridgeConfig(
            lake_root=lake_kwargs.get('lake_root') or os.getenv('LAKE_ROOT', ''),
            source=lake_kwargs.get('source', 'binance'),
            symbol=lake_kwargs.get('symbol', symbol),
            tf=(lake_kwargs.get('tf') or dl_tf),
            date_from=lake_kwargs.get('date_from'),
            date_to=lake_kwargs.get('date_to'),
            mode=lake_kwargs.get('mode', 'bulk'),
            speed_bps=float(lake_kwargs.get('speed_bps', 0.0)),
            rename_ts_to=lake_kwargs.get('rename_ts_to'),
        )
        print(f"[dbg] Datalake cfg: root={cfg.lake_root} source={cfg.source} symbol={cfg.symbol} tf={cfg.tf} from={cfg.date_from} to={cfg.date_to}")
        feed = DatalakeFeed()
        df_exec = feed.load_df(cfg)
        if df_exec is not None and not df_exec.empty:
            # Re-index a DatetimeIndex en UTC si ts es columna.
            if 'ts' in df_exec.columns:
                df_exec = df_exec.set_index('ts')
            if df_exec.index.tz is None:
                df_exec.index = df_exec.index.tz_localize('UTC')
            # Adaptamos nombres a los usados por IB path si difieren.
        else:
            df_exec = None
    else:
        # Importar DataManager solo si se usa IB
        from agent_core.data_manager import DataManager  # type: ignore
        dm = DataManager()
        df_exec = dm.get_main_data(
            symbol=symbol,
            timeframe=timeframe,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency,
            rth=config.USE_RTH,
            what_to_show=config.WHAT_TO_SHOW,
            download_start_date=start_date,
            download_end_date=today,
            use_cache=True,
            market=market,
        )
    if df_exec is None or df_exec.empty:
        output = {"metrics": {}, "trades": []}
        out_path.write_text(json.dumps(output, indent=2))
        return

    df_enriched = add_technical_indicators(df_exec, df_exec, market=market)

    unique_dates = df_enriched.index.normalize().unique()
    all_trades = []
    full_equity_history = []
    current_capital = config.INITIAL_CAPITAL
    first_trade_loss_stop = -abs(config.INITIAL_CAPITAL * 0.06)
    last_day_closing_equity = current_capital

    for date in unique_dates:
        date_obj = date.date()
        df_day = df_enriched[df_enriched.index.date == date_obj]
        if df_day.empty:
            continue

        # Derivar niveles según fuente
        if use_datalake:
            prev_day = date_obj - dt.timedelta(days=1)
            df_prev = df_enriched[df_enriched.index.date == prev_day]
            # Ventana OR por defecto: usar 5 minutos desde inicio del día en zona de mercado (aprox)
            try:
                or_start_time = tz_handler.market_open_time
                or_start_dt = df_day.index[0].replace(hour=or_start_time.hour, minute=or_start_time.minute, second=0, microsecond=0)
                or_end_dt = or_start_dt + dt.timedelta(minutes=5)
                df_pm = df_day[df_day.index < or_start_dt]
                or_candles = df_day[(df_day.index >= or_start_dt) & (df_day.index < or_end_dt)]
            except Exception:
                df_pm = df_day.iloc[0:0]
                or_candles = df_day.iloc[0:0]
            levels = {**_calc_pdh_pdl(df_prev), **_calc_pmh_pml(df_pm)}
            if not or_candles.empty:
                levels["ORH"], levels["ORL"] = or_candles["high"].max(), or_candles["low"].min()
        else:
            # Importar DataManager para niveles IB si no fue importado antes
            if dm is None:
                from agent_core.data_manager import DataManager  # type: ignore
                dm = DataManager()
            df_prev, df_pm = dm.get_levels_data(
                target_date=date_obj,
                symbol=symbol,
                sec_type=sec_type,
                exchange=exchange,
                currency=currency,
                use_cache=True,
                market=market,
            )
            levels = {**_calc_pdh_pdl(df_prev), **_calc_pmh_pml(df_pm)}
            try:
                or_start_time = tz_handler.market_open_time
                or_start_dt = df_day.index[0].replace(hour=or_start_time.hour, minute=or_start_time.minute, second=0, microsecond=0)
                or_end_dt = or_start_dt + dt.timedelta(minutes=5)
                or_candles = df_day[(df_day.index >= or_start_dt) & (df_day.index < or_end_dt)]
                if not or_candles.empty:
                    levels["ORH"], levels["ORL"] = or_candles["high"].max(), or_candles["low"].min()
            except Exception:
                pass

        previous_day_data = df_enriched[df_enriched.index.date < date_obj]
        df_lookback = previous_day_data.tail(60)
        df_combined_for_day = pd.concat([df_lookback, df_day])
        day_start_index = len(df_lookback)

        trades, equity_hist_array = run_fast_backtest_exact(
            df_day_with_context=df_combined_for_day,
            day_start_index=day_start_index,
            day_levels={k: v for k, v in levels.items() if pd.notna(v)},
            ema_filter_mode="Desactivado",
            level_ranges=config.LEVEL_RANGES,
            initial_capital=current_capital,
            commission_per_side=config.COMMISSION_PER_TRADE,
            leverage=float(getattr(config, "DEFAULT_LEVERAGE", 5)),
            stop_after_first_win=True,
            first_trade_loss_stop=first_trade_loss_stop,
            max_trades_per_day=2,
            market=market,
            symbol=symbol,
        )

        if trades.shape[0] > 0:
            trades_df = pd.DataFrame(
                trades,
                columns=[
                    "entry_time",
                    "exit_time",
                    "direction",
                    "size",
                    "entry_price",
                    "exit_price",
                    "pnl_net",
                    "exit_reason",
                ],
            )
            all_trades.append(trades_df)

        if equity_hist_array.shape[0] > 0:
            equity_hist_df = pd.DataFrame(equity_hist_array, columns=["time", "equity"])
            daily_starting_equity = equity_hist_df["equity"].iloc[0]
            equity_adjustment = last_day_closing_equity - daily_starting_equity
            equity_hist_df["equity"] += equity_adjustment
            full_equity_history.append(equity_hist_df)
            current_capital = equity_hist_df["equity"].iloc[-1]
            last_day_closing_equity = current_capital
        else:
            end_of_day_ts = df_day.index[-1].timestamp()
            no_trade_equity_df = pd.DataFrame([{"time": end_of_day_ts, "equity": last_day_closing_equity}])
            full_equity_history.append(no_trade_equity_df)

    final_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    final_equity = pd.concat(full_equity_history, ignore_index=True) if full_equity_history else pd.DataFrame()

    trades_list = final_trades.rename(columns={"pnl_net": "pnl"}).to_dict("records")
    equity_series = final_equity["equity"] if not final_equity.empty else pd.Series(dtype=float)
    metrics = compute_global_metrics(
        equity_series,
        trades_list,
        config.INITIAL_CAPITAL,
    )
    output = {"metrics": metrics, "trades": final_trades.to_dict("records")}
    out_path.write_text(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast backtests")
    parser.add_argument("--engine", default="fast", choices=["fast"], help="Backtest engine")
    parser.add_argument("--market", default="stocks", choices=["stocks", "forex", "crypto"], help="Market type")
    parser.add_argument("--symbol", required=True, help="Instrument symbol")
    parser.add_argument("--timeframe", default="1 min", help="Timeframe string for IB")
    parser.add_argument("--limit", type=int, default=5, help="Number of past days to download")
    parser.add_argument("--out", default="backtest.json", help="Output JSON file")
    # Flags opcionales para datalake
    parser.add_argument("--use-datalake", action="store_true", help="Activar lectura desde Datalake (bulk)")
    parser.add_argument("--lake-root", help="Root del datalake (fallback LAKE_ROOT)")
    parser.add_argument("--dl-source", dest="dl_source", default="binance")
    parser.add_argument("--dl-symbol", dest="dl_symbol", help="Símbolo override para datalake")
    parser.add_argument("--dl-tf", dest="dl_tf", help="TF datalake (M1/M5/M15/M30)")
    parser.add_argument("--dl-date", dest="dl_date", help="Día completo UTC (YYYY-MM-DD)")
    parser.add_argument("--dl-date-from", dest="dl_date_from", help="Inicio rango UTC ISO")
    parser.add_argument("--dl-date-to", dest="dl_date_to", help="Fin rango UTC ISO (exclusivo)")
    parser.add_argument("--dl-rename-ts-to", dest="dl_rename_ts_to", help="Renombrar columna ts")
    args = parser.parse_args()

    lake_kwargs = None
    if args.use_datalake:
        lake_kwargs = {
            'lake_root': args.lake_root,
            'source': args.dl_source,
            'symbol': args.dl_symbol or args.symbol,
            'tf': args.dl_tf,
            'date': args.dl_date,
            'date_from': args.dl_date_from,
            'date_to': args.dl_date_to,
            'rename_ts_to': args.dl_rename_ts_to,
            'mode': 'bulk',  # Por ahora sólo bulk en este entrypoint
        }

    run_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        market=args.market,
        limit_days=args.limit,
        out_path=Path(args.out),
        use_datalake=bool(args.use_datalake),
        lake_kwargs=lake_kwargs,
    )


if __name__ == "__main__":
    main()
