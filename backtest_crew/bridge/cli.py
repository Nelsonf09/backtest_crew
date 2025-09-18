from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd

from .config import BridgeConfig
from .datalake_feed import DatalakeFeed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CLI Bridge Datalake -> DataFrame/Stream")
    ap.add_argument("--lake-root", required=False, help="Ruta root del datalake (fallback LAKE_ROOT env)")
    ap.add_argument("--source", default="binance")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--tf", default="M1", choices=["M1", "M5", "M15", "M30"])
    ap.add_argument("--date-from", dest="date_from", help="Inicio ISO-8601 UTC inclusive")
    ap.add_argument("--date-to", dest="date_to", help="Fin ISO-8601 UTC exclusivo")
    ap.add_argument("--mode", choices=["bulk", "stream"], default="bulk")
    ap.add_argument("--speed-bps", type=float, default=0.0, help="Barras por segundo en modo stream (0 = sin sleep)")
    ap.add_argument("--rename-ts-to", dest="rename_ts_to", help="Renombrar columna ts a este nombre")
    return ap


def main():  # pragma: no cover - CLI manual
    ap = build_parser()
    args = ap.parse_args()

    lake_root = args.lake_root or os.getenv("LAKE_ROOT")
    if not lake_root:
        ap.error("Debe proporcionar --lake-root o variable LAKE_ROOT")

    cfg = BridgeConfig(
        lake_root=lake_root,
        source=args.source,
        symbol=args.symbol,
        tf=args.tf,
        date_from=args.date_from,
        date_to=args.date_to,
        mode=args.mode,
        speed_bps=args.speed_bps,
        rename_ts_to=args.rename_ts_to,
    )

    feed = DatalakeFeed()

    if cfg.mode == "bulk":
        df = feed.load_df(cfg)
        if df.empty:
            print("[bridge] DataFrame vacío.")
            return
        print(f"[bridge] Filas={len(df)} ts_min={df['ts'].min()} ts_max={df['ts'].max()}")
        out_dir = Path('.tmp')
        out_dir.mkdir(exist_ok=True)
        sample_path = out_dir / 'bridge_sample.csv'
        df.head(100).to_csv(sample_path, index=False)
        print(f"[bridge] Muestra guardada en {sample_path}")
    else:
        print("[bridge] Streaming primeras 10 barras:")
        for i, row in enumerate(feed.iter_bars(cfg)):
            if i >= 10:
                break
            ts_col = cfg.rename_ts_to or 'ts'
            print(f"{i+1:02d} {row[ts_col]} close={row['close']}")

if __name__ == "__main__":  # pragma: no cover
    main()
