#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import os
import sys
import shutil

try:
    from backtest_crew.bridge.config import BridgeConfig
    from backtest_crew.bridge.datalake_feed import DatalakeFeed
except Exception:
    # Permitir ejecución directa ajustando el path del proyecto
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backtest_crew.bridge.config import BridgeConfig
    from backtest_crew.bridge.datalake_feed import DatalakeFeed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Crear fixture de Datalake (1 día M1/M5) para tests")
    ap.add_argument("--lake-root", required=True, help="Root del datalake fuente")
    ap.add_argument("--out-root", required=True, help="Root destino del fixture (p. ej. tests/fixtures/datalake_sample)")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--date", default="2025-08-01", help="Día UTC YYYY-MM-DD")
    ap.add_argument("--tfs", default="M1,M5")
    return ap.parse_args()


def write_parquet_partition(df, out_root: Path, source: str, market: str, tf: str, symbol: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datetime import timezone
    df = df.copy()
    df['ts'] = df['ts'].dt.tz_convert(timezone.utc)
    df['tf'] = tf
    df['symbol'] = symbol
    df['source'] = source
    # Infer year/month from first row
    year = int(df['ts'].dt.year.iloc[0])
    month = int(df['ts'].dt.month.iloc[0])
    base = (
        out_root / "data" / f"source={source}" / f"market={market}" /
        f"timeframe={tf}" / f"symbol={symbol}" / f"year={year}" / f"month={month:02d}"
    )
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"part-{year}-{month:02d}.parquet"
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    pq.write_table(table, dest, compression="zstd", version="2.6", use_dictionary=False)
    return dest


def main() -> int:
    args = parse_args()
    lake_root = Path(args.lake_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Copiar metadata mínima (opcional)
    for extra in ("README.md",):
        p = lake_root / extra
        if p.exists():
            shutil.copy2(p, out_root / extra)

    tfs = [t.strip().upper() for t in args.tfs.split(',') if t.strip()]
    for tf in tfs:
        cfg = BridgeConfig(
            lake_root=str(lake_root), source="binance", symbol=args.symbol, tf=tf,
            date_from=f"{args.date}T00:00:00Z", date_to=f"{args.date}T23:59:59Z",
        )
        df = DatalakeFeed().load_df(cfg)
        if df is None or df.empty:
            print(f"[WARN] Sin datos para {args.symbol} {tf} {args.date}")
            continue
        dest = write_parquet_partition(df, out_root, source="binance", market="crypto", tf=tf, symbol=args.symbol)
        print(f"[OK] Escrito: {dest}")

    print("Fixture listo en:", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
