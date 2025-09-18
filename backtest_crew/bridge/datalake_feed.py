from __future__ import annotations
import os
import time
import logging
import sys
from pathlib import Path
from typing import Iterator
import pandas as pd

from .config import BridgeConfig

logger = logging.getLogger(__name__)

def _attempt_import_reader():
    try:  # Primer intento directo
        from datalake.read.api import read_range_df  # type: ignore
        return read_range_df  # type: ignore
    except Exception:
        # Intento de auto-inyección de ruta vendor si existe estructura conocida.
        lake_env = os.getenv('LAKE_ROOT')
        candidate = None
        if lake_env:
            candidate = Path(lake_env) / 'src'
        else:
            # Ruta relativa típica dentro del repo backtest_crew
            candidate = Path(__file__).resolve().parents[2] / 'vendor' / 'datalake' / 'src'
        if candidate and candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            try:
                from datalake.read.api import read_range_df  # type: ignore
                return read_range_df  # type: ignore
            except Exception:
                pass
    return None

read_range_df = _attempt_import_reader()
if read_range_df is None:  # pragma: no cover
    logger.warning("No se pudo importar 'read_range_df' desde datalake.read.api tras intentos. El Bridge permanecerá inerte.")

REQUIRED_COLS = ["open", "high", "low", "close"]

class DatalakeFeed:
    """Feed de datos basado en el datalake estructurado.

    Responsabilidades:
      - Leer un rango de barras parquet a DataFrame.
      - Garantizar orden temporal y unicidad de 'ts'.
      - Aplicar clipping extra [from,to) como defensa (el reader ya debería hacerlo).
      - Proveer iteración tipo stream (fila a fila) opcional con throttling.
    """

    def load_df(self, cfg: BridgeConfig) -> pd.DataFrame:
        if read_range_df is None:
            raise RuntimeError("El módulo datalake.read.api no está disponible en el entorno.")

        if not cfg.lake_root:
            env_root = os.getenv("LAKE_ROOT", "")
            if env_root:
                cfg.lake_root = env_root
            else:
                raise ValueError("Se requiere lake_root (o variable de entorno LAKE_ROOT).")

        logger.info(
            "Bridge leyendo datalake: root=%s source=%s market=crypto tf=%s symbol=%s from=%s to=%s", 
            cfg.lake_root, cfg.source, cfg.tf, cfg.symbol, cfg.date_from, cfg.date_to
        )

        df = read_range_df(
            lake_root=cfg.lake_root,
            market="crypto",
            tf=cfg.tf,
            symbol=cfg.symbol,
            date_from=cfg.date_from,
            date_to=cfg.date_to,
            source=cfg.source,
        )
        if df is None or df.empty:
            logger.warning("Bridge: DataFrame vacío para rango solicitado.")
            return pd.DataFrame()

        # Normalizamos y aseguramos el tipo de ts.
        if 'ts' not in df.columns:
            raise ValueError("El DataFrame devuelto no contiene columna 'ts'.")
        df['ts'] = pd.to_datetime(df['ts'], utc=True)

        # Orden y dedupe explícitos.
        before = len(df)
        df = df.sort_values('ts').drop_duplicates(subset='ts', keep='first')
        after = len(df)
        if after != before:
            logger.info("Bridge: deduplicadas %d filas (restan %d).", before - after, after)

        # Validamos columnas básicas OHLCV.
        self._validate_columns(df)

        # Clipping defensivo half-open.
        df = self._clip_exclusive(df, cfg.date_from, cfg.date_to)

        # Renombrado opcional de ts.
        if cfg.rename_ts_to and cfg.rename_ts_to != 'ts':
            df[cfg.rename_ts_to] = df['ts']
            # Conservamos 'ts' para trazabilidad a menos que se especifique lo contrario.

        logger.info(
            "Bridge: rango final ts=[%s .. %s] filas=%d", 
            df['ts'].min(), df['ts'].max(), len(df)
        )
        return df

    def iter_bars(self, cfg: BridgeConfig) -> Iterator[pd.Series]:
        df = self.load_df(cfg)
        if df.empty:
            return iter(())
        sleep_s = 0.0 if cfg.speed_bps <= 0 else 1.0 / float(cfg.speed_bps)
        for _, row in df.iterrows():
            yield row
            if sleep_s > 0:
                time.sleep(sleep_s)

    # --- Utilidades internas ---
    def _clip_exclusive(self, df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
        if not start and not end:
            return df
        start_ts = pd.to_datetime(start, utc=True) if start else None
        end_ts = pd.to_datetime(end, utc=True) if end else None
        # Usamos un array boolean alineado al índice para evitar problemas de indexación
        mask = pd.Series(True, index=df.index)
        if start_ts is not None:
            mask &= df['ts'] >= start_ts
        if end_ts is not None:
            mask &= df['ts'] < end_ts  # exclusivo
        return df.loc[mask]

    def _validate_columns(self, df: pd.DataFrame):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas en DataFrame: {missing}")
        # Volume puede faltar, lo rellenamos.
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        # Tipos numéricos
        for c in REQUIRED_COLS + ['volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.dropna(subset=REQUIRED_COLS, inplace=True)
