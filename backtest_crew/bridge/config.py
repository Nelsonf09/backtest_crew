from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(slots=True)
class BridgeConfig:
    """Configuración para el Bridge de lectura del Datalake.

    Semántica de clipping temporal:
      - Se aplican rangos half-open [date_from, date_to) en UTC.
      - Si solo se especifica `date_from` y no `date_to`, se lee desde ese instante hasta el final disponible.
      - Si se especifica `date` (en el entrypoint externo) se traduce a
        date_from = dateT00:00:00Z y date_to = (date+1d)T00:00:00Z.

    Esta configuración NO altera la lógica de la estrategia, únicamente
    define cómo se obtienen las barras antes de entregarlas al motor.
    """
    lake_root: str                  # Raíz del datalake (el lector añadirá /data internamente o usará convención similar)
    source: str = "binance"         # Fuente / exchange dentro del datalake
    symbol: str = "BTC-USD"         # Símbolo lógico (debe coincidir con layout del datalake)
    tf: str = "M1"                  # Timeframe: M1, M5, M15, M30
    date_from: Optional[str] = None # ISO-8601 UTC inclusive
    date_to: Optional[str] = None   # ISO-8601 UTC exclusivo
    mode: Literal["bulk", "stream"] = "bulk"  # Cómo consumir los datos
    speed_bps: float = 0.0          # Barras por segundo para modo stream; 0 = sin sleep
    rename_ts_to: Optional[str] = None  # Renombrar columna 'ts' a este nombre (útil para integrar con código existente)
    # Caché (opcional) para acelerar consultas repetidas
    use_cache: bool = True
    cache_ttl_s: int = 600  # 10 minutos por defecto

    def as_kwargs(self) -> dict:
        return {
            'lake_root': self.lake_root,
            'source': self.source,
            'symbol': self.symbol,
            'tf': self.tf,
            'date_from': self.date_from,
            'date_to': self.date_to,
            'mode': self.mode,
            'speed_bps': self.speed_bps,
            'rename_ts_to': self.rename_ts_to,
            'use_cache': self.use_cache,
            'cache_ttl_s': self.cache_ttl_s,
        }
