Esquema mínimo — Datalake Binance (crypto)

Este documento define el contrato mínimo de datos para leer velas de Binance desde el Datalake con el Bridge.

- Columnas requeridas: `ts` (datetime UTC tz-aware), `open`, `high`, `low`, `close`, `volume` (float64).
- Columnas de control recomendadas: `symbol` (p. ej. BTC-USD), `tf` (M1|M5|M15|M30), `source` (binance), `exchange` (BINANCE o BINANCE.US).

Particionamiento bajo `LAKE_ROOT`:
`data/source=binance/market=crypto/timeframe=<TF>/symbol=<SYMBOL>/year=YYYY/month=MM/part-YYYY-MM.parquet`

Reglas:
- `ts` en UTC. Dedupe por `(symbol, tf, ts, source)`.
- Clipping de lectura half-open `[date_from, date_to)`.

Validación rápida (CLI Bridge):
```
PYTHONPATH=/workspaces/backtest_crew:/workspaces/backtest_crew-datalake/src \
LAKE_ROOT=/workspaces/backtest_crew/vendor/datalake \
python -m backtest_crew.bridge.cli \
  --lake-root $LAKE_ROOT --source binance --symbol BTC-USD \
  --tf M1 --date-from 2025-08-01 --date-to 2025-08-02 --mode bulk
```
Deberías ver ~1440 filas para M1 por día.
