# Bridge Datalake (Opcional)

Este Bridge permite consumir datos históricos (bulk) o simulados en tiempo real (stream) desde el Datalake sin alterar la lógica de las estrategias existentes. Si no se activa la bandera `--use-datalake`, el comportamiento del motor se mantiene idéntico.

## Requisitos

1. Variable de entorno `LAKE_ROOT` apuntando a la raíz del vendor:

```bash
export LAKE_ROOT="/workspaces/backtest_crew/vendor/datalake"
```

2. Layout esperado (ejemplo simplificado):
```
{LAKE_ROOT}/data/source=binance/market=crypto/timeframe=M1/symbol=BTC-USD/year=2025/...
```

## Clipping temporal
Los rangos se interpretan como half-open `[from, to)` en UTC. El lector interno ya aplica esta semántica, y el Bridge refuerza el recorte por seguridad.

## CLI de prueba

Bulk (guarda muestra en `.tmp/bridge_sample.csv`):
```bash
python -m backtest_crew.bridge.cli \
  --lake-root "$LAKE_ROOT" \
  --source binance --symbol BTC-USD --tf M1 \
  --date-from 2025-08-01T00:00:00Z --date-to 2025-08-02T00:00:00Z \
  --mode bulk --rename-ts-to timestamp
```

Stream (primeras 10 barras, sin delay):
```bash
python -m backtest_crew.bridge.cli \
  --lake-root "$LAKE_ROOT" \
  --source binance --symbol BTC-USD --tf M1 \
  --date-from 2025-08-01T14:30:00Z --date-to 2025-08-01T14:40:00Z \
  --mode stream --speed-bps 0
```

## Integración con el motor
Se añaden flags opcionales (ejemplo):
```
--use-datalake --lake-root $LAKE_ROOT --source binance --symbol BTC-USD \
--tf M1 --date 2025-08-01
```
O alternativamente `--date-from` y `--date-to`.

Si `--use-datalake` es falso/ausente, no se modifica nada del pipeline actual.

## Conteos esperados por timeframe (1 día completo UTC)
- M1: 1440
- M5: 288
- M15: 96
- M30: 48

Los tests de contrato validan estos conteos, orden temporal, ausencia de duplicados y cumplimiento del rango `[from,to)`.

## Modo Bulk vs Stream
- Bulk: entrega un `DataFrame` completo listo para cálculos vectorizados.
- Stream: itera barra a barra (útil para simular llegada en tiempo real). El parámetro `--speed-bps` controla la cadencia (0 = sin pausa).

## Renombrar columna de timestamp
`--rename-ts-to timestamp` crea una columna adicional duplicando `ts`. No elimina `ts` original.

## Garantías
- No se toca la lógica de estrategias ni cálculos de métricas.
- El Bridge solo sustituye la fuente de datos a petición explícita.
