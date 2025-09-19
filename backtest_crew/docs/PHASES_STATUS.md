# Estado de Fases: Datalake -> Motor rápido + Streamlit

Este documento resume las fases completadas, sus objetivos, entregables y cómo verificar su plenitud. Aplica a los repos:
- `backtest_crew` (motor + UI + Bridge)
- `backtest_crew-datalake` (reader/ingest del Datalake)

Rama de trabajo: `feat/bridge-datalake`.

---

## Fase 0 — Preparación y validación inicial

- Objetivo: confirmar datos y entorno mínimo para pruebas.
- Entregables cumplidos:
  - Especificación de esquema mínimo Binance Parquet: `backtest_crew/docs/datalake_schema_binance.md` (columnas, tipos, layout).
  - Entorno con `pyarrow` disponible en `backtest_crew`.
  - Bridge funcional que carga un DataFrame con columna `ts` tz-aware UTC.
- Cómo verificar:
  1) Generar muestra vía CLI del Bridge (produce CSV de muestra):
     ```bash
     export PYTHONPATH=/workspaces/backtest_crew:/workspaces/backtest_crew-datalake/src
     python -m backtest_crew.bridge.cli --mode bulk --lake-root /workspaces/backtest_crew/tests/fixtures/datalake_sample \
       --source binance --market crypto --symbol BTC-USD --tf M1 --date-from 2025-08-01 --date-to 2025-08-02
     ```
     - Esperado: archivo `.tmp/bridge_sample.csv` con columnas `ts,open,high,low,close,volume` (UTC, sin duplicados).
  2) Revisar `docs/datalake_schema_binance.md` para validación de columnas y layout.

---

## Fase 1 — Requisitos de datos y sample

- Objetivo: disponer de una muestra (M1/M5) para tests locales/CI.
- Entregables cumplidos:
  - Fixture de Datalake (1 día M1 y M5) en `backtest_crew/tests/fixtures/datalake_sample`.
  - Guía en `docs/WORKSPACE_SETUP.md` para usar el fixture como `LAKE_ROOT`.
  - Script de apoyo para generar/actualizar muestra desde el lago real: `backtest_crew/tools/make_fixture_from_lake.py` (y util complementario `tools/generate_sample_datalake.py`).
- Cómo verificar:
  1) Leer el fixture desde el Bridge y validar shape/bounds:
     ```bash
     export PYTHONPATH=/workspaces/backtest_crew:/workspaces/backtest_crew-datalake/src
     pytest -q backtest_crew/tests/test_datalake_feed_fixture.py -q
     ```
     - Esperado: todos los tests en verde; para 2025-08-01: M1=1440 filas, M5=288 filas.
  2) Ejecutar el CLI del Bridge contra el fixture (como en Fase 0) y confirmar `.tmp/bridge_sample.csv`.

---

## Fase 2 — Lector eficiente y contrato

- Objetivo: reader robusto, eficiente y compatible con la estrategia.
- Entregables cumplidos:
  - Bridge `DatalakeFeed` con:
    - Clipping temporal half-open `[from, to)` y normalización `ts` a UTC.
    - Caché LRU opcional (TTL configurable) para evitar relecturas: `backtest_crew/bridge/datalake_feed.py` y flags en `backtest_crew/bridge/config.py`.
  - Optimización de lectura en el repositorio de datalake (ruta de bajo nivel consumida por el Bridge):
    - `backtest_crew-datalake/src/datalake/read/api.py` usa `pyarrow.dataset` con filtros pushdown por `ts`, proyección de columnas, y selección por particiones año/mes (fallback seguro a `pandas.read_parquet`).
  - Pruebas unitarias de contrato del Bridge con el fixture: `backtest_crew/tests/test_datalake_feed_fixture.py`.
  - Documentación de rendimiento y benchmark: `backtest_crew-datalake/docs/PERF.md` + `tools/benchmark_read.py`.
- Cómo verificar (contrato + eficiencia):
  1) Contrato del Bridge sobre el fixture:
     ```bash
     export PYTHONPATH=/workspaces/backtest_crew:/workspaces/backtest_crew-datalake/src
     pytest -q backtest_crew/tests/test_datalake_feed_fixture.py -q
     ```
     - Cubre: día completo, clipping horario, particiones faltantes, `rename_ts_to`.
  2) Eficiencia del camino de lectura (opcional, en repo datalake):
     ```bash
     python /workspaces/backtest_crew-datalake/tools/benchmark_read.py \
       --lake-root /workspaces/backtest_crew/tests/fixtures/datalake_sample \
       --market crypto --source binance --symbol BTC-USD \
       --tf M1 --date-from 2025-08-01 --date-to 2025-08-02
     ```
     - Esperado: misma cantidad de filas; tiempos coherentes con `docs/PERF.md`.
  3) Caché del Bridge (opcional): ejecutar dos veces una misma lectura y observar mejora; parámetros en `BridgeConfig(use_cache=True, cache_ttl_s=600)`.

---

## Observaciones y alcances

- No se modificaron el core del motor, la lógica de la estrategia B&R ni los inputs existentes de la UI.
- Los cambios de optimización de I/O residen en el repo de datalake y son consumidos por el Bridge.
- Para datos reales, se validó ingest con región Binance US (3 días M1/M5) y lectura vía Bridge sin errores de esquema.

---

## Próximo foco (para siguientes fases)

- Fase 3/4: inyección no intrusiva desde `ui/app.py` (bloque "Datalake") y ruta que entregue el DataFrame al motor.
- Fase 5: comparativas IB vs Datalake y documentación en `docs/VALIDATION.md`.
- Fase 6: mediciones ampliadas y switches de rendimiento en UI.

---

## Apéndice: rutas y archivos clave

- Bridge y config: `backtest_crew/bridge/datalake_feed.py`, `backtest_crew/bridge/config.py`.
- Tests Bridge: `backtest_crew/tests/test_datalake_feed_fixture.py`.
- Fixture: `backtest_crew/tests/fixtures/datalake_sample`.
- Reader Datalake: `backtest_crew-datalake/src/datalake/read/api.py`.
- Bench & perf: `backtest_crew-datalake/tools/benchmark_read.py`, `backtest_crew-datalake/docs/PERF.md`.
