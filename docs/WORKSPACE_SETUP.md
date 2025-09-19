# Configuración del área de trabajo (WORKSPACE_SETUP)

Este documento explica cómo añadir ambas repos (`backtest_crew` y `backtest_crew-datalake`) a un mismo área de trabajo en VS Code, cómo activar entornos virtuales, instalar dependencias y ejecutar tareas comunes.

Requisitos
- Tener Git y Python 3.11+ instalados.
- VS Code con la extensión de Python instalada.

1) Añadir ambas repos al área de trabajo

- Abre VS Code.
- Ve a `File -> Add Folder to Workspace...` y añade las carpetas de los dos repos locales: la carpeta del proyecto `backtest_crew` y `backtest_crew-datalake`.
- Guarda el workspace: `File -> Save Workspace As...` y guarda como `backtest_crew.code-workspace` en una ubicación de tu preferencia.

Estructura recomendada del workspace

- root/
  - backtest_crew/ (repositorio principal)
  - backtest_crew-datalake/ (repositorio datalake)

2) Entornos virtuales y dependencias

Se recomiendan dos opciones para manejar dependencias: `venv` (incluido en Python) o `poetry`.

- Opción A — `venv` (por carpeta):

  - Crear un entorno para `backtest_crew`:

    ```bash
    cd /path/to/backtest_crew
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    ```

  - Crear un entorno para `backtest_crew-datalake`:

    ```bash
    cd /path/to/backtest_crew-datalake
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    ```

  - En VS Code: selecciona el intérprete del entorno activo (`.venv`) desde la paleta de comandos `Python: Select Interpreter` para cada carpeta.

- Opción B — `poetry` (si ambos repos usan `pyproject.toml`):

  - Instalar `poetry` si no está instalado:

    ```bash
    curl -sSL https://install.python-poetry.org | python -
    ```

  - En `backtest_crew`:

    ```bash
    cd /path/to/backtest_crew
    poetry install
    poetry shell
    ```

  - En `backtest_crew-datalake`:

    ```bash
    cd /path/to/backtest_crew-datalake
    poetry install
    poetry shell
    ```

3) Ejecución de tareas comunes

- Ejecutar tests (ejemplo con `pytest`):

  - Con `venv`:

    ```bash
    cd /path/to/backtest_crew
    source .venv/bin/activate
    pytest -q
    ```

  - Con `poetry`:

    ```bash
    cd /path/to/backtest_crew
    poetry run pytest -q
    ```

- Ejecutar un backtest de ejemplo (script):

  ```bash
  cd /path/to/backtest_crew/tools
  source ../.venv/bin/activate   # o usa `poetry run`
  python run_backtest.py --help
  ```

- Ejecutar el datalake localmente (si aplica):

  ```bash
  cd /path/to/backtest_crew-datalake
  source .venv/bin/activate
  # comandos de ingest/reader según README del datalake
  ```

4) Configuración recomendada de VS Code

- Workspace settings (`.code-workspace` o `.vscode/settings.json`):

  - Asegúrate de que el `python.pythonPath` (o `python.defaultInterpreterPath`) apunte a la `./.venv/bin/python` para cada carpeta.
  - Configura linting y formateo según preferencias: `pylint`/`flake8` y `black`.

5) Consejos para trabajar con ambos repos

- Mantén los entornos separados para evitar conflictos de dependencias.
- Si necesitas que el código del datalake sea importable desde `backtest_crew`, usa `pip install -e ../backtest_crew-datalake` o configura `PYTHONPATH` en el entorno de desarrollo.

  Ejemplo:

  ```bash
  cd /path/to/backtest_crew
  source .venv/bin/activate
  pip install -e ../backtest_crew-datalake
  ```

6) Atajos y comandos útiles

- Instalar dependencias desde el workspace raíz (ejemplo):

  ```bash
  # desde el directorio que contiene ambas carpetas
  for d in backtest_crew backtest_crew-datalake; do
    cd "$d"
    if [ -f pyproject.toml ]; then
      poetry install
    elif [ -f requirements.txt ]; then
      python -m venv .venv
      source .venv/bin/activate
      pip install -r requirements.txt
      deactivate
    fi
    cd ..
  done
  ```

6.1) Comandos útiles detallados por repositorio

- backtest_crew (comandos frecuentes):

  - Crear y activar entorno (venv):

    ```bash
    cd /path/to/backtest_crew
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    ```

  - Usando `poetry`:

    ```bash
    cd /path/to/backtest_crew
    poetry install
    poetry shell
    ```

  - Ejecutar todos los tests:

    ```bash
    # con venv
    source .venv/bin/activate
    pytest -q

    # con poetry
    poetry run pytest -q
    ```

  - Ejecutar tests específicos (por archivo o marcador):

    ```bash
    pytest -q tests/test_drawdown_unificado.py
    pytest -q -k "forex and not slow"
    ```

  - Ejecutar un backtest rápido (ejemplo):

    ```bash
    cd tools
    source ../.venv/bin/activate
    python run_backtest.py --symbol BTC-USD --market crypto --timeframe "1 min" --limit 1
    ```

  - Uso del Bridge (bulk sample):

    ```bash
    export LAKE_ROOT=/path/to/backtest_crew/vendor/datalake
    python -m backtest_crew.bridge.cli \
      --lake-root "$LAKE_ROOT" \
      --source binance --symbol BTC-USD --tf M1 \
      --date-from 2025-08-01T00:00:00Z --date-to 2025-08-02T00:00:00Z \
      --mode bulk --rename-ts-to timestamp
    ```

  - Comandos de lint/format:

    ```bash
    # format con black
    black .

    # lint con flake8
    flake8
    ```

- backtest_crew-datalake (comandos frecuentes):

  - Crear y activar entorno (venv):

    ```bash
    cd /path/to/backtest_crew-datalake
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    ```

  - Usando `poetry`:

    ```bash
    cd /path/to/backtest_crew-datalake
    poetry install
    poetry shell
    ```

  - Ejecutar tests del datalake (ejemplos):

    ```bash
    # con venv
    source .venv/bin/activate
    pytest -q tests/test_read_api.py

    # con poetry
    poetry run pytest -q tests/test_read_api.py
    ```

  - Ingest / fetch (scripts útiles):

    ```bash
    # fetch tail de binance (ejemplo)
    python tools/binance_fetch_tail.py --help

    # resample desde M1 a otros TF (ejemplo)
    python tools/resample_from_m1.py --help
    ```

  - Instalar datalake local editable (para importar desde backtest_crew):

    ```bash
    cd /path/to/backtest_crew-datalake
    source .venv/bin/activate
    pip install -e .
    ```

  - Comandos de lint/format:

    ```bash
    black .
    flake8
    ```

7) Solución de problemas

- Si VS Code no detecta el intérprete: asegúrate de que la carpeta del proyecto esté abierta en el workspace y recarga la ventana (`Developer: Reload Window`).
- Si hay conflictos de dependencias, prefiere mantener entornos separados y usar `pip install -e` para dependencias locales.

8) Recursos

- Documentación de VS Code Python: https://code.visualstudio.com/docs/python
- Poetry: https://python-poetry.org/

---

Si quieres, puedo adaptar este documento con pasos específicos para tu flujo (por ejemplo, comandos para ejecutar un test en particular o configurar tareas de VS Code).
