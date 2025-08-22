# agent_core/main.py

from crewai import Agent, Task, Crew, Process, LLM
import pandas as pd
import json
import numpy as np

# Ya no necesitamos la herramienta, la estrategia se llama directamente
from strategies.opening_br_strategy import OpeningBreakRetestStrategy

# --- INICIO: Función de Ayuda para Sanitizar Datos ---
def convert_numpy_types(obj):
    """
    Recorre recursivamente un objeto (diccionario, lista) y convierte los tipos
    de datos de NumPy a tipos nativos de Python para que sean serializables a JSON.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
# --- FIN: Función de Ayuda para Sanitizar Datos ---


# --- 1. Configuración del LLM Local (Ollama) ---
llm = LLM(model="ollama/llama3:8b", base_url="http://localhost:11434")

# --- 2. Inicialización de la Estrategia ---
# La estrategia ahora vive aquí, en el core, no dentro de una herramienta.
strategy_params = {
    "max_retest_candles": 15, "risk_reward_ratio": 2.0,
    "sl_method": "LOOKBACK_MIN_MAX", "sl_lookback": 2,
    "ema_filter_mode": "Desactivado" # Este valor se sobreescribe desde la UI
}
obr_strategy = OpeningBreakRetestStrategy(**strategy_params)


# --- 3. Definición del Agente (Ahora es un validador) ---
signal_validator = Agent(
    role='Trading Signal Validator',
    goal='Validate a trading signal. In this deterministic phase, you will always approve the signal unless it is invalid.',
    backstory=(
        "You are an AI bot that receives a pre-calculated trading signal ('BUY', 'SELL', or 'HOLD'). "
        "Your job is to simply confirm and return this signal. You do not perform the analysis yourself."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# --- 4. Definición de la Tarea ---
validation_task = Task(
    description=(
        "You have received the following trading signal: '{signal}'.\n"
        "Confirm if this is a valid signal ('BUY', 'SELL', or 'HOLD').\n"
        "Your final answer MUST be only the signal itself, without any other text."
    ),
    expected_output="The validated signal: 'BUY', 'SELL', or 'HOLD', or the signal dictionary as a JSON string.",
    agent=signal_validator,
)

# --- 5. Creación del Crew ---
deterministic_crew = Crew(
    agents=[signal_validator],
    tasks=[validation_task],
    process=Process.sequential,
    verbose=True
)

# --- 6. Función de Entrada Principal ---
def handle_signal_request(historical_data: pd.DataFrame, current_levels: dict, reset_strategy: bool = False) -> str | dict:
    """
    1. Resetea la estrategia si se solicita.
    2. Calcula la señal usando la estrategia localmente.
    3. Pasa la señal al Crew para validación.
    4. Devuelve el resultado.
    """
    # --- INICIO CORRECCIÓN: Lógica de Reseteo ---
    if reset_strategy:
        obr_strategy.reset()
        return "RESET_OK"
    # --- FIN CORRECCIÓN ---

    # Si la llamada no es para resetear, se necesitan los datos.
    if historical_data is None or historical_data.empty:
        return 'HOLD'

    print("🧠 Strategy Core: Calculating signal locally...")
    signal = obr_strategy.get_signal(
        data=historical_data,
        current_day_levels=current_levels
    )
    print(f"📈 Strategy Signal: {signal}")

    if signal == 'HOLD':
        return "HOLD"
        
    print("🤖 Agent Core: Signal calculated. Kicking off the crew for validation...")
    try:
        sanitized_signal = convert_numpy_types(signal)
        signal_input = json.dumps(sanitized_signal)

        inputs = {"signal": signal_input}
        result = deterministic_crew.kickoff(inputs=inputs)
        
        print(f"✅ Agent Core: Crew finished. Raw Result: '{result}'")
        
        return sanitized_signal

    except Exception as e:
        print(f"❌ Agent Core: An error occurred during crew execution: {e}")
        return "HOLD"
