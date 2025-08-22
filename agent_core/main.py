# agent_core/main.py

from crewai import Agent, Task, Crew, Process, LLM
import pandas as pd
import json

# Ya no necesitamos la herramienta, la estrategia se llama directamente
from strategies.opening_br_strategy import OpeningBreakRetestStrategy

# --- 1. Configuración del LLM Local (Ollama) ---
llm = LLM(model="ollama/llama3:8b", base_url="http://localhost:11434")

# --- 2. Inicialización de la Estrategia ---
# La estrategia ahora vive aquí, en el core, no dentro de una herramienta.
strategy_params = {
    "max_retest_candles": 15, "risk_reward_ratio": 2.0,
    "sl_method": "LOOKBACK_MIN_MAX", "sl_lookback": 2,
    "ema_filter_mode": "Desactivado"
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
def handle_signal_request(historical_data: pd.DataFrame, current_levels: dict) -> str | dict:
    """
    1. Calcula la señal usando la estrategia localmente.
    2. Pasa la señal al Crew para validación (en Fase 1, solo la confirma).
    3. Devuelve el resultado.
    """
    print("🧠 Strategy Core: Calculating signal locally...")
    # Paso 1: Obtenemos la señal de nuestra estrategia, que tiene acceso a todo el historial.
    signal = obr_strategy.get_signal(
        data=historical_data,
        current_day_levels=current_levels
    )
    print(f"📈 Strategy Signal: {signal}")

    # Si la señal es 'HOLD', no necesitamos molestar al agente.
    if signal == 'HOLD':
        return "HOLD"
        
    # Si hay una señal, la pasamos al agente para que la "valide".
    print("🤖 Agent Core: Signal calculated. Kicking off the crew for validation...")
    try:
        # Convertimos la señal a un string para pasarla al agente
        signal_input = json.dumps(signal) if isinstance(signal, dict) else signal

        inputs = {"signal": signal_input}
        result = deterministic_crew.kickoff(inputs=inputs)
        
        print(f"✅ Agent Core: Crew finished. Raw Result: '{result}'")
        
        # Como la señal original ya era un diccionario, la devolvemos directamente.
        # La limpieza de la salida del LLM ya no es tan crítica.
        return signal

    except Exception as e:
        print(f"❌ Agent Core: An error occurred during crew execution: {e}")
        return "HOLD"
