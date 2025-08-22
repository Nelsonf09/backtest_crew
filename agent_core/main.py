# agent_core/main.py

from crewai import Agent, Task, Crew, Process
# Usamos la importación más moderna y específica para Ollama
from langchain_ollama import ChatOllama
import pandas as pd
import json

from .tools.strategy_tools import OBRStrategyTool

# --- 1. Configuración del LLM Local (Ollama) ---
# Esta es la forma más robusta y recomendada.
# Asegúrate de que Ollama esté corriendo en segundo plano.
llm = ChatOllama(model="llama3:8b")

# --- 2. Inicialización de Herramientas ---
obr_tool = OBRStrategyTool()

# --- 3. Definición del Agente Determinista ---
market_analyst = Agent(
    role='Expert Market Analyst for OBR Strategy',
    goal='Execute the OBR trading strategy tool to determine the correct trading signal (BUY, SELL, or HOLD).',
    backstory=(
        "You are a specialized AI agent designed exclusively to run a deterministic trading strategy. "
        "You receive market data and apply the OBR Strategy Tool without deviation. "
        "Your output must be ONLY the direct result from the tool."
    ),
    tools=[obr_tool],
    # Pasamos el LLM explícitamente al agente
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# --- 4. Definición de la Tarea ---
analysis_task = Task(
    description=(
        "1. Take the historical market data (provided as a JSON string) and the current day's key levels. "
        "2. Use the 'OBR Strategy Analysis Tool' to get the trading signal. "
        "3. Return the exact signal provided by the tool."
    ),
    expected_output="The trading signal, which can be 'BUY', 'SELL', 'HOLD', or a dictionary with trade details.",
    agent=market_analyst,
)

# --- 5. Creación del Crew ---
deterministic_crew = Crew(
    agents=[market_analyst],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True
)

# --- 6. Función de Entrada Principal ---
def handle_signal_request(historical_data: pd.DataFrame, current_levels: dict) -> str | dict:
    """
    Recibe los datos de la vela, ejecuta el Crew para obtener una señal y devuelve el resultado.
    """
    print("🤖 Agent Core: Signal request received. Kicking off the crew...")
    try:
        historical_data_json = historical_data.to_json(orient='split', date_format='iso')
        
        inputs = {
            "historical_data_json_string": historical_data_json,
            "current_day_levels": current_levels
        }
        result = deterministic_crew.kickoff(inputs=inputs)
        print(f"✅ Agent Core: Crew finished. Result: {result}")
        return result
    except Exception as e:
        print(f"❌ Agent Core: An error occurred during crew execution: {e}")
        return "HOLD"
