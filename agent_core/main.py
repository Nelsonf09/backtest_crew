# agent_core/main.py

from crewai import Agent, Task, Crew, Process, LLM
import pandas as pd
import json
import numpy as np
import logging
from shared.liquidity_stamper import stamp_liquidity_window

from strategies.opening_br_strategy import OpeningBreakRetestStrategy

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
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

llm = LLM(model="ollama/llama3:8b", base_url="http://localhost:11434")

strategy_params = {
    "max_retest_candles": 15, "risk_reward_ratio": 2.0,
    "sl_method": "LOOKBACK_MIN_MAX", "sl_lookback": 2,
    "ema_filter_mode": "Desactivado"
}
obr_strategy = OpeningBreakRetestStrategy(**strategy_params)

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

validation_task = Task(
    description=(
        "You have received the following trading signal: '{signal}'.\n"
        "Confirm if this is a valid signal ('BUY', 'SELL', or 'HOLD').\n"
        "Your final answer MUST be only the signal itself, without any other text."
    ),
    expected_output="The validated signal: 'BUY', 'SELL', or 'HOLD', or the signal dictionary as a JSON string.",
    agent=signal_validator,
)

deterministic_crew = Crew(
    agents=[signal_validator],
    tasks=[validation_task],
    process=Process.sequential,
    verbose=True
)

def handle_signal_request(historical_data: pd.DataFrame, current_levels: dict, reset_strategy: bool = False, ema_filter_mode: str = "Desactivado", daily_candle_index: int = -1, *, or_window: str | None = None, market: str | None = None) -> str | dict:
    """
    Función principal que recibe la solicitud, actualiza la estrategia y obtiene la señal.
    """
    if reset_strategy:
        obr_strategy.reset()
        return "RESET_OK"

    if historical_data is None or historical_data.empty:
        return 'HOLD'

    if ('in_opening_window' not in historical_data.columns) or or_window:
        historical_data = stamp_liquidity_window(historical_data, market or 'stocks', or_window)

    obr_strategy.ema_filter_mode = ema_filter_mode.capitalize()
    obr_strategy.use_ema_filter = obr_strategy.ema_filter_mode != 'Desactivado'

    signal = obr_strategy.get_signal(
        data=historical_data,
        current_day_levels=current_levels,
        daily_candle_index=daily_candle_index # <-- CORRECCIÓN: Pasar el índice diario
    )

    if signal == 'HOLD':
        return "HOLD"
        
    try:
        sanitized_signal = convert_numpy_types(signal)
        signal_input = json.dumps(sanitized_signal)
        inputs = {"signal": signal_input}
        result = deterministic_crew.kickoff(inputs=inputs)
        return sanitized_signal
    except Exception as e:
        logger.error(f"Error en la ejecución del Crew: {e}")
        return "HOLD"
