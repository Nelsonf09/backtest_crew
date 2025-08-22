# agent_core/tools/strategy_tools.py

from crewai.tools import BaseTool
import pandas as pd
import json

from strategies.opening_br_strategy import OpeningBreakRetestStrategy

# Parámetros de la estrategia
strategy_params = {
    "max_retest_candles": 15,
    "risk_reward_ratio": 2.0,
    "sl_method": "LOOKBACK_MIN_MAX",
    "sl_lookback": 2,
    "ema_filter_mode": "Desactivado"
}
obr_strategy = OpeningBreakRetestStrategy(**strategy_params)

class OBRStrategyTool(BaseTool):
    name: str = "OBR Strategy Analysis Tool"
    description: str = "Executes the Opening Break & Retest (OBR) strategy to get a trading signal (BUY, SELL, or HOLD)."

    def _run(self, historical_data_json_string: str, current_day_levels: dict) -> str | dict:
        """
        Ejecuta la estrategia OBR con los datos y niveles proporcionados.
        """
        try:
            # <--- CAMBIO CLAVE: Reconstruimos el DataFrame a partir del string JSON.
            # El orient='split' es importante para que coincida con cómo lo guardamos.
            data = json.loads(historical_data_json_string)
            historical_data_for_signal = pd.DataFrame(data['data'], index=pd.to_datetime(data['index']), columns=data['columns'])
            
            # Asegurarse de que el índice tenga el timezone correcto (UTC)
            if historical_data_for_signal.index.tz is None:
                historical_data_for_signal.index = historical_data_for_signal.index.tz_localize('UTC')

        except Exception as e:
            print(f"Error parsing historical data JSON in tool: {e}")
            return f"HOLD - Error parsing historical data: {e}"

        # A partir de aquí, la lógica es la misma de antes.
        signal = obr_strategy.get_signal(
            data=historical_data_for_signal,
            current_day_levels=current_day_levels
        )
        return signal
