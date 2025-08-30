# agent_core/technical_analyzer.py
"""
Módulo para calcular indicadores técnicos y enriquecer los DataFrames de mercado.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_technical_indicators(
    df_exec: pd.DataFrame, 
    df_filter: pd.DataFrame | None = None, 
    ema_periods: list[int] = [9, 21, 50]
) -> pd.DataFrame:
    """
    Añade indicadores técnicos (EMAs) a un DataFrame de ejecución.
    
    Args:
        df_exec (pd.DataFrame): DataFrame con el timeframe de ejecución (ej. 1 min).
        df_filter (pd.DataFrame | None): DataFrame con el timeframe para el filtro EMA. 
                                         Si es None o el mismo que df_exec, las EMAs se 
                                         calculan sobre df_exec.
        ema_periods (list[int]): Lista de períodos para las EMAs a calcular.

    Returns:
        pd.DataFrame: El DataFrame df_exec enriquecido con las columnas de los indicadores.
    """
    if df_exec is None or df_exec.empty:
        return pd.DataFrame()

    df_to_enrich = df_exec.copy()
    
    source_df_for_emas = df_exec if df_filter is None or df_filter.empty else df_filter
    timeframe_source_name = "ejecución" if source_df_for_emas is df_exec else "superior"
    logger.info(f"Calculando EMAs en timeframe de {timeframe_source_name}.")

    # Calcular EMAs en el dataframe de origen (ejecución o superior)
    ema_cols = {}
    for period in ema_periods:
        col_name = f'EMA_{period}'
        ema_cols[col_name] = source_df_for_emas['close'].ewm(span=period, adjust=False).mean()
    
    df_emas = pd.DataFrame(ema_cols, index=source_df_for_emas.index)

    # Si los timeframes son diferentes, alinear y rellenar
    if source_df_for_emas is not df_exec:
        df_to_enrich = df_to_enrich.join(df_emas, how='left').ffill()
    else: # Si es el mismo timeframe, simplemente unir
        df_to_enrich = df_to_enrich.join(df_emas, how='left')

    logger.info("DataFrame enriquecido con indicadores técnicos.")
    return df_to_enrich

