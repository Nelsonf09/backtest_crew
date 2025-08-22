# timezone_handler.py
import pandas as pd
import logging
import pytz
import datetime
from datetime import time, timedelta

# Logger específico
logger = logging.getLogger(__name__)

def ensure_timezone_utc(timestamp):
    """ Asegura que un timestamp tenga timezone UTC. """
    if timestamp is None: return None
    if not isinstance(timestamp, pd.Timestamp): timestamp = pd.Timestamp(timestamp)
    if timestamp.tz is None:
        try: return timestamp.tz_localize('UTC', ambiguous='infer')
        except Exception as e_loc: logger.error(f"Fallo al localizar {timestamp} a UTC: {e_loc}"); return None # ERROR
    elif hasattr(timestamp.tz, 'zone'): # pytz
        if timestamp.tz.zone != 'UTC': return timestamp.tz_convert('UTC')
    else: # datetime.timezone
        if str(timestamp.tz) != 'UTC' and timestamp.tz != datetime.timezone.utc: return timestamp.tz_convert('UTC')
    return timestamp

class TimezoneHandler:
    """ Gestiona operaciones relacionadas con zonas horarias. """
    def __init__(self, market_tz_str='America/New_York', default_display_tz_str='America/New_York'):
        self.market_tz = pytz.timezone(market_tz_str)
        self.utc_tz = pytz.utc
        self.display_tz_str = default_display_tz_str
        self.display_tz = pytz.timezone(default_display_tz_str)
        self.market_open_time = time(9, 30); self.market_close_time = time(16, 0)
        self.premarket_start_time = time(4, 0); self.aftermarket_end_time = time(20, 0)
        logger.debug(f"TimezoneHandler inicializado: Market={market_tz_str}, Display={default_display_tz_str}") # DEBUG

    def set_display_timezone(self, display_tz_str):
        try:
            new_tz = pytz.timezone(display_tz_str)
            self.display_tz = new_tz; self.display_tz_str = display_tz_str
            logger.debug(f"Timezone display actualizada a: {display_tz_str}") # DEBUG
        except Exception as e: logger.error(f"Error actualizando timezone: {e}") # ERROR

    def filter_dataframe_by_time(self, df_utc: pd.DataFrame, use_rth: bool = False):
        """ Filtra un DataFrame (asume índice UTC) para incluir solo datos RTH. Devuelve DF filtrado en UTC."""
        if df_utc is None or df_utc.empty or not use_rth: return df_utc
        if not isinstance(df_utc.index, pd.DatetimeIndex) or df_utc.index.tz is None or str(df_utc.index.tz)!='UTC':
             logger.error("filter_dataframe_by_time espera índice DatetimeIndex UTC.") # ERROR
             return df_utc
        try:
            df_market_tz = df_utc.index.tz_convert(self.market_tz)
            time_of_day = df_market_tz.time
            rth_mask = (time_of_day >= self.market_open_time) & (time_of_day <= self.market_close_time)
            filtered_df_utc = df_utc[rth_mask]
            logger.debug(f"Filtro RTH aplicado: {len(df_utc)} -> {len(filtered_df_utc)} filas (UTC)") # DEBUG
            return filtered_df_utc
        except Exception as e: logger.error(f"Error filtrando por RTH: {e}"); return df_utc # ERROR

    def format_timestamp_for_display(self, timestamp, include_date=True):
        """Formatea un timestamp para mostrar en la interfaz."""
        if timestamp is None: return "N/A"
        try:
            timestamp_utc = ensure_timezone_utc(timestamp)
            if timestamp_utc is None: return "Error TZ"
            display_ts = timestamp_utc.tz_convert(self.display_tz)
            fmt = '%Y-%m-%d %H:%M:%S' if include_date else '%H:%M:%S'
            return display_ts.strftime(fmt)
        except Exception as e: logger.error(f"Error formateando timestamp {timestamp}: {e}"); return str(timestamp) # ERROR

# --- VERSIÓN OPTIMIZADA ---
def apply_timezone_fixes(app_state, df_data_utc: pd.DataFrame, display_tz_str: str, use_rth: bool = False):
    """ Aplica filtro RTH y convierte a timezone display. Asume entrada UTC. """
    if df_data_utc is None or df_data_utc.empty:
        logger.debug("apply_timezone_fixes: DataFrame vacío o None.") # DEBUG
        try: # Devolver DF vacío con TZ correcta
            target_tz = pytz.timezone(display_tz_str)
            cols = df_data_utc.columns if df_data_utc is not None else ['open','high','low','close','volume']
            return pd.DataFrame(columns=cols, index=pd.to_datetime([]).tz_localize('UTC').tz_convert(target_tz))
        except Exception: return pd.DataFrame()

    if not isinstance(df_data_utc.index, pd.DatetimeIndex) or df_data_utc.index.tz is None or str(df_data_utc.index.tz) != 'UTC':
         logger.error("apply_timezone_fixes: ¡Error crítico! El DataFrame de entrada NO tiene índice UTC.") # ERROR
         try: # Intentar corrección
             logger.warning("Intentando forzar índice a UTC...") # WARNING
             if not isinstance(df_data_utc.index, pd.DatetimeIndex): df_data_utc.index = pd.to_datetime(df_data_utc.index)
             if df_data_utc.index.tz is None: df_data_utc.index = df_data_utc.index.tz_localize('UTC', ambiguous='infer')
             else: df_data_utc.index = df_data_utc.index.tz_convert('UTC')
             logger.info("Índice forzado a UTC.") # INFO (porque es una corrección importante)
         except Exception as e_force_utc: logger.error(f"No se pudo forzar el índice a UTC: {e_force_utc}. Abortando."); return df_data_utc # ERROR

    try:
        if 'tz_handler' not in app_state:
            logger.warning("apply_timezone_fixes: Creando TimezoneHandler temporal.") # WARNING
            tz_handler = TimezoneHandler(default_display_tz_str=display_tz_str)
        else:
            tz_handler = app_state['tz_handler']
            if tz_handler.display_tz_str != display_tz_str: tz_handler.set_display_timezone(display_tz_str)

        df_processed_utc = df_data_utc
        if use_rth:
            logger.debug(f"apply_timezone_fixes: Aplicando filtro RTH.") # DEBUG
            df_processed_utc = tz_handler.filter_dataframe_by_time(df_data_utc, use_rth=True)
            if df_processed_utc is None or df_processed_utc.empty:
                 logger.warning("apply_timezone_fixes: DataFrame vacío después de filtro RTH.") # WARNING
                 return pd.DataFrame(columns=df_data_utc.columns, index=pd.to_datetime([]).tz_localize('UTC').tz_convert(tz_handler.display_tz))

        logger.debug(f"apply_timezone_fixes: Convirtiendo a display: {tz_handler.display_tz_str}") # DEBUG
        df_display = df_processed_utc.tz_convert(tz_handler.display_tz)
        logger.debug(f"apply_timezone_fixes: Procesamiento completo. Shape: {df_display.shape}") # DEBUG
        return df_display
    except Exception as e:
        logger.error(f"Error general en apply_timezone_fixes: {e}", exc_info=True) # ERROR
        return df_data_utc # Fallback