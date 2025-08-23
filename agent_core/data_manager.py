# agent_core/data_manager.py
"""
Módulo DataManager refactorizado para usar una FSM que gestione el estado de la conexión a IB.
Esto mejora la robustez y previene intentos de descarga de datos cuando no hay una conexión activa.
"""
import logging
from pathlib import Path
import pandas as pd
import datetime
from ib_insync import IB, util, Contract, Forex, Stock, Future, Index
import config
import pytz
import math

from shared.fsm import FSM, ConnectionState # <-- IMPORTAMOS LA FSM

logger = logging.getLogger(__name__)

def calculate_ib_duration(start_date: datetime.date, end_date: datetime.date) -> str:
    """Calcula la cadena de duración para IB basada en un rango de fechas."""
    if start_date > end_date: return "1 D"
    delta_days = (end_date - start_date).days + 1
    if delta_days <= 0: return "1 D"
    elif delta_days <= 30: return f"{delta_days} D"
    elif delta_days <= 365 * 2: return f"{math.ceil(delta_days / 7)} W"
    else: return f"{math.ceil(delta_days / 365.25)} Y"

class DataManager:
    """ Gestiona conexión IB (con FSM), datos históricos, caché y niveles clave. """
    def __init__(self):
        self.ib = IB()
        self.config = config
        self.cache_dir = getattr(config, 'CACHE_DIR', Path('cache').resolve())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.MARKET_TZ = pytz.timezone(getattr(config, 'MARKET_TIMEZONE', 'America/New_York'))
        self.UTC_TZ = pytz.utc
        
        self.connection_fsm = FSM(ConnectionState.DISCONNECTED)
        logger.debug(f"DataManager (FSM) init: Cache={self.cache_dir}, Estado Conexión={self.connection_fsm.state}")

    def _handle_ib_error(self, reqId, errorCode, errorString, contract):
        """Manejador de errores de IB que puede cambiar el estado de la FSM."""
        ignored_codes = {162, 200, 2100, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2119, 2137, 2150, 2158, 300, 366}
        if errorCode not in ignored_codes:
            logger.error(f"IB Error: Req={reqId}, Code={errorCode}, Msg={errorString}")
            # Errores que indican una desconexión o problema de conexión
            if errorCode in {502, 504, 509, 1100, 1101, 1102, 1300, 2109}:
                self.connection_fsm.transition_to(ConnectionState.ERROR)

    def connect_ib(self):
        """Conecta a IB usando la FSM para gestionar el estado."""
        if self.connection_fsm.is_in_state(ConnectionState.CONNECTED):
            logger.debug("Ya conectado a IB.")
            return True
        if self.connection_fsm.is_in_state(ConnectionState.CONNECTING):
            logger.warning("Conexión ya en progreso.")
            return False

        self.connection_fsm.transition_to(ConnectionState.CONNECTING)
        try:
            host = getattr(self.config, 'IB_HOST', '127.0.0.1')
            port = getattr(self.config, 'IB_PORT', 7497)
            client_id = getattr(self.config, 'IB_CLIENT_ID', 1)
            
            try: self.ib.errorEvent -= self._handle_ib_error
            except Exception: pass
            self.ib.errorEvent += self._handle_ib_error

            self.ib.connect(host, port, clientId=client_id, timeout=15)
            if self.ib.isConnected():
                self.connection_fsm.transition_to(ConnectionState.CONNECTED)
                logger.info(f"Conexión IB OK. ServerTime: {self.ib.reqCurrentTime()}")
                return True
            else:
                self.connection_fsm.transition_to(ConnectionState.ERROR)
                logger.error("Fallo conexión IB (isConnected() es False).")
                return False
        except Exception as e:
            self.connection_fsm.transition_to(ConnectionState.ERROR)
            logger.error(f"Error genérico conexión IB: {e}", exc_info=True)
            return False

    def disconnect_ib(self):
        """Desconecta de IB usando la FSM."""
        if self.connection_fsm.is_in_state(ConnectionState.DISCONNECTED):
            return
        
        self.connection_fsm.transition_to(ConnectionState.DISCONNECTING)
        if self.ib.isConnected():
            self.ib.disconnect()
        self.connection_fsm.transition_to(ConnectionState.DISCONNECTED)
        logger.info("Desconexión IB OK.")

    def _fetch_data_core(self, contract: Contract, end_datetime_str: str, duration_str: str, timeframe: str, rth: bool, what_to_show: str) -> pd.DataFrame:
        """Núcleo de la descarga de datos, protegido por la FSM de conexión."""
        if not self.connection_fsm.is_in_state(ConnectionState.CONNECTED):
            logger.error("No se puede hacer fetch de datos, no hay conexión a IB.")
            return pd.DataFrame()

        logger.info(f"Solicitando datos IB: Cont={contract.symbol}, End='{end_datetime_str}', Dur='{duration_str}', TF='{timeframe}'")
        try:
            bars = self.ib.reqHistoricalData(
                contract=contract, endDateTime=end_datetime_str, durationStr=duration_str,
                barSizeSetting=timeframe, whatToShow=what_to_show, useRTH=rth,
                formatDate=2, keepUpToDate=False, timeout=60.0
            )
            if not bars: 
                logger.warning(f"No se recibieron barras de IB para {contract.symbol}")
                return pd.DataFrame()
            
            df = util.df(bars)
            if df is None or df.empty: return pd.DataFrame()
            
            df = df.set_index('date')
            
            # --- CORRECCIÓN AQUÍ ---
            # Asegurarse de que el índice sea DatetimeIndex antes de manipular zonas horarias.
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            if df.index.tz is None: 
                df.index = df.index.tz_localize('UTC')
            elif str(df.index.tz).upper() != 'UTC': 
                df.index = df.index.tz_convert('UTC')
            
            df = df.rename(columns={'open':'open','high':'high','low':'low','close':'close','volume':'volume'})
            required_cols = ['open', 'high', 'low', 'close']
            if 'volume' not in df.columns: df['volume'] = 0.0
            for col in required_cols + ['volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required_cols)
            
            logger.info(f"Datos IB recibidos y procesados (UTC) para {contract.symbol}. {len(df)} filas.")
            return df
        except Exception as e:
            logger.error(f"Error inesperado en _fetch_data_core para {contract.symbol}: {e}", exc_info=True)
            self.connection_fsm.transition_to(ConnectionState.ERROR)
            return pd.DataFrame()

    def _resolve_contract(self, symbol, sec_type, exchange, currency, primary_exchange=None, lastTradeDateOrContractMonth=''):
        sec_type=sec_type.upper()
        contract = None
        if sec_type == 'FOREX': contract=Forex(f'{symbol}') 
        elif sec_type == 'STK':
            contract=Stock(symbol, exchange, currency)
            if primary_exchange: contract.primaryExchange = primary_exchange
        elif sec_type == 'FUT': contract=Future(symbol, lastTradeDateOrContractMonth=lastTradeDateOrContractMonth, exchange=exchange, currency=currency)
        elif sec_type == 'IND': contract=Index(symbol, exchange, currency)
        elif sec_type == 'CFD': contract=Contract(symbol=symbol, secType='CFD', exchange=exchange, currency=currency)
        elif sec_type == 'CRYPTO':
              if not exchange or exchange == 'SMART': raise ValueError("Exchange específico requerido para CRYPTO")
              contract = Contract(symbol=symbol, secType='CRYPTO', exchange=exchange, currency=currency)
        else: raise ValueError(f"Tipo seguridad no soportado: {sec_type}")
        return contract

    def _get_cache_filename(self, prefix, symbol, timeframe, start_date_str, end_date_str, sec_type, exchange, rth):
        tf_safe = timeframe.replace(' ','').replace(':','')
        rth_safe = 'RTH' if rth else 'Ext'
        filename = f"{prefix}_{symbol}_{exchange}_{tf_safe}_{start_date_str}_to_{end_date_str}_{sec_type}_{rth_safe}.parquet"
        return self.cache_dir / filename

    def _save_data_to_cache(self, df, filename):
        if df is None or df.empty: return
        try:
             if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None or str(df.index.tz).upper() != 'UTC':
                  logger.error(f"CRÍTICO: Intento de guardar en caché datos con índice no UTC! Archivo: {filename}.")
                  return 
             df.to_parquet(filename, index=True)
             logger.debug(f"Datos guardados en caché ({len(df)} filas): {filename}")
        except Exception as e: logger.error(f"Error guardando caché {filename}: {e}", exc_info=True)

    def _load_data_from_cache(self, filename: Path) -> pd.DataFrame | None:
        try:
            if filename.exists():
                df = pd.read_parquet(filename)
                if df.empty: return df 
                if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
                if df.index.tz is None: df.index = df.index.tz_localize('UTC')
                elif str(df.index.tz).upper() != 'UTC': df.index = df.index.tz_convert('UTC')
                return df
            return None
        except Exception as e: 
            logger.error(f"Error cargando o procesando caché {filename}: {e}", exc_info=True)
            return None 

    def get_levels_data(self, target_date: datetime.date, symbol: str, sec_type: str, exchange: str, currency: str, 
                        timeframe_daily='1 day', timeframe_intra='5 mins', 
                        premarket_start_hour=4, market_open_hour=9, market_open_minute=30, 
                        use_cache=True, **contract_kwargs):
        df_previous_day = pd.DataFrame()
        df_premarket = pd.DataFrame()
        
        prev_business_day = (pd.Timestamp(target_date) - pd.tseries.offsets.BDay(1)).date()
        prev_day_str = prev_business_day.strftime('%Y%m%d')
        cache_filename_pd = self._get_cache_filename("PD", symbol, timeframe_daily, prev_day_str, prev_day_str, sec_type, exchange, rth=True)
        
        if use_cache: df_previous_day = self._load_data_from_cache(cache_filename_pd)

        if df_previous_day is None or df_previous_day.empty:
            contract_obj = self._resolve_contract(symbol, sec_type, exchange, currency, **contract_kwargs)
            end_dt_prev_day_market = self.MARKET_TZ.localize(datetime.datetime.combine(prev_business_day, datetime.time(23, 59, 59)))
            end_dt_prev_day_utc_str = end_dt_prev_day_market.astimezone(self.UTC_TZ).strftime('%Y%m%d %H:%M:%S %Z')
            df_fetched_pd = self._fetch_data_core(contract_obj, end_dt_prev_day_utc_str, '1 D', timeframe_daily, True, 'TRADES')
            if df_fetched_pd is not None and not df_fetched_pd.empty:
                start_utc = self.MARKET_TZ.localize(datetime.datetime.combine(prev_business_day, datetime.time.min)).astimezone(self.UTC_TZ)
                end_utc = self.MARKET_TZ.localize(datetime.datetime.combine(prev_business_day, datetime.time.max)).astimezone(self.UTC_TZ)
                df_previous_day = df_fetched_pd[(df_fetched_pd.index >= start_utc) & (df_fetched_pd.index <= end_utc)]
                if use_cache and not df_previous_day.empty: self._save_data_to_cache(df_previous_day.copy(), cache_filename_pd)
        
        target_date_str = target_date.strftime('%Y%m%d')
        cache_filename_pm = self._get_cache_filename("PM", symbol, timeframe_intra, target_date_str, target_date_str, sec_type, exchange, rth=False)
        
        if use_cache: df_premarket = self._load_data_from_cache(cache_filename_pm)

        if df_premarket is None or df_premarket.empty:
            start_dt_pm_market = self.MARKET_TZ.localize(datetime.datetime.combine(target_date, datetime.time(premarket_start_hour, 0)))
            end_dt_pm_market = self.MARKET_TZ.localize(datetime.datetime.combine(target_date, datetime.time(market_open_hour, market_open_minute))) - datetime.timedelta(seconds=1)
            end_dt_pm_utc_req_str = end_dt_pm_market.astimezone(self.UTC_TZ).strftime('%Y%m%d %H:%M:%S %Z')
            duration_seconds_pm = int((end_dt_pm_market - start_dt_pm_market).total_seconds())
            duration_str_pm = f"{max(120, duration_seconds_pm + 120)} S"
            
            contract_obj = self._resolve_contract(symbol, sec_type, exchange, currency, **contract_kwargs)
            df_fetched_pm = self._fetch_data_core(contract_obj, end_dt_pm_utc_req_str, duration_str_pm, timeframe_intra, False, 'TRADES')
            if df_fetched_pm is not None and not df_fetched_pm.empty:
                start_utc = start_dt_pm_market.astimezone(self.UTC_TZ)
                end_utc = end_dt_pm_market.astimezone(self.UTC_TZ)
                df_premarket = df_fetched_pm[(df_fetched_pm.index >= start_utc) & (df_fetched_pm.index <= end_utc)]
                if use_cache and not df_premarket.empty: self._save_data_to_cache(df_premarket.copy(), cache_filename_pm)

        return (df_previous_day if df_previous_day is not None else pd.DataFrame(),
                df_premarket if df_premarket is not None else pd.DataFrame())

    def calculate_pdh_pdl(self, df_previous_day: pd.DataFrame | None) -> dict:
        if df_previous_day is None or df_previous_day.empty: return {'PDH': None, 'PDL': None}
        pdh = pd.to_numeric(df_previous_day['high'], errors='coerce').max()
        pdl = pd.to_numeric(df_previous_day['low'], errors='coerce').min()
        return {'PDH': pdh if pd.notna(pdh) else None, 'PDL': pdl if pd.notna(pdl) else None}

    def calculate_pmh_pml(self, df_premarket: pd.DataFrame | None) -> dict:
        if df_premarket is None or df_premarket.empty: return {'PMH': None, 'PML': None}
        pmh = pd.to_numeric(df_premarket['high'], errors='coerce').max()
        pml = pd.to_numeric(df_premarket['low'], errors='coerce').min()
        return {'PMH': pmh if pd.notna(pmh) else None, 'PML': pml if pd.notna(pml) else None}

    def get_main_data(self, symbol: str, timeframe: str, sec_type: str, exchange: str, currency: str, rth: bool, what_to_show: str,
                      download_start_date: datetime.date, download_end_date: datetime.date, 
                      use_cache=True, **kwargs) -> pd.DataFrame:
        start_date_str, end_date_str = download_start_date.strftime('%Y%m%d'), download_end_date.strftime('%Y%m%d')
        cache_filename = self._get_cache_filename(f"MAIN_{timeframe.replace(' ','')}", symbol, timeframe, start_date_str, end_date_str, sec_type, exchange, rth)
        
        if use_cache:
            df = self._load_data_from_cache(cache_filename)
            if df is not None and not df.empty:
                if df.index.min().date() <= download_start_date and df.index.max().date() >= download_end_date:
                    start_utc = self.UTC_TZ.localize(datetime.datetime.combine(download_start_date, datetime.time.min))
                    end_utc = self.UTC_TZ.localize(datetime.datetime.combine(download_end_date, datetime.time.max))
                    return df[(df.index >= start_utc) & (df.index <= end_utc)].copy()
        
        contract = self._resolve_contract(symbol, sec_type, exchange, currency, **kwargs)
        end_dt_market = self.MARKET_TZ.localize(datetime.datetime.combine(download_end_date, datetime.time(23, 59, 59)))
        end_dt_utc_req = end_dt_market.astimezone(self.UTC_TZ).strftime('%Y%m%d %H:%M:%S %Z')
        duration_req = calculate_ib_duration(download_start_date, download_end_date)
        
        df_fetched = self._fetch_data_core(contract, end_dt_utc_req, duration_req, timeframe, rth, what_to_show)
        
        if df_fetched is not None and not df_fetched.empty:
            start_utc = self.UTC_TZ.localize(datetime.datetime.combine(download_start_date, datetime.time.min))
            end_utc = self.UTC_TZ.localize(datetime.datetime.combine(download_end_date, datetime.time.max))
            df = df_fetched[(df_fetched.index >= start_utc) & (df_fetched.index <= end_utc)]
            if use_cache and not df.empty: self._save_data_to_cache(df.copy(), cache_filename)
            return df.copy()
            
        return pd.DataFrame()
