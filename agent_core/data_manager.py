# core/data_manager.py
import logging
from pathlib import Path
import pandas as pd
import datetime
from ib_insync import IB, util, Contract, Forex, Stock, Future, Index 
import config 
import pytz
import math

logger = logging.getLogger(__name__)

def calculate_ib_duration(start_date: datetime.date, end_date: datetime.date) -> str:
    """
    Calcula la cadena de duración para IB basada en un rango de fechas.
    Se asegura de que la duración sea suficiente para cubrir el rango.
    """
    if start_date > end_date:
        logger.warning(f"calculate_ib_duration: start_date {start_date} es posterior a end_date {end_date}. Devolviendo '1 D'.")
        return "1 D" 

    delta_days = (end_date - start_date).days + 1

    if delta_days <= 0: 
        return "1 D"
    elif delta_days <= 30: 
        return f"{delta_days} D"
    elif delta_days <= 365 * 2: 
         weeks = math.ceil(delta_days / 7)
         return f"{weeks} W"
    else: 
         years = math.ceil(delta_days / 365.25)
         return f"{years} Y"

class DataManager:
    """ Gestiona conexión IB, datos históricos, caché y niveles clave. """
    def __init__(self):
        self.ib = IB()
        self.config = config
        self.cache_dir = getattr(config, 'CACHE_DIR', Path('cache').resolve()) 
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.MARKET_TZ = pytz.timezone(getattr(config, 'MARKET_TIMEZONE', 'America/New_York'))
        self.UTC_TZ = pytz.utc
        logger.debug(f"DataManager init: Cache={self.cache_dir}, MarketTZ={self.MARKET_TZ}")

    def _handle_ib_error(self, reqId, errorCode, errorString, contract):
        ignored_codes = {
            162, 200, 2100, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 
            2110, 2119, 2137, 2150, 2158, 300, 366,
        }
        if errorCode not in ignored_codes:
             logger.error(f"IB Error: Req={reqId}, Code={errorCode}, Msg={errorString}, Contract={contract if contract else 'N/A'}")

    def connect_ib(self):
        if self.ib.isConnected():
            logger.debug("Ya conectado a IB.")
            return True
        try:
            host = getattr(self.config, 'IB_HOST', '127.0.0.1')
            port = getattr(self.config, 'IB_PORT', 7497) 
            client_id = getattr(self.config, 'IB_CLIENT_ID', 1)
            logger.debug(f"Conectando a IB {host}:{port} ID {client_id}...")
            
            try:
                self.ib.errorEvent -= self._handle_ib_error
            except Exception: 
                pass 
            self.ib.errorEvent += self._handle_ib_error

            self.ib.connect(host, port, clientId=client_id, timeout=20) 
            if self.ib.isConnected():
                server_time = self.ib.reqCurrentTime()
                logger.debug(f"Conexión IB OK. ServerTime: {server_time}")
                return True
            else:
                logger.error("Fallo conexión IB después del intento (isConnected() es False).")
                return False
        except ConnectionRefusedError:
             logger.error(f"Error conexión IB: Conexión rechazada. ¿Está TWS/Gateway ejecutándose y aceptando conexiones API en {getattr(self.config, 'IB_HOST', '127.0.0.1')}:{getattr(self.config, 'IB_PORT', 7497)}?")
             return False
        except AttributeError as ae: 
             logger.error(f"Error de Atributo en conexión IB (¿problema librería o versión ib_insync?): {ae}", exc_info=True)
             return False
        except Exception as e: 
            logger.error(f"Error genérico conexión IB: {e}", exc_info=True)
            return False

    def disconnect_ib(self):
        if self.ib.isConnected():
            logger.debug("Desconectando de IB...")
            self.ib.disconnect()
            logger.debug("Desconexión IB OK.")
        else:
            logger.debug("Intento de desconexión, pero no estaba conectado a IB.")

    def _resolve_contract(self, symbol, sec_type, exchange, currency, primary_exchange=None, lastTradeDateOrContractMonth=''):
        sec_type=sec_type.upper()
        logger.debug(f"Resolviendo contrato para: Symbol={symbol}, SecType={sec_type}, Exch={exchange}, Curr={currency}, PrimExch={primary_exchange}, LastTradeDate={lastTradeDateOrContractMonth}")
        contract = None
        if sec_type == 'FOREX': 
            contract=Forex(f'{symbol}') 
        elif sec_type == 'STK':
            contract=Stock(symbol, exchange, currency)
            if primary_exchange: 
                contract.primaryExchange = primary_exchange
            elif exchange == 'SMART': 
                 default_primary_exchange_stk = getattr(config, 'DEFAULT_PRIMARY_EXCHANGE', 'ISLAND') 
                 logger.warning(f"Usando exchange SMART para STK sin primaryExchange explícito. Usando por defecto: {default_primary_exchange_stk}")
                 contract.primaryExchange = default_primary_exchange_stk
        elif sec_type == 'FUT': 
            contract=Future(symbol, lastTradeDateOrContractMonth=lastTradeDateOrContractMonth, exchange=exchange, currency=currency)
            if not lastTradeDateOrContractMonth:
                 logger.warning("Para Futuros (FUT), 'lastTradeDateOrContractMonth' es usualmente requerido y no fue provisto.")
        elif sec_type == 'IND': 
            contract=Index(symbol, exchange, currency)
        elif sec_type == 'CFD': 
            contract=Contract(symbol=symbol, secType='CFD', exchange=exchange, currency=currency)
        elif sec_type == 'CRYPTO':
              if not exchange or exchange == 'SMART':
                   logger.error("Para CRYPTO se requiere un Exchange específico (ej: PAXOS, GEMINI), no SMART.")
                   raise ValueError("Exchange específico requerido para CRYPTO")
              contract = Contract(symbol=symbol, secType='CRYPTO', exchange=exchange, currency=currency)
        else:
              logger.error(f"Tipo de seguridad no soportado: {sec_type}")
              raise ValueError(f"Tipo seguridad no soportado: {sec_type}")

        if contract: 
            logger.debug(f"Contrato resuelto: {contract}")
        return contract

    def _get_cache_filename(self, prefix, symbol, timeframe, start_date_str, end_date_str, sec_type, exchange, rth):
        tf_safe = timeframe.replace(' ','').replace(':','')
        rth_safe = 'RTH' if rth else 'Ext'
        filename = f"{prefix}_{symbol}_{exchange}_{tf_safe}_{start_date_str}_to_{end_date_str}_{sec_type}_{rth_safe}.parquet"
        return self.cache_dir / filename

    def _save_data_to_cache(self, df, filename):
        if df is None or df.empty:
            logger.debug(f"No se guarda caché vacío: {filename}")
            return
        try:
             if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None or str(df.index.tz).upper() != 'UTC':
                  logger.error(f"CRÍTICO: Intento de guardar en caché datos con índice no UTC! Archivo: {filename}. Índice TZ: {df.index.tz}. Verifique la lógica de fetch.")
                  return 
             df.to_parquet(filename, index=True)
             logger.debug(f"Datos guardados en caché ({len(df)} filas): {filename}")
        except Exception as e:
             logger.error(f"Error guardando caché {filename}: {e}", exc_info=True)

    def _load_data_from_cache(self, filename: Path) -> pd.DataFrame | None:
        """Carga el DataFrame desde la caché y ASEGURA índice UTC."""
        try:
            if filename.exists():
                df = pd.read_parquet(filename)
                logger.debug(f"Datos cargados desde caché: {filename}. Shape: {df.shape}")
                if df.empty:
                    logger.warning(f"Archivo de caché {filename} está vacío.")
                    return df 

                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"Cache {filename}: Índice no es DatetimeIndex ({type(df.index)}). Intentando convertir.")
                    try:
                        df.index = pd.to_datetime(df.index)
                        logger.info(f"Cache {filename}: Índice convertido a DatetimeIndex.")
                    except Exception as e_conv:
                        logger.error(f"Fallo conversión índice cache a DatetimeIndex: {e_conv}. Invalidando caché.", exc_info=True)
                        return None 
                
                if df.index.tz is None:
                    logger.warning(f"Cache {filename}: Índice DatetimeIndex SIN timezone. Localizando a UTC.")
                    try:
                        df.index = df.index.tz_localize('UTC', ambiguous='infer') 
                    except Exception as e_loc:
                        logger.error(f"Error localizando índice cache a UTC: {e_loc}. Invalidando caché.", exc_info=True)
                        return None 
                elif str(df.index.tz).upper() != 'UTC': 
                    logger.warning(f"Cache {filename}: Índice con TZ {df.index.tz}. Convirtiendo a UTC.")
                    try:
                        df.index = df.index.tz_convert('UTC')
                    except Exception as e_conv_utc:
                        logger.error(f"Error convirtiendo índice cache a UTC: {e_conv_utc}. Invalidando caché.", exc_info=True)
                        return None 
                
                logger.debug(f"Cache {filename}: Índice asegurado en UTC.")
                return df
            else: 
                logger.debug(f"Archivo de caché no encontrado: {filename}")
                return None # Esta es la línea 162 (aproximadamente)
        except Exception as e: 
            logger.error(f"Error cargando o procesando caché {filename}: {e}", exc_info=True)
            return None 

    def _fetch_data_core(self, contract: Contract, end_datetime_str: str, duration_str: str, timeframe: str, rth: bool, what_to_show: str) -> pd.DataFrame:
        if not self.ib.isConnected():
            logger.error("No conectado a IB para fetch de datos.")
            return pd.DataFrame() 

        contract_symbol_for_log = contract.symbol if contract else "ContratoDesconocido"
        logger.info(f"Solicitando datos IB: Cont={contract_symbol_for_log}, End='{end_datetime_str}', Dur='{duration_str}', TF='{timeframe}', RTH={rth}, Show='{what_to_show}'")

        try:
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime=end_datetime_str, 
                durationStr=duration_str,     
                barSizeSetting=timeframe,     
                whatToShow=what_to_show,      
                useRTH=rth,                   
                formatDate=2,                 
                keepUpToDate=False,           
                timeout=60.0                  
            )

            if not bars:
                logger.warning(f"No se recibieron barras de IB para {contract_symbol_for_log} (End={end_datetime_str}, Dur={duration_str}, TF={timeframe}, RTH={rth}).")
                return pd.DataFrame() 

            df = util.df(bars) 

            if df is None or df.empty:
                logger.warning(f"DataFrame vacío después de util.df para {contract_symbol_for_log}.")
                return pd.DataFrame()

            if 'date' in df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date']) 
                        logger.debug(f"Columna 'date' convertida explícitamente a datetime para {contract_symbol_for_log} antes de set_index.")
                except Exception as e_conv_date_col:
                     logger.error(f"Fallo al convertir la columna 'date' a datetime para {contract_symbol_for_log} antes de set_index: {e_conv_date_col}")
                     return pd.DataFrame() 
                df = df.set_index('date') 
            else:
                 logger.error(f"No se encontró columna 'date' en los datos de IB para {contract_symbol_for_log}.")
                 return pd.DataFrame()

            if not isinstance(df.index, pd.DatetimeIndex):
                 logger.error(f"El índice no es DatetimeIndex después de set_index para {contract_symbol_for_log}. Tipo: {type(df.index)}")
                 try: 
                     df.index = pd.to_datetime(df.index, errors='raise') 
                     logger.info(f"Índice convertido forzosamente a DatetimeIndex para {contract_symbol_for_log}.")
                 except Exception as e_conv_idx_force:
                     logger.error(f"Fallo conversión forzada de índice a DatetimeIndex para {contract_symbol_for_log}: {e_conv_idx_force}", exc_info=True)
                     return pd.DataFrame()
            
            if df.index.tz is None:
                 logger.debug(f"IB Fetch {contract_symbol_for_log}: Índice sin timezone. Localizando a UTC (esperado con formatDate=2).")
                 try: 
                     df.index = df.index.tz_localize('UTC', ambiguous='infer')
                 except Exception as e_loc_utc: 
                     logger.error(f"Fallo localizando índice fetch a UTC para {contract_symbol_for_log}: {e_loc_utc}", exc_info=True)
                     return pd.DataFrame()
            elif str(df.index.tz).upper() != 'UTC': 
                 logger.warning(f"IB Fetch {contract_symbol_for_log}: Índice con TZ {df.index.tz} inesperado (esperado UTC con formatDate=2). Convirtiendo a UTC.")
                 try: 
                     df.index = df.index.tz_convert('UTC')
                 except Exception as e_conv_to_utc: 
                     logger.error(f"Fallo convirtiendo índice fetch a UTC para {contract_symbol_for_log}: {e_conv_to_utc}", exc_info=True)
                     return pd.DataFrame()

            df = df.rename(columns={'open':'open','high':'high','low':'low','close':'close','volume':'volume'})
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Faltan columnas OHLC requeridas en datos IB para {contract_symbol_for_log}. Columnas: {list(df.columns)}")
                return pd.DataFrame()

            if 'volume' not in df.columns:
                logger.debug(f"Columna 'volume' no encontrada para {contract_symbol_for_log}, añadiendo con ceros.")
                df['volume'] = 0.0 
            
            for col in required_cols + ['volume']:
                if col in df.columns: 
                    df[col] = pd.to_numeric(df[col], errors='coerce') 

            initial_rows = len(df)
            df = df.dropna(subset=required_cols) 
            if len(df) < initial_rows:
                 logger.warning(f"Se eliminaron {initial_rows - len(df)} filas con NaN en OHLC para {contract_symbol_for_log}.")

            if df.empty:
                 logger.warning(f"DataFrame vacío para {contract_symbol_for_log} después de procesar y eliminar NaNs.")
                 return pd.DataFrame()

            logger.info(f"Datos IB recibidos y procesados (UTC) para {contract_symbol_for_log}. {len(df)} filas.")
            return df

        except ConnectionError as ce: 
             logger.error(f"Error de Conexión IB durante fetch para {contract_symbol_for_log}: {ce}", exc_info=True)
             return pd.DataFrame()
        except ValueError as ve: 
            logger.error(f"Error de Valor durante fetch para {contract_symbol_for_log}: {ve}", exc_info=True)
            return pd.DataFrame()
        except Exception as e: 
            logger.error(f"Error inesperado en _fetch_data_core para {contract_symbol_for_log}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_levels_data(self, target_date: datetime.date, symbol: str, sec_type: str, exchange: str, currency: str, 
                        timeframe_daily='1 day', timeframe_intra='5 mins', 
                        premarket_start_hour=4, market_open_hour=9, market_open_minute=30, 
                        use_cache=True, **contract_kwargs):
        logger.debug(f"Obteniendo datos niveles para {symbol} en fecha {target_date.strftime('%Y-%m-%d')}")
        df_previous_day = pd.DataFrame()
        df_premarket = pd.DataFrame()
        contract_obj_levels = None 

        try:
            prev_business_day = (pd.Timestamp(target_date) - pd.tseries.offsets.BDay(1)).date()
            logger.debug(f"Día hábil anterior para PDH/L: {prev_business_day.strftime('%Y-%m-%d')}")
            prev_day_str = prev_business_day.strftime('%Y%m%d')
            
            cache_filename_pd = self._get_cache_filename(
                "PD", symbol, timeframe_daily, prev_day_str, prev_day_str, 
                sec_type, exchange, rth=True 
            )

            if use_cache:
                df_previous_day = self._load_data_from_cache(cache_filename_pd)

            if df_previous_day is None or df_previous_day.empty:
                logger.debug(f"Datos día anterior ({prev_day_str}) no en caché o vacíos. Descargando...")
                contract_obj_levels = self._resolve_contract(symbol, sec_type, exchange, currency, **contract_kwargs)
                
                end_dt_prev_day_market = self.MARKET_TZ.localize(datetime.datetime.combine(prev_business_day, datetime.time(23, 59, 59)))
                end_dt_prev_day_utc_str = end_dt_prev_day_market.astimezone(self.UTC_TZ).strftime('%Y%m%d %H:%M:%S %Z')
                
                df_fetched_pd = self._fetch_data_core(
                    contract_obj_levels, 
                    end_datetime_str=end_dt_prev_day_utc_str, 
                    duration_str='1 D', 
                    timeframe=timeframe_daily, 
                    rth=True, 
                    what_to_show='TRADES'
                )

                if df_fetched_pd is not None and not df_fetched_pd.empty:
                    prev_bday_start_utc = self.MARKET_TZ.localize(datetime.datetime.combine(prev_business_day, datetime.time.min)).astimezone(self.UTC_TZ)
                    prev_bday_end_utc = self.MARKET_TZ.localize(datetime.datetime.combine(prev_business_day, datetime.time.max)).astimezone(self.UTC_TZ)
                    
                    df_previous_day = df_fetched_pd[
                        (df_fetched_pd.index >= prev_bday_start_utc) & 
                        (df_fetched_pd.index <= prev_bday_end_utc)
                    ]
                    if use_cache and df_previous_day is not None and not df_previous_day.empty:
                        self._save_data_to_cache(df_previous_day.copy(), cache_filename_pd) 
                else:
                    df_previous_day = pd.DataFrame()
            
            if df_previous_day.empty:
                 logger.warning(f"No se obtuvieron datos para el día anterior {prev_business_day.strftime('%Y-%m-%d')} para PDH/PDL.")

        except Exception as e_pd: 
            logger.error(f"Error obteniendo datos/calculando PDH/PDL para {symbol} día anterior a {target_date.strftime('%Y-%m-%d')}: {e_pd}", exc_info=True)
            df_previous_day = pd.DataFrame()

        target_date_str = target_date.strftime('%Y%m%d')
        start_dt_pm_market = None
        end_dt_pm_market = None
        try:
            start_dt_pm_market = self.MARKET_TZ.localize(datetime.datetime.combine(target_date, datetime.time(premarket_start_hour, 0, 0)))
            end_dt_pm_market = self.MARKET_TZ.localize(datetime.datetime.combine(target_date, datetime.time(market_open_hour, market_open_minute, 0))) - datetime.timedelta(seconds=1)
        except pytz.exceptions.InvalidTimeError as e_tz_pm: 
             logger.warning(f"Hora de inicio/fin de premercado inválida para {target_date_str} en {self.MARKET_TZ} (¿cambio horario? Error: {e_tz_pm}). Saltando cálculo PM.")
        except Exception as e_pm_time_create:
            logger.error(f"Error inesperado creando tiempos de premercado para {target_date_str}: {e_pm_time_create}", exc_info=True)

        if start_dt_pm_market and end_dt_pm_market and start_dt_pm_market < end_dt_pm_market : 
            cache_filename_pm = self._get_cache_filename(
                "PM", symbol, timeframe_intra, target_date_str, target_date_str, 
                sec_type, exchange, rth=False 
            )
            if use_cache:
                df_premarket = self._load_data_from_cache(cache_filename_pm)

            if df_premarket is None or df_premarket.empty: 
                logger.debug(f"Datos premercado ({target_date_str}) no en caché o vacíos. Descargando...")
                try:
                    if contract_obj_levels is None: 
                         contract_obj_levels = self._resolve_contract(symbol, sec_type, exchange, currency, **contract_kwargs)
                    
                    end_dt_pm_utc_req_str = end_dt_pm_market.astimezone(self.UTC_TZ).strftime('%Y%m%d %H:%M:%S %Z')
                    duration_seconds_pm = int((end_dt_pm_market - start_dt_pm_market).total_seconds())
                    if duration_seconds_pm <= 0:
                        logger.warning(f"Duración de premercado calculada no positiva ({duration_seconds_pm}s). Saltando descarga PM.")
                    else:
                        duration_str_pm = f"{max(120, duration_seconds_pm + 120)} S" 
                        
                        df_fetched_pm = self._fetch_data_core(
                            contract_obj_levels, 
                            end_datetime_str=end_dt_pm_utc_req_str, 
                            duration_str=duration_str_pm, 
                            timeframe=timeframe_intra, 
                            rth=False, 
                            what_to_show='TRADES'
                        )

                        if df_fetched_pm is not None and not df_fetched_pm.empty:
                            start_dt_pm_utc_filter = start_dt_pm_market.astimezone(self.UTC_TZ)
                            end_dt_pm_utc_filter = end_dt_pm_market.astimezone(self.UTC_TZ)
                            df_premarket = df_fetched_pm[
                                (df_fetched_pm.index >= start_dt_pm_utc_filter) & 
                                (df_fetched_pm.index <= end_dt_pm_utc_filter)
                            ]
                            if use_cache and df_premarket is not None and not df_premarket.empty:
                                self._save_data_to_cache(df_premarket.copy(), cache_filename_pm) 
                        else:
                            df_premarket = pd.DataFrame()
                except Exception as e_pm_fetch: 
                    logger.error(f"Error obteniendo datos premercado {symbol} ({target_date_str}): {e_pm_fetch}", exc_info=True)
                    df_premarket = pd.DataFrame()
            
            if df_premarket.empty:
                logger.warning(f"No se obtuvieron datos para premercado de {target_date_str}.")

        len_pd = len(df_previous_day) if df_previous_day is not None else 0
        len_pm = len(df_premarket) if df_premarket is not None else 0
        logger.debug(f"Datos para niveles: Día Anterior ({prev_business_day.strftime('%Y-%m-%d') if 'prev_business_day' in locals() else 'N/A'}): {len_pd} filas. Premercado ({target_date_str}): {len_pm} filas.")

        return df_previous_day if df_previous_day is not None else pd.DataFrame(), \
               df_premarket if df_premarket is not None else pd.DataFrame()

    def calculate_pdh_pdl(self, df_previous_day: pd.DataFrame | None) -> dict:
        if df_previous_day is None or df_previous_day.empty:
            logger.debug("No hay datos del día anterior para calcular PDH/PDL.")
            return {'PDH': None, 'PDL': None}
        try:
            highs = pd.to_numeric(df_previous_day['high'], errors='coerce').dropna()
            lows = pd.to_numeric(df_previous_day['low'], errors='coerce').dropna()
            if highs.empty or lows.empty:
                 logger.warning("Columnas high/low vacías o no numéricas en datos día anterior para PDH/PDL.")
                 return {'PDH': None, 'PDL': None}
            pdh = highs.max()
            pdl = lows.min()
            logger.info(f"PDH={pdh if pd.notna(pdh) else 'N/A'}, PDL={pdl if pd.notna(pdl) else 'N/A'} (Calculado de {len(df_previous_day)} barras)")
            return {'PDH': pdh if pd.notna(pdh) else None, 'PDL': pdl if pd.notna(pdl) else None}
        except Exception as e:
            logger.error(f"Error calculando PDH/PDL: {e}", exc_info=True)
            return {'PDH': None, 'PDL': None}

    def calculate_pmh_pml(self, df_premarket: pd.DataFrame | None) -> dict:
        if df_premarket is None or df_premarket.empty:
            logger.debug("No hay datos de premercado para calcular PMH/PML.")
            return {'PMH': None, 'PML': None}
        try:
            highs = pd.to_numeric(df_premarket['high'], errors='coerce').dropna()
            lows = pd.to_numeric(df_premarket['low'], errors='coerce').dropna()
            if highs.empty or lows.empty:
                 logger.warning("Columnas high/low vacías o no numéricas en datos premercado para PMH/PML.")
                 return {'PMH': None, 'PML': None}
            pmh = highs.max()
            pml = lows.min()
            logger.info(f"PMH={pmh if pd.notna(pmh) else 'N/A'}, PML={pml if pd.notna(pml) else 'N/A'} (Calculado de {len(df_premarket)} barras)")
            return {'PMH': pmh if pd.notna(pmh) else None, 'PML': pml if pd.notna(pml) else None}
        except Exception as e:
            logger.error(f"Error calculando PMH/PML: {e}", exc_info=True)
            return {'PMH': None, 'PML': None}

    def get_main_data(self, symbol: str, timeframe: str, sec_type: str, exchange: str, currency: str, rth: bool, what_to_show: str,
                      download_start_date: datetime.date, download_end_date: datetime.date, 
                      use_cache=True, **kwargs) -> pd.DataFrame:
        contract_kwargs = {k:v for k,v in kwargs.items() if k in ['primary_exchange', 'lastTradeDateOrContractMonth']}
        cache_prefix = f"MAIN_{timeframe.replace(' ','')}"
        start_date_str = download_start_date.strftime('%Y%m%d')
        end_date_str = download_end_date.strftime('%Y%m%d')

        cache_filename = self._get_cache_filename(cache_prefix, symbol, timeframe, start_date_str, end_date_str, sec_type, exchange, rth)

        df = None 
        if use_cache:
            df = self._load_data_from_cache(cache_filename)
            if df is not None:
                 if not df.empty:
                      try:
                          data_start_date_utc = df.index.min().date()
                          data_end_date_utc = df.index.max().date()

                          if data_start_date_utc <= download_start_date and data_end_date_utc >= download_end_date:
                              logger.info(f"Cache HIT válido encontrado para {symbol} ({timeframe}) [{start_date_str} a {end_date_str}]")
                              start_dt_utc_req_filter = self.UTC_TZ.localize(datetime.datetime.combine(download_start_date, datetime.time.min))
                              end_dt_utc_req_filter = self.UTC_TZ.localize(datetime.datetime.combine(download_end_date, datetime.time.max))
                              
                              df_filtered = df[(df.index >= start_dt_utc_req_filter) & (df.index <= end_dt_utc_req_filter)]
                              logger.debug(f"Retornando {len(df_filtered)} filas desde caché para el rango solicitado.")
                              return df_filtered.copy() if df_filtered is not None else pd.DataFrame()
                          else:
                               logger.info(f"Cache HIT PARCIAL/INVÁLIDO para {symbol} {start_date_str}-{end_date_str} (Rango caché: {data_start_date_utc} a {data_end_date_utc} vs Req: {download_start_date} a {download_end_date}). Descargando de nuevo...")
                               df = None 
                      except Exception as e_cache_check:
                           logger.warning(f"Error validando rango de caché para {symbol}: {e_cache_check}. Forzando descarga.", exc_info=True)
                           df = None 
                 else: 
                      logger.debug(f"Cache {cache_filename} está vacío. Descargando...")
                      df = None 

        if df is None: 
            logger.info(f"Descargando datos para {symbol} ({timeframe}) de {start_date_str} a {end_date_str}...")
            try:
                contract = self._resolve_contract(symbol, sec_type, exchange, currency, **contract_kwargs)
                
                end_time_for_request = datetime.time(23, 59, 59)
                try:
                    end_datetime_market_tz = self.MARKET_TZ.localize(datetime.datetime.combine(download_end_date, end_time_for_request))
                except pytz.exceptions.AmbiguousTimeError: 
                    end_datetime_market_tz = self.MARKET_TZ.localize(datetime.datetime.combine(download_end_date, end_time_for_request), is_dst=True)
                    logger.warning(f"Tiempo ambiguo para end_datetime_market_tz en {download_end_date}, usando is_dst=True.")
                except pytz.exceptions.InvalidTimeError: 
                    end_datetime_market_tz = self.MARKET_TZ.localize(datetime.datetime.combine(download_end_date, datetime.time(23,0,0)))
                    logger.warning(f"Tiempo inválido para end_datetime_market_tz, usando 23:00:00 market time.")

                end_datetime_utc_for_req = end_datetime_market_tz.astimezone(self.UTC_TZ)
                end_datetime_req_str = end_datetime_utc_for_req.strftime('%Y%m%d %H:%M:%S %Z') 
                
                duration_req_str = calculate_ib_duration(download_start_date, download_end_date)
                logger.debug(f"Parámetros IB calculados: endDateTime='{end_datetime_req_str}', durationStr='{duration_req_str}'")
                
                df_fetched = self._fetch_data_core(contract, end_datetime_req_str, duration_req_str, timeframe, rth, what_to_show)
                
                if df_fetched is not None and not df_fetched.empty:
                    logger.debug(f"Datos descargados ({len(df_fetched)} filas) antes de filtrar rango exacto.")
                    start_dt_utc_filter_exact = self.UTC_TZ.localize(datetime.datetime.combine(download_start_date, datetime.time.min))
                    end_dt_utc_filter_exact = self.UTC_TZ.localize(datetime.datetime.combine(download_end_date, datetime.time.max))

                    df = df_fetched[(df_fetched.index >= start_dt_utc_filter_exact) & (df_fetched.index <= end_dt_utc_filter_exact)]
                    
                    if df.empty:
                         logger.warning(f"Datos para {symbol} ({start_date_str}-{end_date_str}) quedaron vacíos DESPUÉS de filtrar el rango exacto. Datos descargados originalmente: {len(df_fetched)} filas.")
                    elif use_cache:
                         logger.debug(f"Guardando {len(df)} filas filtradas en caché: {cache_filename}")
                         self._save_data_to_cache(df.copy(), cache_filename) 
                else:
                     logger.error(f"Fallo total al obtener datos de IB para {symbol} ({start_date_str}-{end_date_str}). _fetch_data_core devolvió vacío.");
                     df = pd.DataFrame() 

            except ValueError as ve: 
                logger.error(f"Error obteniendo datos principales para {symbol}: {ve}", exc_info=True)
                df = pd.DataFrame()
            except Exception as e: 
                logger.error(f"Error general obteniendo datos principales para {symbol}: {e}", exc_info=True)
                df = pd.DataFrame()
        
        return df.copy() if df is not None else pd.DataFrame()
