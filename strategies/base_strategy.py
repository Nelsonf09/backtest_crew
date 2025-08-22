# core/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
import logging

# Logger específico
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """ Clase base abstracta para estrategias de trading. """
    def __init__(self, **params):
        self._params = params
        # Usar logger.debug para la inicialización
        logger.debug(f"Inicializando estrategia {self.__class__.__name__} con parámetros: {params}")

    @property
    def params(self):
        return self._params

    @abstractmethod
    def get_signal(self, data: pd.DataFrame, current_day_levels: dict | None = None) -> str | dict: # MODIFICADO: Añadido current_day_levels
        """ 
        Genera la señal de trading. Debe ser implementado por clases hijas.
        
        Args:
            data (pd.DataFrame): DataFrame con los datos históricos OHLCV hasta la vela actual inclusive.
                                 Se espera que el índice sea un DatetimeIndex.
            current_day_levels (dict | None, optional): Diccionario con los niveles clave calculados para el día actual
                                                        (ej. {'PDH': 150.0, 'ORL': 148.5}). Defaults to None.
        
        Returns:
            str | dict: Puede ser 'HOLD', 'BUY', 'SELL', o un diccionario con detalles adicionales
                        (ej. {'type': 'BUY', 'sl_price': ..., 'tp1_price': ...}).
        """
        pass