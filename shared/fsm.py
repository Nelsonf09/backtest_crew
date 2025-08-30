# shared/fsm.py
"""
Módulo FSM (Finite State Machine) genérico y reutilizable para el proyecto.
"""
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

class State(Enum):
    def __str__(self):
        return self.name

class FSM:
    def __init__(self, initial_state: State):
        if not isinstance(initial_state, State):
            raise TypeError("El estado inicial debe ser una instancia de la clase State Enum.")
        self._state = initial_state
        logger.debug(f"FSM '{self.__class__.__name__}' inicializada en estado: {self._state}")

    @property
    def state(self) -> State:
        return self._state

    def transition_to(self, new_state: State):
        if not isinstance(new_state, State):
            raise TypeError("El nuevo estado debe ser una instancia de la clase State Enum.")
        
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            logger.info(f"Transición de FSM de App: {old_state} -> {new_state}")
        else:
            logger.debug(f"Intento de transición al mismo estado: {self._state}. No se realizaron cambios.")

    def is_in_state(self, state: State) -> bool:
        return self._state == state

class AppState(State):
    """Define los estados del ciclo de vida de la aplicación principal de Streamlit."""
    CONFIGURING = auto()
    LOADING_DATA = auto()
    READY = auto()
    REPLAYING = auto()
    PAUSED = auto()
    FINISHED = auto()
    ERROR = auto()
    GLOBAL_BACKTESTING = auto()
    SHOWING_RESULTS = auto()
    # --- NUEVO ESTADO AÑADIDO ---
    SHOWING_COMPARISON = auto()

class ConnectionState(State):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTING = auto()
    ERROR = auto()

class TradeState(State):
    FLAT = auto()
    ACTIVE = auto()
    CLOSED = auto()