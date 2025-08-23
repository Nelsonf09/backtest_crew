# shared/fsm.py
"""
Módulo FSM (Finite State Machine) genérico y reutilizable para el proyecto.

Este archivo define una clase base 'FSM' que gestiona estados y transiciones,
junto con enumeraciones específicas ('State') que definen los posibles estados
para diferentes componentes de la aplicación (la app principal, la conexión de datos
y el ciclo de vida de una operación de trading).
"""
from enum import Enum, auto
import logging

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)

class State(Enum):
    """
    Clase base para todos los tipos de estado en las FSMs.
    Heredar de esta clase permite usar los enums como estados válidos.
    """
    def __str__(self):
        """Devuelve el nombre del estado como string para facilitar el logging."""
        return self.name

class FSM:
    """
    Clase base para una Máquina de Estados Finitos.
    Gestiona el estado actual y las transiciones de manera segura.
    """
    def __init__(self, initial_state: State):
        """
        Inicializa la FSM con un estado inicial.
        
        Args:
            initial_state (State): El estado con el que comenzará la FSM.
        
        Raises:
            TypeError: Si el estado inicial no es una instancia de la clase State.
        """
        if not isinstance(initial_state, State):
            raise TypeError("El estado inicial debe ser una instancia de la clase State Enum.")
        self._state = initial_state
        logger.debug(f"FSM '{self.__class__.__name__}' inicializada en estado: {self._state}")

    @property
    def state(self) -> State:
        """Devuelve el estado actual de la FSM de forma segura."""
        return self._state

    def transition_to(self, new_state: State):
        """
        Realiza la transición a un nuevo estado.

        Solo cambia el estado si el nuevo estado es diferente al actual,
        y registra la transición en los logs para facilitar la depuración.
        
        Args:
            new_state (State): El estado al que se desea transicionar.
        
        Raises:
            TypeError: Si el nuevo estado no es una instancia de la clase State.
        """
        if not isinstance(new_state, State):
            raise TypeError("El nuevo estado debe ser una instancia de la clase State Enum.")
        
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            logger.debug(f"Transición de FSM: {old_state} -> {new_state}")
        else:
            # Es útil saber si se intentó una transición redundante.
            logger.debug(f"Intento de transición al mismo estado: {self._state}. No se realizaron cambios.")

    def is_in_state(self, state: State) -> bool:
        """Comprueba si la FSM se encuentra en un estado específico."""
        return self._state == state

# --- Enumeraciones de Estado Específicas de la Aplicación ---

class AppState(State):
    """Define los estados del ciclo de vida de la aplicación principal de Streamlit."""
    CONFIGURING = auto()      # El usuario está ajustando parámetros en la sidebar.
    LOADING_DATA = auto()     # Se está descargando/cargando datos desde el DataManager.
    READY = auto()            # Datos cargados, listo para procesar el día de replay.
    REPLAYING = auto()        # El autoplay está activo y avanzando velas automáticamente.
    PAUSED = auto()           # El replay está en pausa, permitiendo el avance manual.
    FINISHED = auto()         # Se ha llegado al final de los datos del día o del rango total.
    ERROR = auto()            # Ha ocurrido un error crítico que detiene la app.

class ConnectionState(State):
    """Define los estados del ciclo de vida de la conexión con el broker (Interactive Brokers)."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTING = auto()
    ERROR = auto()

class TradeState(State):
    """Define los estados del ciclo de vida de una operación de trading individual."""
    FLAT = auto()             # Estado de la cuenta: no hay ninguna operación abierta.
    ACTIVE = auto()           # Estado de una operación: está abierta en el mercado.
    CLOSED = auto()           # Estado de una operación: ya ha sido cerrada.

