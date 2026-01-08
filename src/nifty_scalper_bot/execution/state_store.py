import threading
from typing import Dict
from .fsm import FSMState


class StateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._states: Dict[str, FSMState] = {}

    def get(self, key: str) -> FSMState:
        with self._lock:
            return self._states.get(key, FSMState.IDLE)

    def set(self, key: str, state: FSMState) -> None:
        with self._lock:
            self._states[key] = state

    def reset(self, key: str) -> None:
        with self._lock:
            self._states.pop(key, None)
