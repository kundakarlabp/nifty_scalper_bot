"""Notification interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Notifier(ABC):
    """Abstract interface for sending notifications to users."""

    @abstractmethod
    def send(self, text: str) -> None:
        """Send a plain text message."""
        raise NotImplementedError

    def send_json(self, payload: Dict[str, Any]) -> None:
        """Send a structured payload as a string representation."""
        self.send(str(payload))
