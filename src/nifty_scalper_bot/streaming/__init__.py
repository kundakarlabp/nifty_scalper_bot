"""Streaming utilities for resilient market data delivery.

Architecture:
    - WebSocketManager: primary real-time tick source (Zerodha KiteTicker).
    - PollingStreamer: REST fallback when WebSocket is unavailable.
    - StreamSupervisor: lifecycle/autostart/health manager for the active streamer.

All streamers feed MarketDataManager, which re-emits into DataHub (SSOT).

Hardening note: the WebSocket calendar guard and market-data hardening are
installed explicitly in ``websocket_manager.py`` at the class definition
site — this package import has no side effects.
"""

from .polling_streamer import PollingStreamer
from .stream_supervisor import StreamHealth, StreamSupervisor
from .websocket_manager import WebSocketManager

__all__ = [
    "PollingStreamer",
    "StreamSupervisor",
    "StreamHealth",
    "WebSocketManager",
]
