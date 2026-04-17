"""Streaming utilities for resilient market data delivery.

Architecture:
    - WebSocketManager: primary real-time tick source (Zerodha KiteTicker).
    - PollingStreamer: REST fallback when WebSocket is unavailable.
    - StreamSupervisor: lifecycle/autostart/health manager for the active streamer.

All streamers feed MarketDataManager, which re-emits into DataHub (SSOT).
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
