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
from .market_data_hardening import install_websocket_market_data_hardening
from nifty_scalper_bot.utils.runtime_session_guards import (
    install_websocket_market_calendar_guard,
)

# Direct and package imports both execute this module before exposing the
# transport class, so every runtime instance receives the canonical holiday
# gate without making the top-level package import side-effectful.
install_websocket_market_calendar_guard()
install_websocket_market_data_hardening(WebSocketManager)

__all__ = [
    "PollingStreamer",
    "StreamSupervisor",
    "StreamHealth",
    "WebSocketManager",
]
