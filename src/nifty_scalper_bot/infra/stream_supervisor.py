"""Backwards compatibility wrapper for :mod:`nifty_scalper_bot.streaming`."""

from nifty_scalper_bot.streaming.stream_supervisor import (
    StreamHealth,
    StreamSupervisor,
)

__all__ = ["StreamSupervisor", "StreamHealth"]
