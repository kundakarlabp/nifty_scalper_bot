"""Diagnostic tools and health checks for the scalper bot."""

from . import trace_ctl
from .trade import log_trade_context

__all__ = ["log_trade_context", "trace_ctl"]
