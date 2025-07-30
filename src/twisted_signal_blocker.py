# src/twisted_signal_blocker.py
"""
Instructs Twisted to skip installing signal handlers.
MUST be imported before twisted.internet.reactor is used.
"""

import os

# Prevent Twisted from trying to install signal handlers
os.environ["TWISTED_REACTOR_SIGNAL_HANDLERS"] = "0"
