# src/monkey_patch_twisted_signals.py
"""
Blocks signal registration globally if not running in the main thread.
Must be imported before anything else, especially before importing Twisted.
"""

import signal
import threading
import builtins

if threading.current_thread() is not threading.main_thread():
    def patched_signal(sig, handler):
        # Completely skip signal registration outside main thread
        return None

    # Apply early monkey patch
    signal.signal = patched_signal
    builtins.__original_signal__ = patched_signal  # Optional: backup
