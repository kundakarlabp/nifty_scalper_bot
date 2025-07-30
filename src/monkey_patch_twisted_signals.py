# src/monkey_patch_twisted_signals.py
"""
Prevents Twisted from crashing in environments like Render or Docker
where signal handlers cannot be installed from non-main threads.
This patch overrides signal.signal when not on the main thread.
"""

import signal
import threading

# Only patch if we're not in the main thread
if threading.current_thread() is not threading.main_thread():
    def no_op_signal(*args, **kwargs):
        # Do nothing instead of setting a signal handler
        return None

    signal.signal = no_op_signal
