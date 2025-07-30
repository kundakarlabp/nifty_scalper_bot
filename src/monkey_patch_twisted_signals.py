# src/monkey_patch_twisted_signals.py

import threading
import twisted.internet._signals

# Patch Twisted's install method to prevent signal errors
def safe_install_signal_handlers(self):
    if threading.current_thread() is threading.main_thread():
        self._original_install()

# Save original method
twisted.internet._signals._SignalReactorMixin._original_install = (
    twisted.internet._signals._SignalReactorMixin.install
)

# Replace with safe method
twisted.internet._signals._SignalReactorMixin.install = safe_install_signal_handlers
