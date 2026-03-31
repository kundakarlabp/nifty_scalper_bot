"""Runtime watchdog for polling/websocket liveness."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any


def _is_alive(obj: Any) -> bool:
    """Probe liveness from manager object. Args: obj. Returns: bool. Raises: None."""
    try:
        if obj is None:
            return False
        for attr in ('is_alive', 'alive', 'running'):
            fn = getattr(obj, attr, None)
            if callable(fn):
                return bool(fn())
        return True
    except Exception:
        return False


def start_watchdog(market_data_manager: Any):
    """Start watchdog thread. Args: market_data_manager. Returns: Thread. Raises: None."""
    logger = logging.getLogger('nifty_scalper_bot.watchdog')

    def _monitor() -> None:
        """Watch liveness and restart stale components. Args: none. Returns: None. Raises: None."""
        logger.info('✅ Watchdog Started')
        while True:
            time.sleep(1.0)
            try:
                now = time.time()
                scout_hb = float(getattr(market_data_manager, 'last_poll_heartbeat', 0.0) or 0.0)
                ws_hb = float(getattr(market_data_manager, 'last_ws_heartbeat', 0.0) or 0.0)

                if scout_hb > 0 and now - scout_hb > 5.0:
                    logger.warning('watchdog_polling_stale_restart')
                    restart = getattr(market_data_manager, 'restart_polling', None)
                    if callable(restart):
                        restart()

                if ws_hb > 0 and now - ws_hb > 5.0:
                    logger.warning('watchdog_websocket_stale_restart')
                    restart = getattr(market_data_manager, 'restart_websocket', None)
                    if callable(restart):
                        restart()

                if not _is_alive(market_data_manager):
                    logger.error('watchdog_manager_unhealthy')
            except Exception as exc:
                logger.error('Failure in watchdog monitor loop: %s', exc)

    thread = threading.Thread(target=_monitor, daemon=True, name='watchdog-thread')
    thread.start()
    return thread
