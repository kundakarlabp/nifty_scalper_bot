from __future__ import annotations

import logging


_log = logging.getLogger("structured").info


def emit(event: str, **fields) -> None:
    _log(event, extra={"extra": {"event": event, **fields}})
