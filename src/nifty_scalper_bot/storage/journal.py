"""Durable key-value journal for persisting controller state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any, Dict


class AtomicKV:
    """Atomic JSON key-value store backed by the filesystem."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    def get(self, key: str, default: Any | None = None) -> Any | None:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._flush()

    def delete(self, key: str) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key, None)
                self._flush()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            content = self._path.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, dict):
                self._data = data
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            self._data = {}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        content = json.dumps(self._data, indent=2, sort_keys=True)
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self._path)


__all__ = ["AtomicKV"]
