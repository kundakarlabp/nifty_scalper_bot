from __future__ import annotations  # pragma: no cover

"""Helpers for loading instrument metadata."""  # pragma: no cover

import os  # pragma: no cover
from typing import Any  # pragma: no cover

from src.broker.instruments import InstrumentStore  # pragma: no cover

INSTRUMENTS_CSV_ENV_NAMES: tuple[str, ...] = (
    "INSTRUMENTS__CSV",
    "INSTRUMENTS_CSV",
)


def _first_env(*names: str) -> str | None:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None


def load_instrument_store_from_settings(
    settings: Any, env_key: str = "INSTRUMENTS_CSV"
) -> InstrumentStore:  # pragma: no cover
    """Return an :class:`InstrumentStore` using CSV path from settings or env.

    Parameters
    ----------
    settings:
        Configuration object that may define ``INSTRUMENTS_CSV``.
    env_key:
        Environment variable name to consult if the settings attribute is absent.
    """
    path = getattr(settings, "INSTRUMENTS_CSV", None) or _first_env(
        *INSTRUMENTS_CSV_ENV_NAMES, env_key
    )
    if not path:
        raise FileNotFoundError("INSTRUMENTS_CSV path not provided")
    return InstrumentStore.from_csv(path)
