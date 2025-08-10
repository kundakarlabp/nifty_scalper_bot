# tests/conftest.py
import logging
import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def _quiet_logs():
    logging.getLogger().setLevel(logging.WARNING)
    # Ensure logs directory exists to avoid IO errors when writing trade logs
    os.makedirs("logs", exist_ok=True)
