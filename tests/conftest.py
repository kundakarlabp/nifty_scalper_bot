"""Global pytest fixtures and environment configuration."""

from __future__ import annotations

import os


# Disable Telegram by default to avoid configuration validation errors during
# tests that import modules eagerly loading settings. Provide benign defaults
# for the bot token and chat ID so validation succeeds even if the environment
# is inspected before individual tests patch it.
os.environ.setdefault("TELEGRAM__ENABLED", "false")
os.environ.setdefault("TELEGRAM__BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM__CHAT_ID", "12345")

