"""Minimal uvicorn entrypoint for the FastAPI health endpoints."""

from __future__ import annotations

import os

import uvicorn

from nifty_scalper_bot.core.app import NiftyScalperApp


def main() -> None:
    """Launch the health and metrics server bound to ``$PORT``."""

    port = int(os.environ.get("PORT", "8000"))
    app = NiftyScalperApp()
    uvicorn.run(app.health_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
