#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nifty_scalper_bot.config.env import build_broker_config
from nifty_scalper_bot.data.rest.client import ThrottledBrokerClient
from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.utils.rate_limiter import RateLimiter

load_dotenv()


def main() -> None:
    cfg = build_broker_config()
    inner = ZerodhaKiteClient(
        api_key=cfg.api_key,
        access_token=cfg.access_token,
        base_url=str(cfg.base_url),
    )
    client = ThrottledBrokerClient(inner=inner, limiter=RateLimiter())
    print(
        json.dumps(
            {
                "env_ok": True,
                "api_key": bool(cfg.api_key),
                "access_len": len(cfg.access_token),
            },
            indent=2,
        )
    )
    try:
        # If the client exposes a profile method; tolerate absence
        prof = getattr(client._inner, "get_profile", None)
        ok = bool(prof()) if callable(prof) else True
        print("profile_ok:", ok)
    except Exception as exc:  # noqa: BLE001 - intentional passthrough to CLI
        print("profile_error:", repr(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
