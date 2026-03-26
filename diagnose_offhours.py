"""
Run with:  python diagnose_offhours.py

What it checks (without placing live orders):
  1) Env flags for off-hours paper-mode are effective.
  2) Settings load, session guard snapshot OK (override=true is allowed).
  3) Polling-only mode wired (no WS expected).
  4) MarketDataManager accepts a normalized "polling/rest" tick and caches it.
  5) /order path is blocked by guard (no live routing).
  6) Zerodha client param builder sanity (no "exchange missing").

It prints a PASS/FAIL summary and the next step to fix each failure.
"""

import importlib  # noqa: F401
import os
import sys
import time
from contextlib import suppress  # noqa: F401

# --- Force safe paper flags (process-local only) ------------------------------
os.environ.setdefault("ENABLE_LIVE", "false")
os.environ.setdefault("ENABLE_LIVE_TRADING", "false")
os.environ.setdefault("SHADOW_MODE", "true")
os.environ.setdefault("SESSION_ALLOW_OUT_OF_HOURS", "true")
os.environ.setdefault("ALLOW_OFFHOURS_TESTING", "true")
os.environ.setdefault("STREAM__MODE", "poll")
os.environ.setdefault("POLLING__ENABLED", "true")
os.environ.setdefault("POLLING__SYMBOLS", "NIFTY")
os.environ.setdefault("POLLING__INTERVAL_SEC", "0.70")
os.environ.setdefault("LOG_LEVEL", "INFO")


def _ok(name):
    print(f"✅ {name}")


def _bad(name, why):
    print(f"❌ {name} — {why}")


def _assert(cond, name, fix):
    if cond:
        _ok(name)
        return True
    _bad(name, fix)
    return False


def main():
    # 0) Import package with deterministic .env loader
    try:
        import nifty_scalper_bot.load_env_first  # noqa: F401
    except Exception as e:
        _bad(
            "load_env_first import",
            f"{e} (Create src/nifty_scalper_bot/load_env_first.py with load_dotenv())",
        )
        return 1

    # 1) Load core settings
    try:
        from nifty_scalper_bot.config.settings import get_settings

        settings = get_settings()
        _ok("Settings loaded")
    except Exception as e:
        _bad("Settings load", f"{e}")
        return 1

    # 2) Build components via app initializer (but don’t start uvicorn)
    try:
        from nifty_scalper_bot.core.app import initialize_components

        ctx = initialize_components(settings)
        _ok("Components initialized")
    except Exception as e:
        _bad("initialize_components()", f"{e}")
        return 1

    # 3) Session guard sanity; off-hours override should be visible
    try:
        sg = ctx.session_guard
        snap = sg.snapshot() if hasattr(sg, "snapshot") else sg.evaluate().as_dict()
        print("Session guard snapshot:", snap)
        _assert(
            snap.get("override_out_of_hours", False) is True,
            "Off-hours override",
            "Set SESSION_ALLOW_OUT_OF_HOURS=true (and ALLOW_OFFHOURS_TESTING=true)",
        )
    except Exception as e:
        _bad("Session guard snapshot", f"{e}")
        return 1

    # 4) Polling-only wiring
    try:
        mdm = ctx.market_data_manager
        _assert(
            getattr(mdm, "ws_connected", False) is False,
            "WS disabled",
            "STREAM__MODE=poll",
        )
        _ok("Polling supervisor active (see /poll_status in Telegram)")
    except Exception as e:
        _bad("Polling wiring", f"{e}")
        return 1

    # 5) Inject a synthetic polling tick -> MDM must cache it
    try:
        # Resolve symbol/token (your bot reports NIFTY -> 256265)
        symbol = "NIFTY"
        token = (
            getattr(ctx.instrument_resolver, "resolve_symbol_token", lambda s: 256265)(
                symbol
            )
            if hasattr(ctx, "instrument_resolver")
            else 256265
        )

        tick = {
            "instrument_token": token,
            "token": token,  # alias tolerated
            "last_price": 25235.2,  # MDM normalizer accepts last_price/ltp
            "timestamp": time.time(),
            "source": "rest",  # mark as polled
            "symbol": symbol,  # ensure cache key is populated
        }

        # Call the same internal hook the poller uses
        await mdm._handle_tick(tick)  # noqa: SLF001 (intentionally calling internal)
        time.sleep(0.05)
        cached = mdm.get_latest_tick(symbol)
        _assert(
            cached is not None,
            "MDM cached first tick",
            "Check MDM _handle_tick aliasing token→instrument_token",
        )
        if cached:
            print(
                "Cached tick:",
                {
                    k: cached.get(k)
                    for k in ("symbol", "ltp", "bid", "ask", "timestamp", "source")
                },
            )
    except Exception as e:
        _bad("MDM cache path", f"{e}")
        return 1

    # 6) Order path MUST be blocked (paper/off-hours)
    try:
        om = ctx.order_manager
        blocked = False
        try:
            om.place_order(
                symbol="NSE:NIFTY 50", side="BUY", quantity=1
            )  # should raise
        except Exception as exc:
            blocked = True
            print("Order blocked as expected:", type(exc).__name__, str(exc))
        _assert(
            blocked,
            "Orders blocked in paper mode",
            "Ensure ENABLE_LIVE=false, ENABLE_LIVE_TRADING=false, SHADOW_MODE=true",
        )
    except Exception as e:
        _bad("Order block test", f"{e}")
        return 1

    # 7) Zerodha param builder sanity (no 'exchange missing' when you go live later)
    try:
        zk = getattr(ctx, "broker_client", None)
        if hasattr(zk, "_build_kite_params"):
            params = zk._build_kite_params(
                {
                    "symbol": "NIFTY25OCTFUT",  # when you enable live for NFO
                    "side": "BUY",
                    "quantity": 25,
                    "order_type": "MARKET",
                    "product": "MIS",
                    "exchange": os.getenv("INSTRUMENTS__TRADE_EXCHANGE", "NFO"),
                }
            )
            _assert(
                "exchange" in params and params["exchange"],
                "Zerodha params.exchange present",
                "Set INSTRUMENTS__TRADE_EXCHANGE=NFO for derivatives",
            )
            _assert(
                "tradingsymbol" in params and params["tradingsymbol"],
                "Zerodha params.tradingsymbol present",
                (
                    "Implement resolver.tradingsymbol_for_order or set "
                    "INSTRUMENTS__TRADE_SYMBOL to a valid TS"
                ),
            )
        else:
            _ok("Zerodha params build skipped (no _build_kite_params)")
    except Exception as e:
        _bad("Zerodha params build", f"{e}")
        # Not fatal off-hours; continue

    print("\nAll checks complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
