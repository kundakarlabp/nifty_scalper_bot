"""Verification utility for the Telegram bot integration."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple


def _check_env_var(name: str, required: bool = True) -> Tuple[bool, str]:
    """Return a tuple describing whether an environment variable is set."""

    value = os.getenv(name)
    if value:
        if any(key in name for key in ("TOKEN", "SECRET", "KEY")):
            masked = value[:10] + "..." if len(value) > 10 else "***"
            return True, masked
        return True, value
    if required:
        return False, "MISSING"
    return True, "not set (optional)"


def _print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _check_python_version() -> bool:
    version = sys.version_info
    ok = version >= (3, 10)
    status = "✅" if ok else "❌"
    print(f"{status} Python {version.major}.{version.minor}.{version.micro}")
    if not ok:
        print("   Requires Python 3.10 or newer")
    return ok


def _check_modules(modules: Iterable[Tuple[str, str]]) -> list[bool]:
    results: list[bool] = []
    for module, package in modules:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {package}: {version}")
            results.append(True)
        except ImportError:
            print(f"❌ {package}: NOT INSTALLED")
            print(f"   Install with: pip install {package}")
            results.append(False)
    return results


def _check_required_files(files: Iterable[str]) -> list[bool]:
    results: list[bool] = []
    for file_path in files:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        results.append(exists)
    return results


def main() -> None:
    """Run verification checks for the Telegram bot integration."""

    print("=" * 80)
    print("TELEGRAM BOT INSTALLATION VERIFICATION")
    print("=" * 80)

    checks: list[bool] = []

    _print_header("1. Python Version")
    checks.append(_check_python_version())

    _print_header("2. Dependencies")
    checks.extend(
        _check_modules(
            [
                ("pydantic", "pydantic"),
                ("telegram", "python-telegram-bot"),
                ("requests", "requests"),
            ]
        )
    )

    _print_header("3. Environment Variables")
    required_vars = [
        ("TELEGRAM__BOT_TOKEN", True),
        ("TELEGRAM__CHAT_ID", True),
        ("BROKER_API_KEY", True),
        ("BROKER_API_SECRET", True),
    ]
    optional_vars = [
        ("TELEGRAM_BOT_TOKEN", False),
        ("TELEGRAM_CHAT_ID", False),
        ("ALLOW_LIVE_ORDERS", False),
    ]
    for var, required in required_vars:
        ok, value = _check_env_var(var, required)
        status = "✅" if ok else "❌"
        print(f"{status} {var}: {value}")
        checks.append(ok)
    print("\nOptional:")
    for var, required in optional_vars:
        ok, value = _check_env_var(var, required)
        status = "✅" if ok else "❌"
        print(f"   {status} {var}: {value}")

    _print_header("4. Chat ID Configuration")
    chat_id_raw = os.getenv("TELEGRAM__CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if chat_id_raw:
        try:
            chat_id = int(chat_id_raw.strip())
        except ValueError:
            print(f"❌ Invalid TELEGRAM__CHAT_ID: {chat_id_raw}")
            checks.append(False)
        else:
            print(f"✅ Configured chat ID: {chat_id}")
            checks.append(True)
    else:
        print("❌ TELEGRAM__CHAT_ID not configured")
        checks.append(False)

    _print_header("5. Project Structure")
    checks.extend(
        _check_required_files(
            [
                "src/nifty_scalper_bot/__init__.py",
                "src/nifty_scalper_bot/app.py",
                "src/nifty_scalper_bot/notifications/telegram_controller.py",
                "src/nifty_scalper_bot/config/base.py",
                "src/nifty_scalper_bot/config/env.py",
            ]
        )
    )

    _print_header("6. Module Imports")
    for module in [
        "nifty_scalper_bot.config.base",
        "nifty_scalper_bot.config.env",
        "nifty_scalper_bot.utils.rate_limiter",
        "nifty_scalper_bot.data.rest.client",
        "nifty_scalper_bot.notifications.telegram_controller",
    ]:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
            checks.append(True)
        except ImportError as exc:
            print(f"❌ {module}: {exc}")
            checks.append(False)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  {module}: {exc.__class__.__name__}: {exc}")
            checks.append(False)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for check in checks if check)
    total = len(checks)
    percentage = (passed / total * 100) if total else 0.0
    print(f"Checks passed: {passed}/{total} ({percentage:.1f}%)")
    if passed == total:
        print("\n✅ ALL CHECKS PASSED! You're ready to start the bot.")
        print("\nRun: python -m nifty_scalper_bot.app")
        exit_code = 0
    elif passed >= total * 0.8:
        print("\n⚠️  MOSTLY READY. Fix the failed checks above.")
        exit_code = 1
    else:
        print("\n❌ SETUP INCOMPLETE. Please fix the issues above.")
        exit_code = 1
    print("\nFor detailed setup instructions, see: TELEGRAM_BOT_SETUP.md")
    print("=" * 80)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
