"""Verification helpers for scheduled tasks and position sync integration."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


REQUIRED_FILES: Sequence[str] = (
    "src/nifty_scalper_bot/core/app.py",
    "src/nifty_scalper_bot/infra/scheduled_tasks.py",
    "src/nifty_scalper_bot/execution/position_manager.py",
)
OPTIONAL_FILES: Sequence[str] = ("src/nifty_scalper_bot/infra/log_rotation.py",)
MODULES: Sequence[str] = (
    "nifty_scalper_bot.infra.scheduled_tasks",
    "nifty_scalper_bot.execution.position_manager",
)


def check_files(paths: Sequence[str]) -> bool:
    """Confirm that each filesystem path in *paths* exists.

    Args:
        paths: Iterable of relative file paths to validate.

    Returns:
        True when all files are present, otherwise False.

    Raises:
        None.
    """

    all_present = True
    for relative in paths:
        target = Path(relative)
        if target.exists():
            print(f"✅ {relative}")
        else:
            print(f"❌ {relative} missing")
            all_present = False
    return all_present


def check_imports(modules: Sequence[str]) -> bool:
    """Attempt to import each dotted path in *modules*.

    Args:
        modules: Sequence of module paths to import.

    Returns:
        True when every import succeeds.

    Raises:
        None.
    """

    all_loaded = True
    for dotted in modules:
        try:
            importlib.import_module(dotted)
        except Exception as exc:  # noqa: BLE001 - verification utility
            print(f"❌ {dotted} import failed: {exc}")
            all_loaded = False
        else:
            print(f"✅ {dotted} import ok")
    return all_loaded


def main() -> int:
    """Run verification checks for integration artifacts.

    Args:
        None.

    Returns:
        Process exit code: 0 when checks pass, else 1.

    Raises:
        None.
    """

    print("=== Verifying required files ===")
    required_ok = check_files(REQUIRED_FILES)

    print("\n=== Verifying optional files ===")
    optional_ok = check_files(OPTIONAL_FILES)

    print("\n=== Verifying module imports ===")
    imports_ok = check_imports(MODULES)

    if required_ok and optional_ok and imports_ok:
        print("\n✅ All verifications passed")
        return 0
    print("\n❌ Verification incomplete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
