#!/usr/bin/env python3
# apply_mdm_patch.py — idempotent patcher for market_data_manager.py

from pathlib import Path
import sys, os

ROOT = Path("src/nifty_scalper_bot/data")
TARGET = ROOT / "market_data_manager.py"

if not TARGET.exists():
    print("ERROR: cannot find file:", TARGET)
    sys.exit(1)

text = TARGET.read_text(encoding="utf-8")

bak = TARGET.with_suffix(".py.bak")
if not bak.exists():
    bak.write_text(text, encoding="utf-8")
    print(f"Backup written to {bak}")

# 1) Insert resolver warm block right after `self._logger = get_logger(__name__)`
marker = "        self._logger = get_logger(__name__)\n"
warm_block = """\n        # --- InstrumentResolver warm (non-blocking by default) -----------------\n        try:\n            if self._resolver is not None and hasattr(self._resolver, \"warm\"):\n                # Opt into blocking warm via env, else warm in background.\n                blocking = os.getenv(\"MDM_RESOLVER_WARM_BLOCKING\", \"\").strip().lower() in {\n                    \"1\", \"true\", \"yes\", \"on\"\n                }\n                timeout_s = 5.0\n                try:\n                    timeout_s = float(os.getenv(\"MDM_RESOLVER_WARM_TIMEOUT_SEC\", \"5.0\"))\n                except Exception:\n                    timeout_s = 5.0\n\n                def _do_warm() -> None:\n                    try:\n                        self._logger.info(\"InstrumentResolver: warm() starting\")\n                        self._resolver.warm()\n                        self._logger.info(\"InstrumentResolver: warm() completed\")\n                    except Exception as exc:  # pragma: no cover - resolver warm may fail in some envs\n                        self._logger.warning(\n                            \"InstrumentResolver warm() failed: %s\",\n                            exc,\n                            extra={\"event\": \"resolver_warm_error\"},\n                        )\n\n                if blocking:\n                    thread = threading.Thread(target=_do_warm, name=\"mdm-resolver-warm\", daemon=True)\n                    thread.start()\n                    thread.join(timeout=timeout_s)\n                    if thread.is_alive():\n                        self._logger.warning(\n                            \"InstrumentResolver warm() timed out after %.2fs\",\n                            timeout_s,\n                            extra={\"event\": \"resolver_warm_timeout\"},\n                        )\n                else:\n                    thread = threading.Thread(target=_do_warm, name=\"mdm-resolver-warm\", daemon=True)\n                    thread.start()\n            else:\n                if self._resolver is not None:\n                    self._logger.debug(\"InstrumentResolver provided but has no warm() method\")\n        except Exception as exc:\n            self._logger.error(\n                \"Failed to start InstrumentResolver warm task: %s\",\n                exc,\n                extra={\"event\": \"resolver_warm_start_error\"},\n            )\n        # ---------------------------------------------------------------------\n"""
if marker in text and "InstrumentResolver: warm() starting" not in text:
    text = text.replace(marker, marker + warm_block)
    print("Inserted resolver warm block.")
else:
    print("Resolver warm block already present or marker not found; skipping insertion.")

# 2) Replace the option_contracts try/except with retry version
old_try = (
    "        try:\n"
    "            option_contracts = resolver.option_contracts(  # type: ignore[attr-defined]\n"
    "                normalized_underlying,\n"
    "                force_refresh=force_refresh,\n"
    "            )\n"
    "        except AttributeError:\n"
    "            return None\n"
    "        except Exception as exc:  # noqa: BLE001\n"
    "            self._logger.warning(\n"
    "                \"option_chain_metadata_failed\", extra={\"error\": str(exc)}\n"
    "            )\n"
    "            return None\n"
)

new_try = (
    "        option_contracts = None\n"
    "        try:\n"
    "            option_contracts = resolver.option_contracts(  # type: ignore[attr-defined]\n"
    "                normalized_underlying,\n"
    "                force_refresh=force_refresh,\n"
    "            )\n"
    "        except AttributeError:\n"
    "            # Resolver doesn't implement option_contracts()\n"
    "            return None\n"
    "        except Exception as exc:  # noqa: BLE001\n"
    "            self._logger.warning(\n"
    "                \"option_chain_metadata_failed\", extra={\"error\": str(exc)}\n"
    "            )\n"
    "            return None\n\n"
    "        # If the resolver returned nothing, attempt a one-shot warm() retry and re-fetch.\n"
    "        if not option_contracts:\n"
    "            try:\n"
    "                if hasattr(resolver, \"warm\"):\n"
    "                    self._logger.info(\n"
    "                        \"Option chain empty — attempting resolver.warm() then retry\",\n"
    "                        extra={\"underlying\": normalized_underlying},\n"
    "                    )\n"
    "                    with suppress(Exception):\n"
    "                        resolver.warm()\n"
    "                    # retry once\n"
    "                    option_contracts = resolver.option_contracts(\n"
    "                        normalized_underlying, force_refresh=force_refresh\n"
    "                    )\n"
    "                    if option_contracts:\n"
    "                        self._logger.info(\n"
    "                            \"Resolver retry returned contracts\",\n"
    "                            extra={\"underlying\": normalized_underlying, \"count\": len(option_contracts)},\n"
    "                        )\n"
    "            except Exception as exc:  # defensive\n"
    "                self._logger.debug(\n"
    "                    \"Resolver retry failed\", exc, extra={\"event\": \"resolver_retry_failed\"}\n"
    "                )\n"
)

if old_try in text and "Resolver retry returned contracts" not in text:
    text = text.replace(old_try, new_try)
    print("Replaced option_contracts try/except with retry block.")
else:
    print("Option_contracts block not found or already patched; skipping replacement.")

# 3) Insert missing-tokens debug in _fetch_option_quotes
search_line = "        missing = [token for token in tokens if token not in results]\n"
insert_line = (
    "        if missing:\n"
    "            self._logger.debug(\n"
    "                \"Option quote fetch missing tokens\",\n"
    "                extra={\"missing_count\": len(missing), \"missing_tokens\": missing},\n"
    "            )\n"
)
if search_line in text and "Option quote fetch missing tokens" not in text:
    text = text.replace(search_line, search_line + insert_line)
    print("Inserted missing tokens debug log.")
else:
    print("Missing tokens debug log already present or anchor not found; skipping.")

# Write back
TARGET.write_text(text, encoding="utf-8")
print("Patched file written to", TARGET)
