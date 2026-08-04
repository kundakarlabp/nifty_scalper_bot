from __future__ import annotations

from pathlib import Path
import subprocess
import sys


RUNNER_TEST = Path("tests/strategies/test_runner_symbol_role_gate.py")
MDM_TEST = Path("tests/data/test_mdm_tick_coalescing.py")
RUNNER_SOURCE = Path("src/nifty_scalper_bot/strategies/runner.py")
MDM_SOURCE = Path("src/nifty_scalper_bot/data/market_data_manager.py")

RUNNER_TEST_NAME = (
    "test_expiry_entry_policy_blocks_before_live_readiness_and_preparation"
)
MDM_TEST_NAME = "test_optional_context_option_is_not_recovery_critical"

RUNNER_TEST_CODE = r'''


def test_expiry_entry_policy_blocks_before_live_readiness_and_preparation(monkeypatch):
    runner_obj, strategy_manager, risk_manager, order_manager, _selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner_obj._trigger_candidate_symbols = {"NSE:NIFTY"}
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.expiry_theta_block",
        lambda: (True, "expiry_day_after_13:30_ist"),
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "expiry-entry-pregate",
            "source": "ws",
        },
    )

    strategy_manager.generate_signal.assert_called_once()
    risk_manager.validate.assert_not_called()
    order_manager.submit.assert_not_called()
    assert any(
        call.kwargs.get("stage") == "phase10_entry_policy"
        and call.kwargs.get("reason") == "expiry_day_after_13:30_ist"
        and call.kwargs.get("allowed") is False
        for call in runner_obj._emit_runner_eval_decision.call_args_list
    )
'''

MDM_TEST_CODE = r'''


def test_optional_context_option_is_not_recovery_critical():
    mdm = MarketDataManager(kite=None)
    mapping = _wire_symbols(mdm)
    selected_ce = "NFO:NIFTY26JUN24000CE"
    selected_pe = "NFO:NIFTY26JUN24000PE"
    optional_ce = "NFO:NIFTY26JUN25000CE"
    mdm.set_active_contract_basket(
        {
            "all_tokens": list(mapping),
            "token_by_symbol": {symbol: token for token, symbol in mapping.items()},
            "spot_symbol": "NSE:NIFTY",
            "futures_symbol": "NFO:NIFTY26JUNFUT",
            "selected_ce": selected_ce,
            "selected_pe": selected_pe,
            "option_symbols": [selected_ce, selected_pe, optional_ce],
        }
    )

    required = mdm._required_live_symbols()

    assert "NFO:NIFTY26JUNFUT" in required
    assert selected_ce in required
    assert selected_pe in required
    assert optional_ce not in required
'''

RUNNER_OLD = '''            if signal and signal.action != "HOLD":
                phase = "phase10_signal_execution"
                self._last_strategy_versions[symbol] = current_version
                signal_phase = self._data_phase.get(symbol)
                live_ready, live_ready_reason, live_ready_details = (
'''
RUNNER_NEW = '''            if signal and signal.action != "HOLD":
                phase = "phase10_signal_execution"
                self._last_strategy_versions[symbol] = current_version
                signal_phase = self._data_phase.get(symbol)
                entry_symbol = str(signal.symbol or symbol)
                if signal.action in {"BUY", "SELL"} and is_nifty_option_symbol(
                    entry_symbol
                ):
                    expiry_blocked, expiry_reason = expiry_theta_block()
                    if expiry_blocked:
                        self._emit_runner_eval_decision(
                            symbol=entry_symbol,
                            stage="phase10_entry_policy",
                            reason=expiry_reason,
                            allowed=False,
                            trace_id=trace_id,
                        )
                        return
                live_ready, live_ready_reason, live_ready_details = (
'''

MDM_OLD = '        add(self._basket_value(basket, "option_symbols", None))\n'
MDM_NEW = (
    "        # Nearby option strikes are context-only; the selected pair owns recovery.\n"
)


def run_pytest(node: str, *, expect_failure: bool = False) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", node],
        check=False,
    )
    if expect_failure:
        if completed.returncode != 1:
            raise SystemExit(
                f"expected one pre-fix test failure for {node}, "
                f"pytest returned {completed.returncode}"
            )
        return
    if completed.returncode != 0:
        raise SystemExit(f"focused validation failed for {node}")


def append_once(path: Path, marker: str, code: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        path.write_text(text.rstrip() + code + "\n", encoding="utf-8")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    append_once(RUNNER_TEST, RUNNER_TEST_NAME, RUNNER_TEST_CODE)
    append_once(MDM_TEST, MDM_TEST_NAME, MDM_TEST_CODE)

    run_pytest(
        f"{RUNNER_TEST}::{RUNNER_TEST_NAME}",
        expect_failure=True,
    )
    run_pytest(
        f"{MDM_TEST}::{MDM_TEST_NAME}",
        expect_failure=True,
    )

    replace_once(RUNNER_SOURCE, RUNNER_OLD, RUNNER_NEW)
    replace_once(MDM_SOURCE, MDM_OLD, MDM_NEW)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            str(RUNNER_SOURCE),
            str(MDM_SOURCE),
        ],
        check=True,
    )
    run_pytest(str(RUNNER_TEST))
    run_pytest(str(MDM_TEST))
    run_pytest("tests/risk/test_expiry_gate.py")
    run_pytest("tests/core/test_polling_failover.py")


if __name__ == "__main__":
    main()
