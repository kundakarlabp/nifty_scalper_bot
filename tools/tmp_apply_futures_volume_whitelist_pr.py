from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]
VOLUME_TEST = ROOT / "tests/strategies/test_active_futures_resolution.py"
WHITELIST_TEST = ROOT / "tests/strategies/test_runner_active_basket_pending.py"
SIGNAL_GENERATOR = ROOT / "src/nifty_scalper_bot/strategies/signal_generator.py"
RUNNER = ROOT / "src/nifty_scalper_bot/strategies/runner.py"
WORKFLOW = ROOT / ".github/workflows/tmp-futures-volume-whitelist-pr.yml"
THIS_FILE = Path(__file__).resolve()


def run(*args: str, expect: int = 0) -> None:
    completed = subprocess.run(args, cwd=ROOT, check=False)
    if completed.returncode != expect:
        raise SystemExit(
            f"command returned {completed.returncode}, expected {expect}: {' '.join(args)}"
        )


def append_once(path: Path, marker: str, payload: str) -> None:
    text = path.read_text()
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n\n" + textwrap.dedent(payload).lstrip())


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one replacement in {path}, found {count}")
    path.write_text(text.replace(old, new, 1))


replace_once(
    VOLUME_TEST,
    "from typing import Any\n",
    "from datetime import date\nfrom typing import Any\n",
)

append_once(
    VOLUME_TEST,
    "test_futures_volume_ignores_trade_quantity_and_keeps_baseline",
    r'''
    class _VolumeHub:
        def __init__(
            self,
            quotes: list[dict[str, float]],
            active: str = "NFO:NIFTY26JUNFUT",
        ) -> None:
            self._quotes = list(quotes)
            self._active = active

        def get_active_futures_symbol(self):
            return self._active

        def get_quote(self, symbol: str, allow_pull: bool = False):
            del allow_pull
            if symbol == "NSE:NIFTY":
                return {"ltp": 24000.0, "vwap": 24000.0}
            if symbol == self._active and self._quotes:
                return self._quotes.pop(0)
            return {}


    def test_futures_volume_ignores_trade_quantity_and_keeps_baseline() -> None:
        symbol = "NFO:NIFTY26JUNFUT"
        hub = _VolumeHub(
            [
                {"volume_traded_today": 1000.0},
                {"last_quantity": 10.0},
                {"volume_traded_today": 1050.0},
            ],
            active=symbol,
        )
        manager = _manager(futures_symbol=symbol, hub=hub)

        first: dict[str, Any] = {}
        manager._augment_futures_metrics(first)
        assert manager._last_futures_volume == 1000.0
        assert list(manager._futures_volume_history) == []
        assert first["futures_volume_source"] == "volume_traded_today"
        assert first["futures_volume_trusted"] is True

        trade_quantity_only: dict[str, Any] = {}
        manager._augment_futures_metrics(trade_quantity_only)
        assert manager._last_futures_volume == 1000.0
        assert list(manager._futures_volume_history) == []
        assert trade_quantity_only["futures_volume"] is None
        assert trade_quantity_only["futures_volume_trusted"] is False

        next_cumulative: dict[str, Any] = {}
        manager._augment_futures_metrics(next_cumulative)
        assert manager._last_futures_volume == 1050.0
        assert list(manager._futures_volume_history) == [50.0]
        assert next_cumulative["futures_volume_ratio"] == 1.0


    def test_futures_volume_rejects_same_session_decrease_without_false_spike() -> None:
        symbol = "NFO:NIFTY26JUNFUT"
        hub = _VolumeHub(
            [
                {"volume_traded": 1000.0},
                {"volume_traded": 1100.0},
                {"volume_traded": 900.0},
                {"volume_traded": 1150.0},
            ],
            active=symbol,
        )
        manager = _manager(futures_symbol=symbol, hub=hub)

        manager._augment_futures_metrics({})
        manager._augment_futures_metrics({})

        decreased: dict[str, Any] = {}
        manager._augment_futures_metrics(decreased)
        assert manager._last_futures_volume == 1100.0
        assert list(manager._futures_volume_history) == [100.0]
        assert decreased["futures_volume"] is None
        assert decreased["futures_volume_trusted"] is False

        recovered: dict[str, Any] = {}
        manager._augment_futures_metrics(recovered)
        assert manager._last_futures_volume == 1150.0
        assert list(manager._futures_volume_history) == [100.0, 50.0]
        assert recovered["futures_volume_ratio"] == 50.0 / 75.0


    def test_futures_volume_resets_baseline_when_contract_changes() -> None:
        symbol = "NFO:NIFTY26JULFUT"
        hub = _VolumeHub([{"volume_traded_today": 125.0}], active=symbol)
        manager = _manager(futures_symbol=symbol, hub=hub)
        manager._last_futures_volume = 9000.0
        manager._last_futures_volume_symbol = "NFO:NIFTY26JUNFUT"
        manager._last_futures_volume_date = date.today()
        manager._futures_volume_history.append(75.0)

        indicators: dict[str, Any] = {}
        manager._augment_futures_metrics(indicators)

        assert manager._last_futures_volume == 125.0
        assert manager._last_futures_volume_symbol == symbol
        assert list(manager._futures_volume_history) == []
        assert indicators["futures_volume_ratio"] is None
        assert indicators["futures_volume_trusted"] is True
    ''',
)

append_once(
    WHITELIST_TEST,
    "test_pending_promotion_refreshes_option_evaluation_whitelist",
    r'''
    def test_pending_promotion_refreshes_option_evaluation_whitelist() -> None:
        ce = "NFO:NIFTY26MAY23300CE"
        pe = "NFO:NIFTY26MAY23300PE"
        old_context = "NFO:NIFTY26MAY23250CE"
        runner = _runner({ce: 30, pe: 30})
        runner._pending_selected_ce = ce
        runner._pending_selected_pe = pe
        runner._pending_atm_strike = 23300
        runner._active_option_symbols = {old_context}
        runner._eval_option_whitelist = {"STALE"}
        calls = []

        def compute(option_symbols, atm_strike, selected_ce, selected_pe):
            calls.append((set(option_symbols), atm_strike, selected_ce, selected_pe))
            return {selected_ce, selected_pe}

        runner._compute_eval_option_whitelist = compute

        assert runner._maybe_promote_pending_active_basket(source="test") is True
        assert calls == [({old_context, ce, pe}, 23300, ce, pe)]
        assert runner._eval_option_whitelist == {ce, pe}
    ''',
)

run(
    sys.executable,
    "-m",
    "pytest",
    "-q",
    str(VOLUME_TEST.relative_to(ROOT)),
    "-k",
    "test_futures_volume_ignores_trade_quantity_and_keeps_baseline",
    expect=1,
)
run(
    sys.executable,
    "-m",
    "pytest",
    "-q",
    str(WHITELIST_TEST.relative_to(ROOT)),
    "-k",
    "test_pending_promotion_refreshes_option_evaluation_whitelist",
    expect=1,
)

replace_once(
    SIGNAL_GENERATOR,
    '''        self._futures_symbol = self._canonical_futures_symbol(futures_symbol)
        self._futures_volume_history: Deque[float] = deque(maxlen=120)
        self._last_index_ltp: float | None = None
''',
    '''        self._futures_symbol = self._canonical_futures_symbol(futures_symbol)
        self._futures_volume_history: Deque[float] = deque(maxlen=120)
        self._last_futures_volume: float | None = None
        self._last_futures_volume_date: dt.date | None = None
        self._last_futures_volume_symbol: str | None = None
        self._last_index_ltp: float | None = None
''',
)

replace_once(
    SIGNAL_GENERATOR,
    '''        indicators.setdefault("futures_volume", None)
        indicators.setdefault("futures_volume_avg", None)
        indicators.setdefault("futures_volume_ratio", None)
        indicators.setdefault("nifty_index_ltp", None)  # ✅ NEW
''',
    '''        indicators.setdefault("futures_volume", None)
        indicators.setdefault("futures_volume_avg", None)
        indicators.setdefault("futures_volume_ratio", None)
        indicators.setdefault("futures_volume_source", None)
        indicators.setdefault("futures_volume_trusted", False)
        indicators.setdefault("nifty_index_ltp", None)  # ✅ NEW
''',
)

replace_once(
    SIGNAL_GENERATOR,
    '''        if quote:
            volume = self._extract_float(
                quote,
                ("volume_traded_today", "volume_traded", "volume", "last_quantity"),
            )
            if volume is not None:
                last_volume = getattr(self, "_last_futures_volume", None)
                if last_volume is None:
                    volume_delta = volume
                elif volume < last_volume:
                    volume_delta = volume
                    self._logger.info(
                        "Condition met: futures_volume_reset",
                        extra={
                            "event": "futures_volume_reset",
                            "symbol": self._futures_symbol,
                            "last_volume": last_volume,
                            "current_volume": volume,
                        },
                    )
                else:
                    volume_delta = volume - last_volume

                self._last_futures_volume = volume
                if volume_delta >= 0:
                    self._futures_volume_history.append(volume_delta)

                indicators["futures_volume"] = volume
                if self._futures_volume_history:
                    avg = sum(self._futures_volume_history) / len(
                        self._futures_volume_history
                    )
                    indicators["futures_volume_avg"] = avg
                    if avg > 0:
                        indicators["futures_volume_ratio"] = volume_delta / avg
                        self._logger.info(
                            "Condition met: futures_volume_ratio_updated",
                            extra={
                                "event": "futures_volume_ratio_updated",
                                "symbol": self._futures_symbol,
                                "volume_delta": volume_delta,
                                "volume_avg": avg,
                                "volume_ratio": indicators["futures_volume_ratio"],
                            },
                        )
''',
    '''        if quote:
            volume = None
            volume_source = None
            for key in ("volume_traded_today", "volume_traded"):
                candidate = self._extract_float(quote, (key,))
                if candidate is not None and math.isfinite(candidate) and candidate >= 0:
                    volume = candidate
                    volume_source = key
                    break

            if volume is not None:
                current_date = dt.datetime.now().date()
                same_stream = (
                    self._last_futures_volume_symbol == self._futures_symbol
                    and self._last_futures_volume_date == current_date
                )
                volume_delta: float | None = None
                trusted = True

                if not same_stream or self._last_futures_volume is None:
                    self._futures_volume_history.clear()
                    self._last_futures_volume = volume
                    self._last_futures_volume_symbol = self._futures_symbol
                    self._last_futures_volume_date = current_date
                elif volume >= self._last_futures_volume:
                    volume_delta = volume - self._last_futures_volume
                    self._last_futures_volume = volume
                else:
                    trusted = False
                    self._logger.warning(
                        "FUTURES_VOLUME_DECREASE_REJECTED symbol=%s last=%s current=%s",
                        self._futures_symbol,
                        self._last_futures_volume,
                        volume,
                        extra={
                            "event": "FUTURES_VOLUME_DECREASE_REJECTED",
                            "symbol": self._futures_symbol,
                            "last_volume": self._last_futures_volume,
                            "current_volume": volume,
                            "volume_source": volume_source,
                        },
                    )

                indicators["futures_volume_source"] = volume_source
                indicators["futures_volume_trusted"] = trusted
                if trusted:
                    indicators["futures_volume"] = volume
                if volume_delta is not None:
                    self._futures_volume_history.append(volume_delta)

                if volume_delta is not None and self._futures_volume_history:
                    avg = sum(self._futures_volume_history) / len(
                        self._futures_volume_history
                    )
                    indicators["futures_volume_avg"] = avg
                    if avg > 0:
                        indicators["futures_volume_ratio"] = volume_delta / avg
                        self._logger.info(
                            "Condition met: futures_volume_ratio_updated",
                            extra={
                                "event": "futures_volume_ratio_updated",
                                "symbol": self._futures_symbol,
                                "volume_delta": volume_delta,
                                "volume_avg": avg,
                                "volume_ratio": indicators["futures_volume_ratio"],
                                "volume_source": volume_source,
                            },
                        )
''',
)

replace_once(
    RUNNER,
    '''        active.update({pending_ce, pending_pe})
        self._active_option_symbols = active
        self._pending_selected_ce = None
''',
    '''        active.update({pending_ce, pending_pe})
        self._active_option_symbols = active
        self._eval_option_whitelist = self._compute_eval_option_whitelist(
            active,
            self._active_atm_strike,
            pending_ce,
            pending_pe,
        )
        self._pending_selected_ce = None
''',
)

run(
    sys.executable,
    "-m",
    "pytest",
    "-q",
    str(VOLUME_TEST.relative_to(ROOT)),
    str(WHITELIST_TEST.relative_to(ROOT)),
)
run(
    sys.executable,
    "-m",
    "compileall",
    "-q",
    "src/nifty_scalper_bot/strategies/signal_generator.py",
    "src/nifty_scalper_bot/strategies/runner.py",
    "tests/strategies/test_active_futures_resolution.py",
    "tests/strategies/test_runner_active_basket_pending.py",
)
run("git", "diff", "--check")

run("git", "config", "user.name", "github-actions[bot]")
run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
run("git", "rm", str(THIS_FILE.relative_to(ROOT)), str(WORKFLOW.relative_to(ROOT)))
run(
    "git",
    "add",
    "src/nifty_scalper_bot/strategies/signal_generator.py",
    "src/nifty_scalper_bot/strategies/runner.py",
    "tests/strategies/test_active_futures_resolution.py",
    "tests/strategies/test_runner_active_basket_pending.py",
)
run(
    "git",
    "commit",
    "-m",
    "fix(runtime): preserve futures volume provenance and promotion whitelist",
)
run(
    "git",
    "push",
    "origin",
    "HEAD:fix/futures-volume-provenance-whitelist-20260804",
)
