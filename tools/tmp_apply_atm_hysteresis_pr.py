from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/nifty_scalper_bot/core/instrument_manager.py"
TESTS = ROOT / "tests/core/test_instrument_manager_active_contract_basket.py"
WORKFLOW = ROOT / ".github/workflows/tmp-atm-hysteresis-pr.yml"
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


append_once(
    TESTS,
    "test_active_contract_basket_holds_atm_inside_hysteresis_band",
    r'''
    def test_active_contract_basket_holds_atm_inside_hysteresis_band(monkeypatch):
        rows, _weekly, _monthly = _dump()
        mgr = InstrumentManager(KiteDump(rows))
        monkeypatch.setenv("ATM_STRIKE_HYSTERESIS_POINTS", "5")

        assert mgr.get_active_nifty_contracts(25010).atm_strike == 25000
        assert mgr.get_active_nifty_contracts(25024).atm_strike == 25000
        assert mgr.get_active_nifty_contracts(25026).atm_strike == 25000
        assert mgr.get_active_nifty_contracts(25023).atm_strike == 25000


    def test_active_contract_basket_shifts_once_after_confirmed_crossing(monkeypatch):
        rows, _weekly, _monthly = _dump()
        mgr = InstrumentManager(KiteDump(rows))
        monkeypatch.setenv("ATM_STRIKE_HYSTERESIS_POINTS", "5")

        assert mgr.get_active_nifty_contracts(25010).atm_strike == 25000
        assert mgr.get_active_nifty_contracts(25030).atm_strike == 25050
        assert mgr.get_active_nifty_contracts(25024).atm_strike == 25050
        assert mgr.get_active_nifty_contracts(25020).atm_strike == 25000


    def test_hysteresis_does_not_block_option_or_future_rollover(monkeypatch):
        from nifty_scalper_bot.core import instrument_manager as im

        rows, weekly, monthly = _dump()
        rows.append(_row(3, "NIFTYWEEKFUT", "FUT", weekly))
        mgr = InstrumentManager(KiteDump(rows))
        monkeypatch.setenv("ATM_STRIKE_HYSTERESIS_POINTS", "5")
        trading_day = {"value": weekly - timedelta(days=1)}
        monkeypatch.setattr(im, "_exchange_trading_date", lambda: trading_day["value"])

        initial = mgr.get_active_nifty_contracts(25010)
        assert initial.atm_strike == 25000
        assert initial.option_expiry == weekly
        assert initial.futures_symbol == "NFO:NIFTYWEEKFUT"

        trading_day["value"] = weekly + timedelta(days=1)
        rolled = mgr.get_active_nifty_contracts(25026)
        assert rolled.atm_strike == 25000
        assert rolled.option_expiry == monthly
        assert rolled.futures_symbol == "NFO:NIFTYCURFUT"


    def test_hysteresis_falls_back_when_held_pair_is_unavailable(monkeypatch):
        from nifty_scalper_bot.core import instrument_manager as im

        rows, weekly, monthly = _dump()
        rows = [
            row
            for row in rows
            if not (
                row.get("expiry") == monthly
                and row.get("instrument_type") in {"CE", "PE"}
                and row.get("strike") == 25000
            )
        ]
        mgr = InstrumentManager(KiteDump(rows))
        monkeypatch.setenv("ATM_STRIKE_HYSTERESIS_POINTS", "5")
        trading_day = {"value": weekly - timedelta(days=1)}
        monkeypatch.setattr(im, "_exchange_trading_date", lambda: trading_day["value"])

        assert mgr.get_active_nifty_contracts(25010).atm_strike == 25000
        trading_day["value"] = weekly + timedelta(days=1)

        rolled = mgr.get_active_nifty_contracts(25026)
        assert rolled.option_expiry == monthly
        assert rolled.atm_strike == 25050
        assert rolled.selected_ce == "NFO:NIFTYM25050CE"
        assert rolled.selected_pe == "NFO:NIFTYM25050PE"
    ''',
)

run(
    sys.executable,
    "-m",
    "pytest",
    "-q",
    str(TESTS.relative_to(ROOT)),
    "-k",
    "test_active_contract_basket_holds_atm_inside_hysteresis_band",
    expect=1,
)

replace_once(
    SOURCE,
    '''        self._lock = threading.RLock()
        self._loaded = False
''',
    '''        self._lock = threading.RLock()
        self._loaded = False
        self._active_nifty_atm_strike: int | None = None
''',
)

replace_once(
    SOURCE,
    '''    def get_active_nifty_contracts(
        self,
        spot_price: float,
''',
    '''    def _stabilized_nifty_atm_strike(
        self, spot: float, strike_step: int
    ) -> tuple[int, int, int, bool]:
        """Return ATM with a dead-band around the currently active strike."""
        raw_atm = _atm_strike_for_spot(spot, strike_step)
        with self._lock:
            active_atm = self._active_nifty_atm_strike
        if active_atm is None or active_atm <= 0 or active_atm % strike_step:
            return raw_atm, raw_atm, 0, False

        configured = parse_int_env(os.getenv("ATM_STRIKE_HYSTERESIS_POINTS"), 5)
        max_hysteresis = max(0, (strike_step // 2) - 1)
        hysteresis = max(0, min(configured, max_hysteresis))
        if hysteresis == 0 or raw_atm == active_atm:
            return raw_atm, raw_atm, hysteresis, False

        lower_boundary = active_atm - (strike_step / 2.0) - hysteresis
        upper_boundary = active_atm + (strike_step / 2.0) + hysteresis
        if lower_boundary < spot < upper_boundary:
            return active_atm, raw_atm, hysteresis, True
        return raw_atm, raw_atm, hysteresis, False

    def get_active_nifty_contracts(
        self,
        spot_price: float,
''',
)

replace_once(
    SOURCE,
    '''        atm = _atm_strike_for_spot(spot, int(strike_step))
        around = max(0, int(strikes_around_atm))
        target_strikes = {float(atm + i * int(strike_step)) for i in range(-around, around + 1)}
        expiry_options = [item for item in options if item[0] == option_expiry]

        selected_ce = next((item for item in expiry_options if item[2] == float(atm) and item[3] == "CE"), None)
        selected_pe = next((item for item in expiry_options if item[2] == float(atm) and item[3] == "PE"), None)
        if selected_ce is None or selected_pe is None:
            raise RuntimeError(
                f"CONTRACT_SSOT_ATM_PAIR_MISSING atm_strike={atm} expiry={option_expiry} "
                f"ce_found={selected_ce is not None} pe_found={selected_pe is not None}"
            )
''',
    '''        atm, raw_atm, atm_hysteresis, hysteresis_held = (
            self._stabilized_nifty_atm_strike(spot, int(strike_step))
        )
        around = max(0, int(strikes_around_atm))
        expiry_options = [item for item in options if item[0] == option_expiry]

        def find_atm_pair(
            strike: int,
        ) -> tuple[
            tuple[date, int, float, str, dict[str, Any]] | None,
            tuple[date, int, float, str, dict[str, Any]] | None,
        ]:
            ce = next(
                (
                    item
                    for item in expiry_options
                    if item[2] == float(strike) and item[3] == "CE"
                ),
                None,
            )
            pe = next(
                (
                    item
                    for item in expiry_options
                    if item[2] == float(strike) and item[3] == "PE"
                ),
                None,
            )
            return ce, pe

        selected_ce, selected_pe = find_atm_pair(atm)
        if (selected_ce is None or selected_pe is None) and atm != raw_atm:
            raw_ce, raw_pe = find_atm_pair(raw_atm)
            if raw_ce is not None and raw_pe is not None:
                LOGGER.warning(
                    "CONTRACT_SSOT_ATM_HYSTERESIS_FALLBACK held_atm=%s raw_atm=%s expiry=%s reason=held_pair_unavailable",
                    atm,
                    raw_atm,
                    option_expiry,
                    extra={
                        "event": "CONTRACT_SSOT_ATM_HYSTERESIS_FALLBACK",
                        "held_atm": atm,
                        "raw_atm": raw_atm,
                        "expiry": str(option_expiry),
                        "reason": "held_pair_unavailable",
                    },
                )
                atm = raw_atm
                selected_ce, selected_pe = raw_ce, raw_pe
                hysteresis_held = False
        if selected_ce is None or selected_pe is None:
            raise RuntimeError(
                f"CONTRACT_SSOT_ATM_PAIR_MISSING atm_strike={atm} expiry={option_expiry} "
                f"ce_found={selected_ce is not None} pe_found={selected_pe is not None}"
            )

        target_strikes = {
            float(atm + i * int(strike_step)) for i in range(-around, around + 1)
        }
        if hysteresis_held:
            LOGGER.debug(
                "CONTRACT_SSOT_ATM_HYSTERESIS_HELD active_atm=%s raw_atm=%s spot=%s hysteresis=%s",
                atm,
                raw_atm,
                spot,
                atm_hysteresis,
                extra={
                    "event": "CONTRACT_SSOT_ATM_HYSTERESIS_HELD",
                    "active_atm": atm,
                    "raw_atm": raw_atm,
                    "spot_price": spot,
                    "hysteresis_points": atm_hysteresis,
                },
            )
''',
)

replace_once(
    SOURCE,
    '''                "future_available": bool(future_symbol),
            },
        )
''',
    '''                "future_available": bool(future_symbol),
                "raw_atm_strike": raw_atm,
                "atm_hysteresis_points": atm_hysteresis,
                "atm_hysteresis_held": hysteresis_held,
            },
        )
        with self._lock:
            self._active_nifty_atm_strike = atm
''',
)

run(
    sys.executable,
    "-m",
    "pytest",
    "-q",
    str(TESTS.relative_to(ROOT)),
)
run(
    sys.executable,
    "-m",
    "compileall",
    "-q",
    str(SOURCE.relative_to(ROOT)),
    str(TESTS.relative_to(ROOT)),
)
run("git", "diff", "--check")

run("git", "config", "user.name", "github-actions[bot]")
run(
    "git",
    "config",
    "user.email",
    "41898282+github-actions[bot]@users.noreply.github.com",
)
run("git", "rm", str(THIS_FILE.relative_to(ROOT)), str(WORKFLOW.relative_to(ROOT)))
run(
    "git",
    "add",
    str(SOURCE.relative_to(ROOT)),
    str(TESTS.relative_to(ROOT)),
)
run("git", "commit", "-m", "fix(runtime): stabilize ATM basket selection")
run("git", "push", "origin", "HEAD:fix/atm-basket-hysteresis-20260804")
