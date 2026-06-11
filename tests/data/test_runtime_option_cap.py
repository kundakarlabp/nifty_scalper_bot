"""Regression tests: runtime option-token clamp, post-close dynamic skip."""

from __future__ import annotations

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.strategies.runner import StrategyRunner


class _BareMDM:
    _clamp_option_tokens = MarketDataManager._clamp_option_tokens

    def __init__(self) -> None:
        import logging

        self._logger = logging.getLogger("test_mdm")


def _token_map(n_options: int = 14) -> tuple[list[int], dict[str, int]]:
    token_map: dict[str, int] = {"NSE:NIFTY": 1, "NFO:NIFTY26JUNFUT": 2}
    tokens = [1, 2]
    for i in range(n_options):
        side = "CE" if i % 2 == 0 else "PE"
        strike = 23150 + (i // 2) * 50
        sym = f"NFO:NIFTY26616{strike}{side}"
        tok = 100 + i
        token_map[sym] = tok
        tokens.append(tok)
    return tokens, token_map


def test_runtime_clamp_caps_option_tokens_to_8(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ACTIVE_OPTION_SYMBOLS", "8")
    monkeypatch.delenv("DIAGNOSTIC_FULL_UNIVERSE", raising=False)
    tokens, token_map = _token_map(14)
    assert len(tokens) == 16
    mdm = _BareMDM()
    clamped = mdm._clamp_option_tokens(tokens, token_map, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    option_tokens = [t for t in clamped if t >= 100]
    assert len(option_tokens) == 8
    assert len(clamped) == 10  # 8 options + spot + future
    assert token_map["NFO:NIFTY2661623300CE"] in clamped
    assert token_map["NFO:NIFTY2661623300PE"] in clamped
    assert 1 in clamped and 2 in clamped  # spot/future never clamped


def test_runtime_clamp_diagnostic_bypass(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ACTIVE_OPTION_SYMBOLS", "8")
    monkeypatch.setenv("DIAGNOSTIC_FULL_UNIVERSE", "true")
    tokens, token_map = _token_map(14)
    mdm = _BareMDM()
    assert mdm._clamp_option_tokens(tokens, token_map, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE") == tokens


def test_runtime_clamp_noop_under_cap(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ACTIVE_OPTION_SYMBOLS", "8")
    monkeypatch.delenv("DIAGNOSTIC_FULL_UNIVERSE", raising=False)
    tokens, token_map = _token_map(6)
    mdm = _BareMDM()
    assert mdm._clamp_option_tokens(tokens, token_map, "NFO:NIFTY2661623150CE", "NFO:NIFTY2661623150PE") == tokens


class _FakePositions:
    def __init__(self, open_positions: list[object]) -> None:
        self._open = open_positions

    def get_open_positions(self) -> list[object]:
        return self._open


def test_dynamic_add_blocked_when_closed_and_flat(monkeypatch) -> None:
    import nifty_scalper_bot.strategies.runner as runner_mod

    runner = object.__new__(StrategyRunner)
    runner._position_manager = _FakePositions([])  # type: ignore[attr-defined]
    monkeypatch.delenv("DIAGNOSTIC_FULL_UNIVERSE", raising=False)
    monkeypatch.setattr(runner_mod, "get_market_state", lambda: runner_mod.MarketState.CLOSED)
    assert StrategyRunner._dynamic_add_blocked_market_closed(runner) is True
    # open position -> allowed (exit management may need new symbols)
    runner._position_manager = _FakePositions([object()])  # type: ignore[attr-defined]
    assert StrategyRunner._dynamic_add_blocked_market_closed(runner) is False
    # market open -> allowed
    runner._position_manager = _FakePositions([])  # type: ignore[attr-defined]
    monkeypatch.setattr(runner_mod, "get_market_state", lambda: runner_mod.MarketState.OPEN)
    assert StrategyRunner._dynamic_add_blocked_market_closed(runner) is False
    # diagnostic bypass
    monkeypatch.setattr(runner_mod, "get_market_state", lambda: runner_mod.MarketState.CLOSED)
    monkeypatch.setenv("DIAGNOSTIC_FULL_UNIVERSE", "true")
    assert StrategyRunner._dynamic_add_blocked_market_closed(runner) is False
