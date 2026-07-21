from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


@dataclass
class Hydration:
    symbol: str
    minimum_ready: bool = True
    target_ready: bool = True
    fetched_rows: int = 0
    accepted_rows: int = 0
    failure_reason: str | None = None


@dataclass
class Sync:
    runner_bars: int
    indicator_bars: int
    success: bool = True
    failure_reason: str | None = None


class MDM:
    def __init__(self, bars: int = 0):
        self.bars = bars
        self.calls = []

    async def ensure_history(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        if kwargs.get("target_bars", 0) > self.bars:
            self.bars = kwargs["target_bars"]
        return Hydration(symbol, fetched_rows=self.bars, accepted_rows=self.bars)

    def get_ohlc_bars(self, *_a, **_k):
        return [object()] * self.bars


class Runner:
    def __init__(self, bars: int = 0):
        self.bars = bars
        self.calls = []
        self._context_required_bars = 20
        self._option_required_bars = 30

    def sync_history_from_mdm(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        self.bars = max(self.bars, kwargs["required_bars"])
        return Sync(self.bars, self.bars)


def ctx(mdm=None, runner=None):
    return SimpleNamespace(
        market_data_manager=mdm, strategy_runner=runner, settings=SimpleNamespace()
    )


@pytest.mark.asyncio
async def test_spot_startup_policy_and_orchestration() -> None:
    mdm = MDM()
    runner = Runner()
    result = await app.ensure_symbol_runtime_history(
        ctx(mdm, runner), "NSE:NIFTY", role="spot_context", phase="startup", reason="t"
    )
    assert result.minimum_ready and result.sync_success
    assert mdm.calls and runner.calls


@pytest.mark.asyncio
async def test_selected_option_priority_and_target() -> None:
    mdm = MDM()
    runner = Runner()
    result = await app.ensure_symbol_runtime_history(
        ctx(mdm, runner), "NFO:CE", role="selected_option", phase="startup", reason="t"
    )
    assert result.required_bars >= 30
    assert result.target_bars >= result.required_bars


@pytest.mark.asyncio
async def test_context_option_post_market_suppresses_broker(monkeypatch) -> None:
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "POST_MARKET")
    mdm = MDM()
    runner = Runner()
    result = await app.ensure_symbol_runtime_history(
        ctx(mdm, runner),
        "NFO:CTXCE",
        role="option_context",
        phase="dynamic_update",
        reason="t",
    )
    assert result.failure_reason == "broker_fetch_not_allowed"
    assert mdm.calls == []


@pytest.mark.asyncio
async def test_warm_runner_sync_only_no_broker_refetch(monkeypatch) -> None:
    mdm = MDM(bars=300)
    runner = Runner()
    result = await app.ensure_symbol_runtime_history(
        ctx(mdm, runner),
        "NFO:FUT",
        role="futures_context",
        phase="dynamic_update",
        reason="futures_context_refresh",
    )
    assert result.minimum_ready
    assert (
        len(mdm.calls) == 1
    )  # ensure_history performs target-sufficient skip inside MDM


@pytest.mark.asyncio
async def test_missing_interfaces_return_controlled_failures() -> None:
    assert (
        await app.ensure_symbol_runtime_history(
            ctx(SimpleNamespace(), Runner()),
            "NSE:NIFTY",
            role="spot_context",
            phase="startup",
            reason="t",
        )
    ).failure_reason == "mdm_ensure_history_missing"
    assert (
        await app.ensure_symbol_runtime_history(
            ctx(MDM(30), SimpleNamespace()),
            "NSE:NIFTY",
            role="spot_context",
            phase="startup",
            reason="t",
        )
    ).failure_reason == "runner_sync_history_missing"


async def test_role_caps_bound_targets_modestly(monkeypatch):
    """Spec §1: targets are bounded by per-role caps, not derived deep.
    async so it executes under the repo conftest pyfunc hook."""
    import nifty_scalper_bot.core.app as app

    class _Runner:
        _option_required_bars = 30
        _context_required_bars = 20
        _required_candles = 200  # deliberately large generic requirement

    class _MDM:
        _min_required_bars = 250  # deliberately large

        def history_capacity_for(self, *_a, **_k):
            return 1000

    ctx = SimpleNamespace(strategy_runner=_Runner(), market_data_manager=_MDM())
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "OPEN", raising=False)
    monkeypatch.delenv("HYDRATION_MAX_BARS", raising=False)

    sel = app.resolve_history_policy(
        ctx,
        "NFO:NIFTY2661623150CE",
        role="selected_option",
        phase="startup",
        reason="t",
    )
    assert sel.target_bars <= 75, sel.target_bars
    assert sel.required_bars >= 30
    octx = app.resolve_history_policy(
        ctx, "NFO:NIFTY2661623450CE", role="option_context", phase="startup", reason="t"
    )
    assert octx.target_bars <= 50, octx.target_bars
    spot = app.resolve_history_policy(
        ctx, "NSE:NIFTY 50", role="spot_context", phase="startup", reason="t"
    )
    assert spot.target_bars <= 100, spot.target_bars


async def test_capacity_clamps_explicit_target(monkeypatch):
    """Spec §2/§3: explicit target override is clamped to MDM retention."""
    import nifty_scalper_bot.core.app as app

    captured = {}

    class _MDM:
        _min_required_bars = 0

        def history_capacity_for(self, *_a, **_k):
            return 40  # smaller than the 75 selected_option cap

        async def ensure_history(self, symbol, **kw):
            captured.update(kw)
            return SimpleNamespace(failure_reason=None)

        def get_ohlc_bars(self, *_a, **_k):
            return []

    class _Runner:
        _option_required_bars = 30
        _required_candles = 60

        def sync_history_from_mdm(self, symbol, **kw):
            return SimpleNamespace(
                runner_bars=40, indicator_bars=40, success=True, failure_reason=None
            )

    ctx = SimpleNamespace(strategy_runner=_Runner(), market_data_manager=_MDM())
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "OPEN", raising=False)
    await app.ensure_symbol_runtime_history(
        ctx,
        "NFO:NIFTY2661623150CE",
        role="selected_option",
        phase="startup",
        reason="t",
        target_bars=500,
    )
    assert captured["target_bars"] <= 40, captured.get("target_bars")


async def test_explicit_deep_target_300_supported_for_selected_option(monkeypatch):
    """Spec §2/§6: explicit deep target (e.g. EMA200 warm-up) is honored up to
    the role deep cap, not silently reduced to the normal role cap."""
    import nifty_scalper_bot.core.app as app

    captured = {}

    class _MDM:
        _min_required_bars = 0

        def history_capacity_for(self, *_a, **_k):
            return 1000

        async def ensure_history(self, symbol, **kw):
            captured.update(kw)
            return SimpleNamespace(failure_reason=None)

        def get_ohlc_bars(self, *_a, **_k):
            return []

    class _Runner:
        _option_required_bars = 30
        _required_candles = 60

        def sync_history_from_mdm(self, symbol, **kw):
            return SimpleNamespace(
                runner_bars=300, indicator_bars=300, success=True, failure_reason=None
            )

    ctx = SimpleNamespace(strategy_runner=_Runner(), market_data_manager=_MDM())
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "OPEN", raising=False)
    await app.ensure_symbol_runtime_history(
        ctx,
        "NFO:NIFTY2661623150CE",
        role="selected_option",
        phase="startup",
        reason="ema200",
        target_bars=300,
    )
    assert captured["target_bars"] == 300, captured  # deep target preserved


async def test_explicit_required_cannot_exceed_deep_cap(monkeypatch):
    """Spec §6: an explicit required override is clamped to the deep cap, not
    allowed to demand unbounded history."""
    import nifty_scalper_bot.core.app as app

    captured = {}

    class _MDM:
        _min_required_bars = 0

        def history_capacity_for(self, *_a, **_k):
            return 1000

        async def ensure_history(self, symbol, **kw):
            captured.update(kw)
            return SimpleNamespace(failure_reason=None)

        def get_ohlc_bars(self, *_a, **_k):
            return []

    class _Runner:
        _option_required_bars = 30
        _required_candles = 60

        def sync_history_from_mdm(self, symbol, **kw):
            return SimpleNamespace(
                runner_bars=0, indicator_bars=0, success=False, failure_reason=None
            )

    ctx = SimpleNamespace(strategy_runner=_Runner(), market_data_manager=_MDM())
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "OPEN", raising=False)
    monkeypatch.setenv("HYDRATION_DEEP_SELECTED_OPTION", "300")
    await app.ensure_symbol_runtime_history(
        ctx,
        "NFO:NIFTY2661623150CE",
        role="selected_option",
        phase="recovery",
        reason="x",
        required_bars=999,
    )
    assert captured["required_bars"] <= 300, captured


async def test_generic_sync_helper_resolves_role(monkeypatch):
    """Spec §5: _sync_mdm_bars_to_runner resolves role."""
    import nifty_scalper_bot.core.app as app

    seen = {}

    class _Runner:
        _active_selected_ce = "NFO:NIFTY2661623300CE"
        _active_selected_pe = "NFO:NIFTY2661623300PE"
        _active_futures_symbol = "NFO:NIFTY26JUNFUT"

        def sync_history_from_mdm(self, symbol, **kw):
            seen[symbol] = kw.get("role")
            return SimpleNamespace(indicator_bars=10)

    ctx = SimpleNamespace(
        strategy_runner=_Runner(),
        active_contract_basket=None,
        position_manager=None,
        spot_symbol="NSE:NIFTY",
    )
    # a far context strike must NOT resolve to selected_option
    app._sync_mdm_bars_to_runner(ctx, "NFO:NIFTY2661623450CE", min_bars=10)
    assert seen["NFO:NIFTY2661623450CE"] == "option_context"
    # the selected CE must resolve to selected_option
    app._sync_mdm_bars_to_runner(ctx, "NFO:NIFTY2661623300CE", min_bars=10)
    assert seen["NFO:NIFTY2661623300CE"] == "selected_option"


async def test_large_required_without_deep_mode_clamps_with_diagnostic(
    monkeypatch, caplog
):
    """Spec §7: required_bars > role_cap WITHOUT deep mode is clamped, logged."""
    import logging

    import nifty_scalper_bot.core.app as app

    captured = {}

    class _MDM:
        _min_required_bars = 0

        def history_capacity_for(self, *_a, **_k):
            return 1000

        async def ensure_history(self, symbol, **kw):
            captured.update(kw)
            return SimpleNamespace(failure_reason=None)

        def get_ohlc_bars(self, *_a, **_k):
            return []

    class _Runner:
        _option_required_bars = 30
        _required_candles = 60

        def sync_history_from_mdm(self, symbol, **kw):
            return SimpleNamespace(
                runner_bars=0, indicator_bars=0, success=False, failure_reason=None
            )

    ctx = SimpleNamespace(strategy_runner=_Runner(), market_data_manager=_MDM())
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "OPEN", raising=False)
    with caplog.at_level(logging.INFO, logger="nifty_scalper_bot.core.app"):
        await app.ensure_symbol_runtime_history(
            ctx,
            "NFO:NIFTY2661623150CE",
            role="selected_option",
            phase="startup",
            reason="legacy_big",
            required_bars=300,  # no deep_history, no target
        )
    # clamped to normal role cap (75), NOT 300
    assert captured["required_bars"] <= 75, captured
    assert any(
        getattr(r, "event", "") == "HISTORY_REQUIRED_BARS_CLAMPED"
        for r in caplog.records
    )


async def test_deep_history_flag_allows_large_required(monkeypatch):
    """Spec §7: explicit deep_history=True permits the role deep cap."""
    import nifty_scalper_bot.core.app as app

    captured = {}

    class _MDM:
        _min_required_bars = 0

        def history_capacity_for(self, *_a, **_k):
            return 1000

        async def ensure_history(self, symbol, **kw):
            captured.update(kw)
            return SimpleNamespace(failure_reason=None)

        def get_ohlc_bars(self, *_a, **_k):
            return []

    class _Runner:
        _option_required_bars = 30
        _required_candles = 60

        def sync_history_from_mdm(self, symbol, **kw):
            return SimpleNamespace(
                runner_bars=0, indicator_bars=0, success=False, failure_reason=None
            )

    ctx = SimpleNamespace(strategy_runner=_Runner(), market_data_manager=_MDM())
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "OPEN", raising=False)
    monkeypatch.setenv("HYDRATION_DEEP_SELECTED_OPTION", "300")
    await app.ensure_symbol_runtime_history(
        ctx,
        "NFO:NIFTY2661623150CE",
        role="selected_option",
        phase="recovery",
        reason="ema200",
        required_bars=200,
        target_bars=300,
        deep_history=True,
    )
    assert captured["target_bars"] == 300, captured
    assert captured["required_bars"] == 200, captured  # preserved, not clamped


def _active_ctx(mdm=None, runner=None):
    return SimpleNamespace(
        market_data_manager=mdm or MDM(),
        strategy_runner=runner or Runner(),
        settings=SimpleNamespace(),
        active_symbol_tokens={},
        active_contract_basket={
            "basket_version": "v1",
            "spot_symbol": "NSE:NIFTY",
            "futures_symbol": "NFO:NIFTY26JUNFUT",
            "selected_ce": "NFO:NIFTY2661623000CE",
            "selected_pe": "NFO:NIFTY2661623000PE",
            "option_symbols": [
                "NFO:NIFTY2661623000CE",
                "nfo:nifty2661623000ce",
                "NFO:NIFTY2661623100CE",
            ],
        },
        active_trading_universe={},
    )


class _ActiveMDM:
    def __init__(self):
        self.calls = []

    def get_ohlc_bars(self, symbol, *args, **kwargs):
        return []

    async def ensure_history(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        return SimpleNamespace(symbol=symbol, failure_reason=None)


class _ActiveRunner:
    def __init__(self):
        self.calls = []
        self.runner_bars = 0
        self.indicator_bars = 0

    def sync_history_from_mdm(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        self.runner_bars = kwargs["required_bars"]
        self.indicator_bars = kwargs["required_bars"]
        return SimpleNamespace(
            success=True,
            runner_bars=self.runner_bars,
            indicator_bars=self.indicator_bars,
        )


def _status(
    *,
    bars=30,
    runner_bars=30,
    indicator_bars=30,
    fresh=True,
    contiguous=True,
    ready=True,
):
    return SimpleNamespace(
        mdm_bars=bars,
        runner_bars=runner_bars,
        indicator_bars=indicator_bars,
        latest_bar_fresh=fresh,
        recent_window_contiguous=contiguous,
        ready_for_evaluation=ready,
    )


def test_app_direct_mdm_ensure_is_centralized_in_runtime_history_helpers() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse(Path(app.__file__).read_text())
    callers: set[str] = set()
    parents: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def generic_visit(self, node):
            parents.append(node)
            super().generic_visit(node)
            parents.pop()

        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "ensure_history"
            ):
                for parent in reversed(parents):
                    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        callers.add(parent.name)
                        break
            self.generic_visit(node)

    Visitor().visit(tree)
    assert callers == set()


@pytest.mark.asyncio
async def test_active_basket_helper_deduplicates_selected_and_passes_short_reason(
    monkeypatch,
) -> None:
    mdm = _ActiveMDM()
    runner = _ActiveRunner()
    c = _active_ctx(mdm, runner)
    monkeypatch.setattr(
        app,
        "build_symbol_hydration_status",
        lambda *_a, **_k: _status(bars=0, fresh=True, contiguous=True),
    )
    await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )
    symbols = [symbol for symbol, _ in mdm.calls]
    assert symbols.count("NFO:NIFTY2661623000CE") == 1
    assert {kw["reason"] for _, kw in mdm.calls} == {"history_short"}
    assert {kw["phase"] for _, kw in mdm.calls} == {"runtime"}
    assert len(runner.calls) == len(mdm.calls)


@pytest.mark.asyncio
async def test_active_basket_helper_skips_ready_state_and_wrappers_delegate(
    monkeypatch,
) -> None:
    mdm = _ActiveMDM()
    runner = _ActiveRunner()
    c = _active_ctx(mdm, runner)
    monkeypatch.setattr(
        app,
        "build_symbol_hydration_status",
        lambda *_a, **_k: _status(bars=40, fresh=True, contiguous=True),
    )
    result = await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )
    assert mdm.calls == []
    assert all(value["reason"] == "ready" for value in result.values())

    async def fake_active(*a, **k):
        return {"NSE:NIFTY": {"ok": True}}

    monkeypatch.setattr(app, "_ensure_active_basket_history", fake_active)
    wrapped = await app._ensure_context_history_hydrated(
        c, "NSE:NIFTY", None, 20, "compat"
    )
    assert wrapped == {"NSE:NIFTY": {"ok": True}}


@pytest.mark.asyncio
async def test_warm_mdm_reseeds_cold_runner_without_broker_fetch(monkeypatch) -> None:
    mdm = _ActiveMDM()
    runner = _ActiveRunner()
    c = _active_ctx(mdm, runner)
    monkeypatch.setattr(
        app,
        "build_symbol_hydration_status",
        lambda *_a, **_k: _status(
            bars=40, runner_bars=0, indicator_bars=0, fresh=True, contiguous=True
        ),
    )

    result = await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )

    assert mdm.calls == []
    assert runner.calls
    selected = [call for call in runner.calls if call[0] == "NFO:NIFTY2661623000CE"]
    assert len(selected) == 1
    assert selected[0][1]["required_bars"] == 30
    assert selected[0][1]["reason"] == "mdm_cache_sync"
    assert runner.runner_bars >= 30
    assert runner.indicator_bars >= 30
    assert result["NFO:NIFTY2661623000CE"]["reason"] == "synced_from_mdm"


@pytest.mark.asyncio
async def test_warm_mdm_and_warm_runner_do_not_fetch_or_sync(monkeypatch) -> None:
    mdm = _ActiveMDM()
    runner = _ActiveRunner()
    c = _active_ctx(mdm, runner)
    monkeypatch.setattr(
        app,
        "build_symbol_hydration_status",
        lambda *_a, **_k: _status(
            bars=40, runner_bars=40, indicator_bars=40, fresh=True, contiguous=True
        ),
    )

    result = await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )

    assert mdm.calls == []
    assert runner.calls == []
    assert all(value["reason"] == "ready" for value in result.values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fresh", "contiguous", "expected_reason", "expected_phase"),
    [
        (False, True, "history_stale", "recovery"),
        (True, False, "recovery_gap_fill", "recovery"),
    ],
)
async def test_active_basket_helper_uses_recovery_phase_for_stale_and_gap(
    monkeypatch, fresh, contiguous, expected_reason, expected_phase
) -> None:
    mdm = _ActiveMDM()
    c = _active_ctx(mdm, _ActiveRunner())
    monkeypatch.setattr(
        app,
        "build_symbol_hydration_status",
        lambda *_a, **_k: _status(bars=40, fresh=fresh, contiguous=contiguous),
    )
    await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )
    assert mdm.calls
    assert {kw["reason"] for _, kw in mdm.calls} == {expected_reason}
    assert {kw["phase"] for _, kw in mdm.calls} == {expected_phase}


@pytest.mark.asyncio
async def test_active_basket_hydration_skips_symbol_subscription_outage(monkeypatch):
    mdm = _ActiveMDM()
    c = _active_ctx(mdm, _ActiveRunner())
    mdm.classify_transport_backlog = lambda symbol: {
        "transport_classification": (
            "symbol_subscription_stale"
            if symbol == "NFO:NIFTY26JUNFUT"
            else "transport_healthy"
        )
    }

    monkeypatch.setattr(
        app,
        "build_symbol_hydration_status",
        lambda _ctx, symbol, _role, _required: _status(
            bars=40,
            runner_bars=40,
            indicator_bars=40,
            fresh=symbol != "NFO:NIFTY26JUNFUT",
            contiguous=True,
        ),
    )

    result = await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )

    assert mdm.calls == []
    assert result["NFO:NIFTY26JUNFUT"]["reason"] == "symbol_subscription_stale"


@pytest.mark.asyncio
async def test_active_basket_cooldown_bypassed_by_change_or_requirement(
    monkeypatch,
) -> None:
    mdm = _ActiveMDM()
    c = _active_ctx(mdm, _ActiveRunner())
    monkeypatch.setenv("HISTORY_HYDRATION_RETRY_COOLDOWN_SECONDS", "120")
    monkeypatch.setattr(
        app,
        "build_symbol_hydration_status",
        lambda *_a, **_k: _status(bars=0, fresh=True, contiguous=True),
    )
    await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )
    first = len(mdm.calls)
    await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )
    assert len(mdm.calls) == first
    c.active_contract_basket["basket_version"] = "v2"
    await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )
    assert len(mdm.calls) == first * 2
    await app._ensure_active_basket_history(
        c,
        option_required_bars=31,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )
    assert len(mdm.calls) > first * 2


@pytest.mark.asyncio
async def test_non_gating_option_context_gap_does_not_fetch_or_sync(
    monkeypatch,
) -> None:
    mdm = _ActiveMDM()
    runner = _ActiveRunner()
    c = _active_ctx(mdm, runner)
    context_symbol = "NFO:NIFTY2661623100CE"

    def fake_status(_ctx, symbol, _role, _required):
        if symbol == context_symbol:
            return _status(
                bars=40,
                runner_bars=40,
                indicator_bars=40,
                fresh=False,
                contiguous=False,
            )
        return _status(bars=40, runner_bars=40, indicator_bars=40)

    monkeypatch.setattr(app, "build_symbol_hydration_status", fake_status)

    result = await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )

    assert [call for call in mdm.calls if call[0] == context_symbol] == []
    assert [call for call in runner.calls if call[0] == context_symbol] == []
    assert result[context_symbol]["reason"] in {"ready", "skipped", "non_gating"}


@pytest.mark.asyncio
async def test_selected_option_gap_still_fetches_recovery_gap_fill(monkeypatch) -> None:
    mdm = _ActiveMDM()
    runner = _ActiveRunner()
    c = _active_ctx(mdm, runner)
    selected_symbol = "NFO:NIFTY2661623000CE"

    def fake_status(_ctx, symbol, _role, _required):
        if symbol == selected_symbol:
            return _status(
                bars=40,
                runner_bars=40,
                indicator_bars=40,
                fresh=True,
                contiguous=False,
            )
        return _status(bars=40, runner_bars=40, indicator_bars=40)

    monkeypatch.setattr(app, "build_symbol_hydration_status", fake_status)

    result = await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )

    selected_calls = [call for call in mdm.calls if call[0] == selected_symbol]
    assert selected_calls
    assert selected_calls[0][1]["reason"] == "recovery_gap_fill"
    assert result[selected_symbol].symbol == selected_symbol


@pytest.mark.asyncio
async def test_non_gating_option_context_short_history_still_hydrates(
    monkeypatch,
) -> None:
    mdm = _ActiveMDM()
    runner = _ActiveRunner()
    c = _active_ctx(mdm, runner)
    context_symbol = "NFO:NIFTY2661623100CE"

    def fake_status(_ctx, symbol, _role, _required):
        if symbol == context_symbol:
            return _status(
                bars=10,
                runner_bars=40,
                indicator_bars=40,
                fresh=True,
                contiguous=True,
            )
        return _status(bars=40, runner_bars=40, indicator_bars=40)

    monkeypatch.setattr(app, "build_symbol_hydration_status", fake_status)

    await app._ensure_active_basket_history(
        c,
        option_required_bars=30,
        context_required_bars=20,
        reason="readiness",
        phase="runtime",
    )

    context_calls = [call for call in mdm.calls if call[0] == context_symbol]
    assert context_calls
    assert context_calls[0][1]["reason"] == "history_short"
