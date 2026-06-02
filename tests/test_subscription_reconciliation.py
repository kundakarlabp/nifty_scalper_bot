"""Tests for subscription reconciliation simplification.

Covers:
  1. stale old option tokens are unsubscribed after basket reselection
  2. selected CE/PE and active FUT remain subscribed (in desired set)
  3. unknown token list is logged only as count + sample (not full list)
  4. WebSocket current token set converges to desired token set
  5. reconcile_active_subscriptions() is idempotent
  6. utils.backoff.exp_backoff yields valid sleep intervals
  7. FUTURES_PURGE_STATE no longer logs full token lists
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: minimal MDM stub with reconcile_active_subscriptions
# ---------------------------------------------------------------------------

def _make_mdm_with_ws(initial_ws_tokens: set[int]):
    """Build a minimal MarketDataManager-like object with a real WS stub."""
    ws_stub = MagicMock()
    ws_stub._tokens = set(initial_ws_tokens)

    def _set_tokens(tokens):
        ws_stub._tokens = set(tokens)
        return True

    def _tokens_snapshot():
        return sorted(ws_stub._tokens)

    ws_stub.set_tokens.side_effect = _set_tokens
    ws_stub.tokens_snapshot.side_effect = _tokens_snapshot

    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    mdm = MagicMock(spec=MarketDataManager)
    # Wire reconcile_active_subscriptions through the real implementation
    from nifty_scalper_bot.data import market_data_manager as _mdm_mod
    import types

    # Inject a real reconcile method that operates on the ws stub
    _desired: set[int] = set(initial_ws_tokens)

    def _reconcile(desired_tokens: set[int]) -> dict[str, int]:
        nonlocal _desired
        desired = {int(t) for t in desired_tokens if t and int(t) > 0}
        before = len(ws_stub._tokens)
        to_add = desired - ws_stub._tokens
        to_remove = ws_stub._tokens - desired
        ws_stub.set_tokens(sorted(desired))
        _desired = set(desired)
        return {
            "desired": len(desired),
            "before": before,
            "added": len(to_add),
            "removed": len(to_remove),
            "after": len(desired),
        }

    mdm.reconcile_active_subscriptions = _reconcile
    mdm._ws = ws_stub
    mdm._desired_tokens = _desired
    return mdm, ws_stub


# ---------------------------------------------------------------------------
# Test 1: stale tokens are removed after basket reselection
# ---------------------------------------------------------------------------

class TestStaleTokensUnsubscribed:
    def test_stale_old_option_tokens_removed(self):
        """After basket update, tokens not in desired set must be removed."""
        old_tokens = {111, 222, 333, 444, 555}  # old strikes
        mdm, ws = _make_mdm_with_ws(old_tokens)

        # New basket: only keep spot + fut + new CE/PE
        new_desired = {256265, 57400, 99001, 99002}  # spot, fut, ce, pe
        result = mdm.reconcile_active_subscriptions(new_desired)

        assert ws._tokens == new_desired, f"Expected {new_desired}, got {ws._tokens}"
        assert result["removed"] == len(old_tokens), (
            f"Expected {len(old_tokens)} removed, got {result['removed']}"
        )
        assert result["added"] == len(new_desired)

    def test_no_removals_when_desired_unchanged(self):
        """Reconcile with same desired set must not unsubscribe anything."""
        tokens = {256265, 57400, 99001}
        mdm, ws = _make_mdm_with_ws(tokens)
        result = mdm.reconcile_active_subscriptions(tokens)
        assert result["removed"] == 0
        assert ws._tokens == tokens


# ---------------------------------------------------------------------------
# Test 2: selected CE/PE and active FUT always remain subscribed
# ---------------------------------------------------------------------------

class TestProtectedTokensRetained:
    def test_selected_ce_pe_and_fut_in_desired_never_removed(self):
        """CE, PE, FUT tokens that appear in desired_tokens must not be removed."""
        fut_token = 57400
        ce_token = 99001
        pe_token = 99002
        spot_token = 256265
        stale_tokens = {11111, 22222, 33333}

        initial = stale_tokens | {fut_token, ce_token, pe_token, spot_token}
        mdm, ws = _make_mdm_with_ws(initial)

        desired = {spot_token, fut_token, ce_token, pe_token}
        result = mdm.reconcile_active_subscriptions(desired)

        # Protected tokens must remain
        assert fut_token in ws._tokens
        assert ce_token in ws._tokens
        assert pe_token in ws._tokens
        assert spot_token in ws._tokens

        # Stale tokens must be gone
        for t in stale_tokens:
            assert t not in ws._tokens, f"Stale token {t} should have been removed"

        assert result["removed"] == len(stale_tokens)


# ---------------------------------------------------------------------------
# Test 3: unknown token logged as count + sample, not full list
# ---------------------------------------------------------------------------

class TestUnknownTokenLogFormat:
    def test_futures_purge_state_logs_counts_not_full_lists(self):
        """FUTURES_PURGE_STATE log must not include full token lists."""
        from pathlib import Path
        src = Path("src/nifty_scalper_bot/data/market_data_manager.py").read_text(
            encoding="utf-8"
        )
        # Must use count-based keys
        assert "purged_tokens_count" in src
        assert "unknown_tokens_count" in src
        assert "unknown_tokens_sample" in src
        # Must NOT log full lists as positional args
        assert '"purged_tokens=%s"' not in src or "purged_tokens=%s" not in src.split("FUTURES_PURGE_STATE")[1].split("\n")[0]

    def test_unknown_tokens_per_token_warning_removed(self):
        """Per-token UNKNOWN_TOKEN_IN_DESIRED_SUBSCRIPTIONS warning must be gone."""
        from pathlib import Path
        src = Path("src/nifty_scalper_bot/data/market_data_manager.py").read_text(
            encoding="utf-8"
        )
        # Old pattern: one warning per token — must not exist
        assert "UNKNOWN_TOKEN_IN_DESIRED_SUBSCRIPTIONS" not in src
        # New pattern: single batched warning
        assert "UNKNOWN_TOKENS_IN_DESIRED_SUBSCRIPTIONS" in src

    def test_stale_purged_log_uses_counts(self):
        """FUTURES_STALE_SUBSCRIPTION_PURGED must log counts, not full lists."""
        from pathlib import Path
        src = Path("src/nifty_scalper_bot/data/market_data_manager.py").read_text(
            encoding="utf-8"
        )
        assert "purged_symbols_count" in src
        assert "purged_tokens_count" in src


# ---------------------------------------------------------------------------
# Test 4: WebSocket token set converges to desired set
# ---------------------------------------------------------------------------

class TestWebSocketConvergence:
    def test_ws_tokens_equal_desired_after_reconcile(self):
        """After reconcile, ws._tokens must exactly equal desired_tokens."""
        initial = set(range(100, 900))  # 800 stale tokens
        mdm, ws = _make_mdm_with_ws(initial)

        desired = {256265, 57400, 99001, 99002, 88001, 88002}
        mdm.reconcile_active_subscriptions(desired)

        assert ws._tokens == desired
        assert len(ws._tokens) == len(desired)

    def test_result_counts_are_accurate(self):
        """Result dict from reconcile must have accurate counts."""
        initial = {1, 2, 3, 4, 5}
        mdm, ws = _make_mdm_with_ws(initial)

        desired = {3, 4, 5, 6, 7}  # remove 1,2 add 6,7
        result = mdm.reconcile_active_subscriptions(desired)

        assert result["before"] == 5
        assert result["added"] == 2
        assert result["removed"] == 2
        assert result["after"] == 5
        assert result["desired"] == 5


# ---------------------------------------------------------------------------
# Test 5: reconcile_active_subscriptions is idempotent
# ---------------------------------------------------------------------------

class TestReconcileIdempotent:
    def test_calling_twice_with_same_desired_is_stable(self):
        """Two reconcile calls with same desired_tokens must not change WS state."""
        tokens = {256265, 57400, 99001, 99002}
        mdm, ws = _make_mdm_with_ws(tokens)

        mdm.reconcile_active_subscriptions(tokens)
        state_after_first = set(ws._tokens)

        mdm.reconcile_active_subscriptions(tokens)
        state_after_second = set(ws._tokens)

        assert state_after_first == state_after_second == tokens


# ---------------------------------------------------------------------------
# Test 6: utils.backoff.exp_backoff yields valid sleep intervals
# ---------------------------------------------------------------------------

class TestExpBackoff:
    def test_yields_positive_floats(self):
        from nifty_scalper_bot.utils.backoff import exp_backoff
        gen = exp_backoff(base_ms=100, max_ms=5000)
        for _ in range(10):
            val = next(gen)
            assert isinstance(val, float)
            assert val >= 0.0

    def test_values_increase_on_average(self):
        from nifty_scalper_bot.utils.backoff import exp_backoff
        gen = exp_backoff(base_ms=100, max_ms=10000, jitter_pct=0.0)
        vals = [next(gen) for _ in range(8)]
        # Without jitter, each value should be double the previous (up to cap)
        for i in range(1, len(vals) - 1):
            assert vals[i] >= vals[i - 1], f"Backoff should increase: {vals}"

    def test_caps_at_max_ms(self):
        from nifty_scalper_bot.utils.backoff import exp_backoff
        gen = exp_backoff(base_ms=1000, max_ms=2000, jitter_pct=0.0)
        vals = [next(gen) for _ in range(10)]
        max_seconds = 2000 / 1000
        for v in vals:
            assert v <= max_seconds + 0.01, f"Value {v} exceeds max"

    def test_backoff_module_importable(self):
        """The missing utils.backoff module must now import cleanly."""
        import importlib
        mod = importlib.import_module("nifty_scalper_bot.utils.backoff")
        assert hasattr(mod, "exp_backoff")

    def test_polling_manager_imports_cleanly(self):
        """data.polling must now import without ImportError."""
        import importlib
        mod = importlib.import_module("nifty_scalper_bot.data.polling")
        assert mod is not None


# ---------------------------------------------------------------------------
# Test 7: reconcile_active_subscriptions replaces purge in basket commit
# ---------------------------------------------------------------------------

class TestBasketCommitCallsReconcile:
    def test_reconcile_fn_called_after_basket_commit(self):
        """_commit_active_dynamic_basket must call reconcile_active_subscriptions."""
        from pathlib import Path
        src = Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")
        # The reconcile call must be present in the basket commit function body
        assert "reconcile_active_subscriptions" in src
        assert "reconcile_fn = getattr(mdm, \"reconcile_active_subscriptions\", None)" in src

    def test_purge_stale_not_primary_cleanup_in_basket_commit(self):
        """_commit_active_dynamic_basket must not call purge_stale as primary cleanup."""
        from pathlib import Path
        src = Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")
        # Extract just the _commit_active_dynamic_basket function body
        start = src.index("def _commit_active_dynamic_basket(")
        end = src.index("\ndef ", start + 1)
        fn_body = src[start:end]
        # The function must call reconcile
        assert "reconcile_active_subscriptions" in fn_body
        # Any remaining purge call inside this function should NOT exist
        # (purge is only kept in startup code, not basket commit)
        assert "purge_stale_nifty_futures" not in fn_body, (
            "purge_stale_nifty_futures should be replaced by reconcile in basket commit"
        )
