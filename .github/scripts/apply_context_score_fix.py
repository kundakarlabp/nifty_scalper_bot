from pathlib import Path

strategy_path = Path("src/nifty_scalper_bot/core/strategy_manager.py")
text = strategy_path.read_text()

old_helper = '''    def _extract_context_score(self, vote: StrategyVote) -> float:
        """Args: vote. Returns: context score. Raises: none."""
        payload = dict(vote.metadata or {})
        for key in ("context_bonus_score", "context_score", "raw_setup_score", "raw_vote_score", "vote_score"):
            try:
                if payload.get(key) is not None:
                    return max(0.0, float(payload.get(key)))
            except (TypeError, ValueError):
                continue
        return max(0.0, self._extract_raw_score(vote))
'''
new_helper = '''    def _extract_raw_context_score(self, vote: StrategyVote) -> float:
        """Return context evidence before regime weighting."""
        payload = dict(vote.metadata or {})
        for key in (
            "context_bonus_score",
            "context_score",
            "raw_setup_score",
            "raw_vote_score",
            "vote_score",
        ):
            try:
                if payload.get(key) is not None:
                    return max(0.0, float(payload.get(key)))
            except (TypeError, ValueError):
                continue
        return max(0.0, self._extract_raw_score(vote))

    def _extract_context_score(self, vote: StrategyVote) -> float:
        """Return positive context evidence after exactly one regime weighting."""
        payload = dict(vote.metadata or {})
        explicit_weighted = payload.get("regime_weighted_context_score")
        try:
            if explicit_weighted is not None:
                return max(0.0, float(explicit_weighted))
        except (TypeError, ValueError):
            pass
        raw_context_score = self._extract_raw_context_score(vote)
        try:
            regime_weight = float(payload.get("regime_weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            regime_weight = 1.0
        return max(0.0, raw_context_score * max(0.0, regime_weight))
'''
if old_helper not in text:
    raise SystemExit("context score helper patch needle not found")
text = text.replace(old_helper, new_helper, 1)

old_score_block = '''            confirmed_positive_context = sum(
                self._extract_context_score(vote)
                for vote in qualifying_context_votes
            )
            confirmed_context_bonus = min(
                1.5, 0.45 * confirmed_positive_context
            )
            context_confirmed_final_score = max(
                0.0,
                min(
                    10.0,
                    weighted_trigger_score
                    + confirmed_context_bonus
                    - context_penalty,
                ),
            )
            context_confirmed_single_allowed = bool(
                mode_profile.get("allow_single_vote", True)
                and threshold_passed
                and selected_option
                and qualifying_context_votes
                and context_confirmed_final_score >= selected_single_min
            )
'''
new_score_block = '''            confirmed_raw_context_score = sum(
                self._extract_raw_context_score(vote)
                for vote in qualifying_context_votes
            )
            confirmed_positive_context = sum(
                self._extract_context_score(vote)
                for vote in qualifying_context_votes
            )
            confirmed_context_bonus = min(
                1.5, 0.45 * confirmed_positive_context
            )
            context_confirmed_final_score = max(
                0.0,
                min(
                    10.0,
                    weighted_trigger_score
                    + confirmed_context_bonus
                    - context_penalty,
                ),
            )
            context_confirmed_single_allowed = bool(
                mode_profile.get("allow_single_vote", True)
                and threshold_passed
                and selected_option
                and qualifying_context_votes
                and context_confirmed_final_score >= selected_single_min
            )
            if qualifying_context_votes:
                log.info(
                    "SINGLE_TRIGGER_CONTEXT_SCORE symbol=%s trigger_strategy=%s "
                    "weighted_trigger_score=%.3f raw_context_score=%.3f "
                    "regime_weighted_context_score=%.3f context_bonus=%.3f "
                    "context_penalty=%.3f final_score=%.3f final_min=%.3f allowed=%s",
                    symbol_norm,
                    best_vote.strategy,
                    weighted_trigger_score,
                    confirmed_raw_context_score,
                    confirmed_positive_context,
                    confirmed_context_bonus,
                    context_penalty,
                    context_confirmed_final_score,
                    selected_single_min,
                    context_confirmed_single_allowed,
                    extra={
                        "event": "SINGLE_TRIGGER_CONTEXT_SCORE",
                        "symbol": symbol_norm,
                        "trigger_strategy": best_vote.strategy,
                        "weighted_trigger_score": weighted_trigger_score,
                        "raw_context_score": confirmed_raw_context_score,
                        "regime_weighted_context_score": confirmed_positive_context,
                        "context_bonus": confirmed_context_bonus,
                        "context_penalty": context_penalty,
                        "final_score": context_confirmed_final_score,
                        "final_min": selected_single_min,
                        "allowed": context_confirmed_single_allowed,
                        "qualifying_context_strategies": [
                            vote.strategy for vote in qualifying_context_votes
                        ],
                        "qualifying_context_regime_weights": [
                            float((vote.metadata or {}).get("regime_weight", 1.0) or 1.0)
                            for vote in qualifying_context_votes
                        ],
                    },
                )
'''
if old_score_block not in text:
    raise SystemExit("context score equation patch needle not found")
text = text.replace(old_score_block, new_score_block, 1)

old_metadata = '''                metadata["context_confirmation_final_score"] = round(
                    context_confirmed_final_score, 3
                )
'''
new_metadata = '''                metadata["context_confirmation_final_score"] = round(
                    context_confirmed_final_score, 3
                )
                metadata["context_confirmation_raw_score"] = round(
                    confirmed_raw_context_score, 3
                )
                metadata["context_confirmation_regime_weighted_score"] = round(
                    confirmed_positive_context, 3
                )
                metadata["context_confirmation_score_min"] = round(
                    selected_single_min, 3
                )
'''
if old_metadata not in text:
    raise SystemExit("context confirmation metadata patch needle not found")
text = text.replace(old_metadata, new_metadata, 1)
strategy_path.write_text(text)

test_path = Path("tests/strategies/test_single_vote_score_floor.py")
test_text = test_path.read_text()
regressions = '''\n\nasync def test_regime_downweighted_context_cannot_unlock_single_trigger(\n    monkeypatch,\n) -> None:\n    monkeypatch.setenv("EXECUTION_MODE", "LIVE")\n    monkeypatch.setenv("ENABLE_LIVE", "true")\n    monkeypatch.delenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", raising=False)\n    monkeypatch.delenv("STRATEGY_ALLOW_SELECTED_OPTION_SINGLE_VOTE", raising=False)\n    manager = _manager_probe()\n    trigger = _signal_vote(strategy="SMC", raw_score=8.6, weighted_score=8.6)\n    trigger[0].metadata.update({"strategy": "SMC", "is_selected_option": True})\n    context_signal, context_vote = _context_vote(score=10.0, confidence=0.85)\n    context_vote.score = 2.5\n    context_vote.metadata["regime_weight"] = 0.25\n    context_vote.metadata["regime_weighted_vote_score"] = 2.5\n\n    result = manager._combine_strategy_votes(\n        symbol="NFO:NIFTY2670724050CE",\n        signals=[trigger, (context_signal, context_vote)],\n        indicators=_valid_entry_context(),\n    )\n\n    assert result is None\n    decision = manager._last_no_signal_decision_by_symbol[\n        "NFO:NIFTY2670724050CE"\n    ]\n    assert decision.reason == "single_trigger_context_score_below_min"\n\n\nasync def test_context_confirmation_applies_regime_weight_exactly_once(\n    monkeypatch,\n) -> None:\n    monkeypatch.setenv("EXECUTION_MODE", "LIVE")\n    monkeypatch.setenv("ENABLE_LIVE", "true")\n    monkeypatch.delenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", raising=False)\n    monkeypatch.delenv("STRATEGY_ALLOW_SELECTED_OPTION_SINGLE_VOTE", raising=False)\n    manager = _manager_probe()\n    trigger = _signal_vote(strategy="SMC", raw_score=8.8, weighted_score=8.8)\n    trigger[0].metadata.update({"strategy": "SMC", "is_selected_option": True})\n    context_signal, context_vote = _context_vote(score=10.0, confidence=0.85)\n    context_vote.score = 5.0\n    context_vote.metadata["regime_weight"] = 0.5\n    context_vote.metadata["regime_weighted_vote_score"] = 5.0\n\n    result = manager._combine_strategy_votes(\n        symbol="NFO:NIFTY2670724050CE",\n        signals=[trigger, (context_signal, context_vote)],\n        indicators=_valid_entry_context(),\n    )\n\n    assert result is not None\n    assert result.metadata["approval_path"] == "single_trigger_context_confirmed"\n    assert result.metadata["context_confirmation_raw_score"] == 3.0\n    assert result.metadata["context_confirmation_regime_weighted_score"] == 1.5\n    assert result.metadata["context_confirmation_score_min"] == 9.0\n    assert result.metadata["context_bonus"] == 0.675\n    assert result.metadata["final_trade_score"] == 9.475\n'''
if "test_regime_downweighted_context_cannot_unlock_single_trigger" in test_text:
    raise SystemExit("context weighting regressions already exist")
test_path.write_text(test_text.rstrip() + regressions + "\n")
