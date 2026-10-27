# NIFTY Scalper — 27 Oct 2026 closed-session VWAP/SMC diagnostic

**Scope:** Read-only inspection of the finalized 27 October 2026 Supabase archive (`nifty_daily_market_logs`), trade ledger, trade events and runtime heartbeat. No change to signal admission, live risk, broker orders or bracket exits.

## Production authority and effective configuration

- The archived startup line at **14:11:33 IST** identifies `VWAP_EARLY_TREND_CONTEXT_GATE_CONFIG min_conf=0.80 max_age_s=2.00 trend_momentum_enabled=False live_neutral_slope_enabled=True`. Production fingerprint `8db06c8fbe47` appeared at 14:11:32 IST.
- The immediately preceding fingerprint `9989ee219611` at 13:54:21 IST showed `min_conf=0.90`. This establishes **the actual runtime setting changed**, not only the GitHub default.
- Most recent post-market heartbeat was engine healthy, build `8db06c8fbe47`, `release_state=current`, `trading_status=blocked`, `blockers=[market_closed]`. The market-close blocker is expected outside the entry window.
- The finalized archive spans **09:15–15:30 IST**, with **16,788 log lines**. The public trade ledger and event journal show **zero trades on this date**.

## Post-0.80 market decision funnel (14:11:33–15:30 IST)

After the new setting, **31 non-overlapping periodic strategy-status reports** recorded the following VWAPPro no-vote evaluations. These are *evaluation counts*, not distinct missed profitable trades. The mode of recording is repeated symbol/context evaluation; some individual setups can appear repeatedly.

| VWAPPro rejection | No-vote evaluations |
| --- | ---: |
| `option_candle_no_trades` | 135 |
| `vwap_thesis_reset` | 101 |
| `distance_outside_band` | 56 |
| `trend_momentum_premium_not_confirmed` | 36 |
| `trend_momentum_weak_close` | 10 |
| `futures_slope_not_aligned` | 10 |
| `trend_momentum_bar_unconfirmed` | 4 |
| `trend_momentum_option_volume_not_confirmed` | 3 |
| `trend_retest_structure_unconfirmed` | 1 |

**Zero** sampled post-release VWAP status reports contained the old `trend_momentum_underlying_quality_unconfirmed` rejection. This confirms the gate changed behavior, **not** that 0.80 improved after-cost strategy results. A recent pre-0.80 interval also had other first-failing gates, so absence of that reason alone cannot be interpreted as full counterfactual profitability.

The SMC post-release no-vote counts were: missing underlying sweep 200, awaiting confirmation 48, wrong underlying side 46, shallow sweep 35, reclaim failure 11, duplicate confirmation bar 10, direction not ready 2, sweep expired 1, and confirmed vote expired 1. The ORB window had expired; OrderFlow remained confirmation-only.

## Two SMC votes were one setup

Both post-release SMC trigger votes at **14:15:04** and **14:17:00 IST** referred to the **same** `smcv2:NFO:NIFTY26OCTFUT:PE:2026-10-27T08:43:00+00:00` underlying setup. On contract rotation to 22650PE at 14:15, one aligned eligible temporal-OFI observation was present, but no sustained independent proof or confirming VWAP vote; runner rejected `smc_independent_confirmation_missing`. At 14:17, OFI became adverse for that PE setup (and VWAP had no confirmed finalized premium bar); runner rejected `fresh_opposing_context`. These are **not** two distinct missed entries and do not justify relaxing the adverse-flow guard.

## Next investigation and safe code change

The top VWAP reason (`option_candle_no_trades`) can represent either a genuine untraded premium candle or a missing/zeroed trade-volume field. The archived log lacks the contemporaneous bar/volume provenance needed to distinguish them, and executable rejected-candidate bid/ask history is insufficient for an economic replay. **Do not treat current volume=0 as positive confirmation, borrow futures volume to manufacture option trades, or force an entry.**

One small diagnostics-only correction adds rate-limited `VWAP_OPTION_VOLUME_REJECT` evidence in the existing `StrategyManager` *after* the VWAP rejection, at most once per option symbol per **60 seconds**. The event exposes raw option volume (distinguishes `None` from `0`), average volume, last bar anchor/completion/provisional flags, quote version, data age and history source when those fields are available. The original `VWAPPro` gate `if vol <= 0: self._no_vote("option_candle_no_trades"); return None` remains intact. This is **diagnostic only**, not a fix for a proven data-feed fault.

## Unverified

No complete timestamp-aligned rejected-candidate option bid/ask/depth series, fill/slippage outcomes, or synchronized bar data were available for a 0.90-versus-0.80 economic replay. Root cause of individual zero-volume bars remains **unproven**. Neither a profit benefit nor a technical ingestion defect can currently be asserted. Avoid further parameter relaxation without better data.
