# 20 November 2026 SMC losing trade: adaptive trailing and market data audit

## Scope, observed incident and evidence boundary

NFO:NIFTY26NOV25700CE, 65 units, bought at ₹145.35; initial broker-resident
stop at ₹138.85, exited by broker protective stop at ₹138.85, net -₹491
including ₹68.50 broker cost evidence. The SMC signal's sweep/reclaim and
sustained order-flow route are not changed by this work.

The research quote archive contains exactly 286 cached best-bid/ask samples,
with maximum sampled bid ₹150.30, i.e. (150.30 - 145.35) / (145.35 - 138.85)
= 0.7615R. At least 12 sampled bids were at or above the previous 0.60R
threshold ₹149.25. The original broker stop was never modified; trailing
revision remained zero. This is *not proof* that 65 units were executable
at the sampled top bid: depth quantity, source exchange timestamps and broker
stop-modification acknowledgment times were NOT archived. A counterfactual
₹147 exit is explicitly hypothetical.

## Reconstructed live path and code audit

1. **WebSocket KiteTicker / callback**: transport supplies tick and optional
   market depth. Timezone-free feed datetimes can originate from process-local
   `datetime.fromtimestamp`; host environment is not currently verified.
   Reject unparseable and future event times. Do not synthesize current event
   time to explain an old quote.
2. **MDM normalize/ingest** `data/market_data_manager.py`: creates canonical
   symbol, bid/ask, depth and source-verified aware timestamp. Best bid and
   ordinary quote readiness do not prove top-level visible quantity >=65.
   Capture timestamps/versions and depth provenance separately.
3. **DataHub facade** `data/data_hub.py`: execution's
   `set_active_bracket_symbols` delegate was missing, so bracket-owned
   symbols did not reliably reach MDM's `_active_bracket_symbols`.
   Fixed via minimal delegation, no second registry.
4. **Basket/subscription owner** `MarketDataManager.reconcile_active_subscriptions`:
   previously replaced desired tokens with exactly the new active basket and
   could remove a held 25700CE contract on a shift to 25750CE. At
   13:40:38 IST the live log reported old 25700CE/PE -> new 25750CE/PE
   and two removed tokens, eight seconds after 25700CE's 13:40:30 fill.
   Fixed by unioning active bracket/open-position tokens before removal.
   Broker-held positions must remain required until confirmed closure.
   The original WS token list was not archived, so a particular
   unsubscribe is still a strong hypothesis rather than direct proof.
5. **Bracket on-tick owner** `execution/ownership.py`: records exact normalized
   protection-tick bid/ask into canonical bracket's quote cache before SL/TP.
   Original virtual stop / broker stop remain hard protection irrespective of
   quote/strategy and regardless of optional trail eligibility.
6. **Adaptive full-size executable proof** `_adaptive_executable_bid`:
   requires fresh event source, depth.buy[0] valid price+visible size >=held lot,
   ask, non-crossed book, acceptable spread and top-level consistency.
   The incident archive alone cannot pass this validator.
7. **Adaptive cost/activation/decision** `tighten_adaptive_profit_lock`:
   validates broker stop identity and recorded full quantity, positive modeled
   all-in fees, novel quote sample identity and 0.6R MFE from certified full
   bids; continuation uses independent premium/underlying/completed futures
   evidence. STRONG/NORMAL/WEAK govern one giveback allowance; not three
   independent tightly combined stops. Expected states need distinct
   assessments; no repeated cached-quote votes.
8. **Timestamp source correctness** `_quote_age_seconds`:
   prior raw exchange_timestamp-first parsing treated timezone-free market
   times as UTC; MDM already supplies a validated aware canonical source
   timestamp. Fix prefers that canonical timestamp only when
   `source_timestamp_valid == True` and `timestamp_source` identifies
   an exchange event; stale/future canonical sources continue to fail closed.
   The host's effective timezone remains to be independently verified.
9. **Optional policy owner** `BoundBracketManager._apply_trailing_math`:
   previous LIVE adaptive no-op suppressed the legacy trailing entirely.
   Adaptive remains first. Only on a NO-RATCHET may an executable, full-lot
   verified instance of the existing tier schedule propose protection.
   Paper/replay legacy remains separate. No LTP-only LIVE ratchet.
10. **Fallback tier and cost floor** `tighten_verified_tier_fallback`:
   requires same-tick bid cache agreement, fresh complete depth, fees and
   exchange-stop coverage; checks initial-risk based tier activation,
   canonical tier candidate, full-cost+slippage net floor, spread room
   and monotonic improvement before invoking `_ratchet_stop`.
   No fee proof -> no optional profit ratchet; hard stop still exists.
11. **Shared validator and broker sync** `_ratchet_stop` /
   `_sync_exchange_protective_stop`: previously suppressed exceptions from
   `_is_trail_candidate_allowed`; now fails closed. One authority
   coalesces exchange modifications, enforces protected quantity,
   monotonic broker stop and order modification budgets. A virtual stop
   revision is **not** proof of broker modification or executed exit.
12. **Observability**: bounded `ADAPTIVE_PROFIT_LOCK_SKIP` includes
   stage/sub-reason, top-of-book quantity and required size, event age,
   receive age, quote source/version and broker stop. Legacy ratchet has a
   distinct event. Logs do not trigger broker calls.

## Regression and replay matrix

- Real 286-sample incident replay **without invented depth**: must never
  authorize a full-lot LIVE ratchet based on best-bid-only records.
- Broker-free scenario with 65+ visible best-bid quantity, fresh age, low
  spread, verified stop: fallback may ratchet at its cost-adjusted tier.
- Fail closed for missing/undersized depth, stale/future event timestamp,
  invalid book, excessive spread, mismatched bracket quote cache, missing or
  undersized exchange-stop proof, degraded stop, missing fee proof.
- Canonical aware exchange timestamp from MDM must beat ambiguous raw naive
  exchange datetime; do NOT allow fresh receive time to override stale
  exchange evidence.
- One optional stop owner, one broker sync, no fallback if adaptive already
  ratcheted, no validation-exception bypass, monotonic SL.
- DataHub bridge + MDM basket shift 25700->25750 preserves open 25700 token;
  owner release permits later unsubscribe.
- Ordinary idle ticks avoid a second costly fallback depth evaluation until
  a best-bid/certified MFE reaches activation.

## Not demonstrated and not promised

The 20 November records cannot establish full depth, actual counterfactual
broker modification acceptance/fill, or positive net expectancy of the fallback.
No stop, even a broker-resident SL-limit, guarantees fill in a gap or fast
book. No risk policy can prevent all losses. Avoid post-hoc SMC threshold
changes from one sample. Run the full CI shards, broker-free end-to-end
simulation, and verify LIVE broker flat/open positions, stop state and active
subscriptions before activating any release. Keep ₹773 daily loss gate,
normal risk sizing, no parallel ordered-tick processing, and no bypasses.
