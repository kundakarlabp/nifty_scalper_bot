# Active Kite history archive

`scripts/archive_kite_history.py` is a read-only offline exporter using the
existing `ZerodhaKiteClient.instruments()` and `historical_data()` interfaces.
It adds no broker adapter, execution route, runtime history owner, or strategy
contract selector. Existing broker rate limiting, authentication and retries
remain authoritative.

Run in the bot's configured environment, using its existing `.env`/session:

```bash
python scripts/archive_kite_history.py --start 2026-09-01 --end 2026-09-30 \
  --symbols 'NSE:NIFTY 50' 'NFO:<EXACT_CURRENT_CE_TRADINGSYMBOL>' \
  'NFO:<EXACT_CURRENT_PE_TRADINGSYMBOL>' --outdir data/kite_history
```

Replace placeholders with current broker instrument symbols from the bot's
existing selected basket. A current NIFTY future may also be specified for
context. The exporter validates exact master identity before history requests;
it does not infer ATM strikes or change the live basket. NSE/NFO NIFTY data
only; no orders are placed. Do not put API credentials in command arguments.

For the existing Lightsail installation, pass
`--env-file /home/ubuntu/.config/niftybot/niftybot.env` to reuse the operator
configuration outside the checkout. An explicitly missing file fails before
broker initialization. Existing environment values take precedence; omitting
the option preserves normal dotenv discovery. No credential values are printed.

Requests cover at most 30 completed calendar days each. `oi=True` and
`continuous=False` preserve option history semantics. Each atomic JSON archive
contains the raw candle response, requested IST bounds, symbol/token, exact
instrument mapping, acquisition time, and explicit bar-start timestamp
convention. Historical bid/ask/depth are not fabricated.

Matching nonempty archives are reused. Corrupt archives are refetched;
empty and failed requests remain explicit in `coverage.json`, exit status is
nonzero for them, and later runs retry them. Upstream exception text is omitted
from persisted reports to avoid leaking authentication/request details.
Coverage is request-level: a nonempty response does not prove every expected
market minute is present. Check session calendars, listed-contract lifetimes,
missing minutes, OI availability and cross-instrument synchronization before
strategy optimization. Raw JSON is archival input, not a ready-made FULL-tick
replay dataset; use the existing market-data normalization/research path for
analysis and make completed bars available at bar end, not bar start.

Capture relevant active contracts regularly before expiry. Kite does not
provide expired option candles and instrument tokens must not be treated as
permanent historical contract IDs. The per-request instrument mapping remains
inside each archive even if the current master changes later.

## Validation and actual coverage

Test-first public seam regressions verify chunk bounds, IST timestamps,
raw OI and contract identity, resume behavior, corrupt-cache recovery,
unavailable-contract preflight, and explicit empty/failed retry behavior.
No real Kite download was performed in this workspace because no configured
broker credentials/session were accessible. This limitation must remain
visible; mock tests establish orchestration behavior, not broker entitlement
or strategy profitability.

The separate real NSE collection completed at zero provider cost:

- 246 archives/dates from 1 October 2025 through 30 September 2026.
- 420,960 unique contract-day rows; zero missing contract identity fields.
- 249,106 traded rows with valid OHLC; 171,854 non-traded zero-OHLC rows.
- Non-traded daily rows are unsuitable for executable entry/exit assumptions.
- 67 unavailable attempted dates include Saturdays/holidays; no downloaded
  archive failed parsing. This is not proof of complete intraday coverage.

The downloadable ZIP includes the full combined daily CSV, coverage JSON,
audit summary, and data-source documentation. Runtime strategy, stops,
position sizing and risk/execution safeguards are unchanged. Profitability
optimization remains dependent on real synchronized intraday history,
cost-aware chronological comparisons and execution evidence.
