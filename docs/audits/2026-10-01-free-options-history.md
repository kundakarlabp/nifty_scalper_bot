# Free options history: coverage and engineering correction

The existing NSE collector requested a format discontinued on 8 July 2024.
It also recognized only legacy ZIP members and overwrote earlier collected
days on each incremental run. These were reproducible engineering defects,
not evidence that a strategy should trade more frequently.

## Correction

`scripts/sync_nifty_options.py` now selects UDiFF for dates from 8 July 2024,
normalizes its fields into the existing CSV schema, retains older history,
and replaces duplicate contract-day rows with the latest fetched version.
ZIP CRC validation prevents truncated caches or downloads being accepted;
downloads are replaced atomically. Terminal HTTP failures are not repeatedly
retried at the same URL. `--end-date` permits reproducible historical ranges.
The coverage JSON distinguishes unavailable dates and unparsed archives and
explicitly labels the data as daily, without historical bid/ask or depth.

FO volume remains contracts; OI remains exchange quantity. Turnover is divided
by 100,000 to retain the legacy lakh-rupee column. Option turnover must not be
interpreted as premium-only turnover. Units were checked against the actual
September 30, 2026 archive and the exchange's UDiFF specification.

Test-first regressions cover URL migration, contract/daily-price normalization,
corrupt-cache replacement, and idempotent incremental retention. The legacy
expiry-calendar tests remain applicable. Runtime strategy, risk, and execution
paths are unchanged.

## Reproducible collection

```bash
python scripts/sync_nifty_options.py --days 365 --end-date 2026-09-30 \
  --outdir data/nse_options_eod --verbose
```

Read `nifty_options_coverage.json` before analysis. Unavailable calendar dates
include weekends/holidays and cannot all be classified as missing trading days.
An existing combined CSV is retained unless `--force-rebuild` is explicitly
specified. The combined file contains multiple option contracts per day:
partition by expiry, strike and CE/PE before any contract-level analysis.
Do not feed it into an intraday replay as underlying or executable tick data.

## Provider evidence, checked 1 October 2026

| Source | Useful coverage | Material boundary |
|---|---|---|
| Existing Kite subscription | Active-contract minute OHLC, volume, requested OI | Expired option history unavailable; no historical bid/ask depth |
| Free NSE bhavcopies | Actual contract dates, expiry, strike, CE/PE, daily OHLC, volume/OI | EOD only; no entry/exit ordering or intraday execution validation |
| ICICI Direct Breeze | Provider advertises free customer API and three years of F&O history; documented expired-contract minute request | Requires that broker account/session; actual contract coverage needs a download audit; no verified unlimited historical depth |
| Upstox expired API | Expired option OHLC endpoints | Requires Plus entitlement/account; cannot assume the user's Kite entitlement grants access |
| NSE order/trade historical product | Institutional historical order/trade data | Subscription/SFTP product; not an unrestricted free substitute |

Primary sources:

- NSE migration/current reports: https://www.nseindia.com/all-reports-derivatives
- NSE UDiFF specification: https://www.nseindia.com/static/resources/forms-formats-members
- Kite historical API: https://kite.trade/docs/connect/v3/historical/
- Kite support on included history and active options: https://kite.trade/forum/discussion/14806/historical-data-is-now-free-with-base-kite-connect-subscription
- Breeze free customer API/coverage: https://www.icicidirect.com/futures-and-options/api/breeze
- Breeze historical request example: https://www.icicidirect.com/futures-and-options/api/breeze/article/how-to-download-historical-data-using-breezeapi-python-sdk
- Upstox expired API entitlement: https://upstox.com/developer/api-documentation/get-expired-historical-candle-data/
- NSE historical subscriptions: https://www.nseindia.com/static/market-data/eod-historical-data-subscription

## Optimization implications

Use daily archives for contract-universe and data-quality research, not claims
that ORB/VWAP/SMC/squeeze entries or brackets improve net intraday expectancy.
Those need synchronized spot/futures/options minute history and chronological
holdouts; depth-sensitive OFI and fills additionally need actual quote/depth
captures and order/fill journals. Capture active Kite contracts before expiry,
including contemporaneous instrument mappings, rather than relying on tokens
remaining valid after expiry. Keep missing depth explicitly absent.

No authenticated Kite credentials were available in this workspace. The
existing read-only runtime diagnostic relay timed out; that does not prove
the production bot is offline. No account was opened, subscription purchased,
live order placed, or live restart performed. This patch restores an existing
free collection path; it does not establish profitability or unlimited data.
