# ICICI Breeze two-year historical data audit — 9 November 2026

## Scope and provenance
- Historical acquisition window: 2024-11-06 to 2026-11-05 inclusive.
- Original workflow run: https://github.com/kundakarlabp/nifty_scalper_bot/actions/runs/42588287357
- 25 encrypted read-only research archives; each archived SHA256SUMS and outer CMS checksum verified. Source archives are immutable.
- All 25 encrypted ZIPs are backed up to the user's existing private Drive folder. Exact filenames and byte sizes match GitHub's records (107,821,151 bytes total; zero duplicates). Do not put private broker rows or keys in the repository.

## Verified observations
| Component | 1-minute bars | Contract-days |
| --- | ---: | ---: |
| NIFTY index | 184,786 | 522 attempted, 493 nonempty |
| NIFTY dated futures | 536,695 | 1,472 |
| Selected NIFTY CE/PE options | 733,386 | 1,960 |
| **Total** | **1,454,867** | **3,954** |

- 6/25 windows pass the original collector's strict completion flag; 19/25 are flagged partial.
- Independent local audit checked 4,491 archived files against SHA256SUMS: no mismatches.
- All normalized candle rows passed timestamp parse, IST offset, ordering/uniqueness, session-envelope and OHLC range checks. NIFTY cash-index OI is legitimately null.
- All 29 `source_unavailable` weekdays match **official NSE exchange trading holidays**, including election-related closures 2024-11-20 and 2026-01-15. Do **not** backfill these weekdays as normal trading sessions.
- 63 empty contract-days comprise 29 holiday index requests and 34 futures requests; the futures empties require no-trade/provider/mapping differentiation.
- 19 contract requests failed provider or validation checks. 113 nonempty contract-days contain missing minute timestamps under the 375-minute default calendar. Of those, eight are the special 2025-10-21 Muhurat session; the normal 375-minute denominator must not be used for these.
- 2025-10-21 F&O bars cover 13:45–14:45 IST; this matches the NSE one-hour Muhurat session. The separate index source returned 370 regular-day timestamps and must be independently reconciled before using that day's spot-price series.
- The special **Saturday 2025-02-01 Budget session (09:15–15:30)** is not in the original Breeze intraday enumeration. The existing EOD special-session addendum is not an intraday substitute.
- Source timezone/UTC conversion and bar-start convention have not been independently corroborated; importer labels remain research-candidate only.
- Options universe was just the opening-nearest CE and PE in two expiries; not full historical chains, intraday point-in-time eligibility, historical bid/ask/depth, Greeks, IV or executable quote snapshots.

## Bounded remediation
- 158 targeted retry contract-days: 19 provider/validation failures, 34 empty futures contracts, and 105 nonempty missing-minute contracts (special 2025-10-21 excluded).
- Recovery script `scripts/archive_breeze_backfill.py` and trigger CSV `deploy/breeze_targeted_backfill_20261109.csv`; original archived observations never overwritten. All re-acquired private rows encrypted; failed retries retained as explicit deficiencies.
- Repair run: https://github.com/kundakarlabp/nifty_scalper_bot/actions/runs/42635897210
- These 158 requests cannot create the uncollected historical option chains or bid/ask data. A recovered candle series does not prove real-world fill price or available historical contract.
- Follow-up work: decrypt and compare backfill hashes with originals, select better observations without silent overwrites, validate actual trade calendar and listing/expiry date, quantify irrecoverable gaps and special-session coverage, then run chronological out-of-sample research using realistic costs and appropriately labelled evidence levels.

## Official calendar references
- 2024 standard F&O holidays: https://archives.nseindia.com/content/circulars/FAOP59723.pdf
- 2024-11-20 election closure: https://nsearchives.nseindia.com/content/circulars/FAOP64959.pdf
- 2025 F&O holidays/Muhurat: https://nsearchives.nseindia.com/content/circulars/FAOP65588.pdf
- 2025-02-01 special F&O session: https://nsearchives.nseindia.com/content/circulars/FAOP65730.pdf
- 2026 F&O holidays: https://nsearchives.nseindia.com/content/circulars/FAOP71777.pdf
- 2026-01-15 additional F&O election closure: https://nsearchives.nseindia.com/content/circulars/FAOP72262.pdf
- 2025-10-21 Muhurat timing NSE/FAOP/70320: https://noticeblue.com/circulars/fbb8e152-c59b-40ef-bd5b-1e115ef4f759

This is a research audit, not evidence of historical profitable fills or backtest readiness.
