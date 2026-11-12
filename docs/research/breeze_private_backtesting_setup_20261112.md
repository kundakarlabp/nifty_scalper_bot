# Private Breeze -> queryable NIFTY backtest dataset (12 November 2026)

Scope: **offline OHLCV research only**, not execution-grade options fills. Keep production and original private backups untouched. This guide accompanies draft PR #2024.

## Source and safety
- The 25 original encrypted Breeze ZIPs already reside in the private Google Drive folder 1ZEt9KR5a0gHL_q6rUhSe9v98EszNMgWx, totaling 107,821,151 bytes. Never re-upload, overwrite, or commit them.
- Original 2024-11-06 to 2026-11-05 data: 1,454,867 one-minute candles: index 184,786; dated futures 536,695; selected CE/PE options 733,386. Repair run 42635897210 recovered zero new bars (139 identical responses; 19 persistent failures).
- A usable research dataset does **not** mean full historical option-chain, executable quotes, correct broker token/lot mappings, historical bid/ask depth or validated fills. All imported observations must remain RESEARCH_CANDIDATE / BAR_PROXY_ONLY, promotion_eligible=false.

## Offline private setup

Use a trusted local Python 3.11+ checkout and OpenSSL, with an encrypted private filesystem. Download the original 25 archives once into /private/breeze/source-zips; keep the existing owner's private decryption key on that workstation only. The public matching cert is deploy/breeze_transport_recipient.pem. Do NOT send the key to chat, GitHub, or Drive.

Install project locally: python -m pip install -e '.[dev]'

1. Unseal and verify the immutable transport and archive checksums:

~~~sh
umask 077
python -m scripts.unseal_breeze_private_windows \
 --encrypted-windows /private/breeze/source-zips \
 --private-key /private/keys/breeze_research_private.pem \
 --recipient-cert deploy/breeze_transport_recipient.pem \
 --output /private/breeze/extracted \
 --expected-windows 25
~~~

2. Build a separate private SQLite research catalogue with hard-fail digest/identity checks:

~~~sh
python -m scripts.prepare_breeze_research_dataset build \
 --extracted-windows /private/breeze/extracted \
 --database /private/breeze/research/nifty_candles.sqlite \
 --summary /private/breeze/research/coverage_summary.json \
 --expected-windows 25
~~~

3. Query available contracts, then export 1m/5m/15m completed causal bars:

~~~python
import sqlite3
with sqlite3.connect("file:/private/breeze/research/nifty_candles.sqlite?mode=ro", uri=True) as db:
    for item in db.execute(
        "SELECT trade_day,contract_key,status,observed_minutes,missing_minutes "
        "FROM series ORDER BY trade_day,contract_key LIMIT 30"
    ):
        print(item)
~~~

~~~sh
python -m scripts.prepare_breeze_research_dataset export \
 --database /private/breeze/research/nifty_candles.sqlite \
 --output /private/breeze/research/identified_CE_5m.csv \
 --day 2025-02-03 --product options \
 --expiry 2025-02-27 --strike 22500 --side CE --interval 5
~~~

The export date/strike is an **example**, not a guarantee that the contract exists. Choose only a contract actually returned by the private catalogue. CSV contains start_ist and **available_at_ist**; trades cannot use a bar's closing information before its available_at_ist.

## Hard validation

- Outer history.cms.sha256 must match encrypted bytes; recipient CMS decryption must succeed; tar entries must have safe paths and permitted types; inner SHA256SUMS must check every source file. Unexpected entries or mutations fail.
- Imported JSON requires Breeze provenance, valid product/date/expiry/side/strike, matching provider_rows and candles SHA256, timezone-aware IST stamps and valid OHLC/volume/OI. Duplicate days must have identical candle hashes. Source rows are never overwritten.
- Exports reject incomplete 5m/15m windows. No forward fill/interpolation. 2025-10-21 Muhurat uses a 13:45-14:45 derivatives session: anomalous 14:45 boundary records excluded with explicit count. The mis-timestamped special-session spot index series is quarantined; no joint spot-futures study may consume it.
- 2025-02-01 Budget special-session intraday remains unverified; 29 actual NSE holidays do not count as ordinary missing sessions. The historical source timestamp/start-of-bar convention is not independently verified.
- Derived SQLite and CSV are created private (0600) and never sent to GitHub/production. Exact reconstructed row counts must be compared to the prior 1,454,867-bar source audit, accounting for explicit quarantines, before claiming the actual 25-window import succeeded.
- Existing replay/risk/strategy thresholds stay unchanged. Never claim candle OHLC fills are executable quotes or promote research automatically.

## Code and acceptance checks

~~~sh
python -m pytest -q \
 tests/scripts/test_prepare_breeze_research_dataset.py \
 tests/scripts/test_unseal_breeze_private_windows.py \
 tests/backtests/test_breeze_bar_research.py
python scripts/agent_task.py check --files \
 scripts/prepare_breeze_research_dataset.py \
 scripts/unseal_breeze_private_windows.py \
 tests/scripts/test_prepare_breeze_research_dataset.py \
 tests/scripts/test_unseal_breeze_private_windows.py
~~~

**Synthetic tests and GitHub CI do not validate the owner's actual 25 archives.** The full conversion requires owner-only private-key access on a trusted research machine.

## NSE/alternative execution data decision

Published NSE domestic schedule effective 1 April 2026 lists F&O historical trade data at ₹1,10,000 per annum/site and historical **order and trade** data at ₹12,50,000 per annum/site, excluding taxes. These are **not** an individual two-year quote. Before paying request a subset/sample, individual-research concession, exact bid/ask reconstruction proof, contract-master access and licensed local backtesting rights. Trade prints alone cannot reproduce order-book execution.

Official price document (page 8):
https://nsearchives.nseindia.com/web/mediaattachment/2026-04/Download_Pricing_file_-_Domestic_clients_20260424122229.pdf
